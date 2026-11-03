"""Встроенная нейросеть сеточного бота: учится на ходу предсказывать доходность нейтральной сетки.

Скальпинг — значит, важен только текущий рынок:
  - в памяти только последние --nn-window-hours часов минутных свечей (никакой старой истории);
  - каждую минуту бот дочитывает новые свечи и создаёт «моменты рынка» по монетам из сканера и самым ликвидным;
  - в каждый момент нейросеть сразу выбирает настройки сетки (до того, как известен итог),
    а бот прогоняет по будущим свечам её выбор, обычные настройки и несколько случайных;
  - как только итог сетки стал известен (через 20…90 минут — сколько сетка живёт), он идёт в обучение;
  - дообучение каждую минуту, полное переобучение раз в --nn-full-every минут;
  - свежие минуты важнее (вес вдвое меньше каждые --nn-half-life-min минут);
  - реальные сетки на бирже идут в обучение с весом ×3.

Честность (онлайн-проверка): прогноз фиксируется до итога. Решения сеть принимает, только если за последний
час её выбор по факту в плюсе и лучше обычных настроек.
"""

import collections
import json
import logging
import math
import os
import random
import shutil
import threading
import time
import zlib

import numpy as np

log = logging.getLogger("grid")
HORIZON_MIN = 90                      # самая долгая сетка, мин (скальпинг до 1.5 часа)
# Всё, что нейросеть настраивает в сетке. Сеть оценивает каждую комбинацию и берёт лучшую.
CONFIG_SPACE = {
    "mult": [0.6, 1.0, 1.6, 2.5],     # шаг: ×0.6 частая … ×2.5 очень редкая (уровней меньше)
    "wmult": [0.7, 1.0, 1.5],         # ширина диапазона сетки
    "geo": [0, 1],                    # 1 = геометрическая (равный шаг в %)
    "lotk": [1.0, 1.2, 1.5],          # рост лота на каждый уровень от цены
    "shift": [1, 2, 4],               # идти за ценой, когда она дальше края на N шагов
    "stop": [3, 6, 10],               # стоп по убытку, % от объёма позиции
    "hold": [10, 20, 40, 60, 90],     # сколько минут держать сетку (10 — итог узнаю быстрее всего)
    "tp": [0, 0.3, 0.6, 1.0],         # тейк-профит сетки, % от объёма позиции (0 = нет)
}
CONFIG_KEYS = list(CONFIG_SPACE)
MARKET = ["vol1m", "atr", "er", "width", "log_turnover", "ret_win", "ret_5", "pos_range", "atr_trend"]
FEATURES = MARKET + ["step", "levels"] + CONFIG_KEYS
# как признаки называть в «мыслях» нейросети
FEATURE_RU = {
    "vol1m": "размах за минуту", "atr": "средняя волатильность", "er": "трендовость (ER)", "step": "шаг сетки",
    "levels": "число уровней", "width": "ширина диапазона рынка", "log_turnover": "объём торгов",
    "ret_win": "движение за 15 мин", "ret_5": "движение за 5 мин", "pos_range": "место цены в диапазоне",
    "atr_trend": "ускорение волатильности", "mult": "частота сетки", "wmult": "ширина сетки",
    "geo": "геометрическая сетка", "lotk": "рост лота от цены", "shift": "когда идти за ценой",
    "stop": "стоп по убытку", "hold": "время сетки", "tp": "тейк-профит сетки",
}


def all_configs():
    out = [{}]
    for k in CONFIG_KEYS:
        out = [{**c, k: v} for c in out for v in CONFIG_SPACE[k]]
    return out


ALL_CONFIGS = all_configs()


def default_cfg(a):
    """Обычные настройки из BOT_ARGS — то, что бот делает без нейросети."""
    return {"mult": 1.0, "wmult": 1.0, "geo": 0, "lotk": 1.0, "shift": float(a.shift_steps),
            "stop": round(a.max_loss / max(a.leverage, 1), 4) if a.max_loss > 0 else 100.0,
            "hold": float(min(HORIZON_MIN, a.max_minutes)), "tp": 0.0}


CFG_ARR = np.array([[c[k] for k in CONFIG_KEYS] for c in ALL_CONFIGS], float)
FIDX = {k: i for i, k in enumerate(FEATURES)}


def feature_matrix(base, cfg_idx, max_levels):
    """Признаки сразу для многих комбинаций настроек (то же, что variant(), но быстро)."""
    C = CFG_ARR[cfg_idx]
    levels = np.clip(np.round(base["grids"] / C[:, CONFIG_KEYS.index("mult")]), 2, max_levels)
    X = np.empty((len(C), len(FEATURES)))
    for k in MARKET:
        X[:, FIDX[k]] = base[k]
    X[:, FIDX["levels"]] = levels / 40.0
    X[:, FIDX["step"]] = base["width"] * C[:, CONFIG_KEYS.index("wmult")] / levels
    for j, k in enumerate(CONFIG_KEYS):
        X[:, FIDX[k]] = C[:, j]
    return X, levels.astype(int)


def grid_bounds(lower, upper, wmult):
    """Границы сетки с учётом множителя ширины — так же, как в боте."""
    if wmult == 1.0:
        return lower, upper
    mid = (lower + upper) / 2
    return max(mid - (mid - lower) * wmult, mid * 0.5), mid + (upper - mid) * wmult


def cfg_short(cfg, levels=None):
    return (f"{'геом.' if cfg.get('geo') else 'обычн.'}{f' {levels} ур.' if levels else ''}, ширина ×{cfg['wmult']:g}, "
            f"лот {'равный' if cfg['lotk'] == 1 else '×%g' % cfg['lotk']}, сдвиг {cfg['shift']:g}, "
            f"стоп {cfg['stop']:g}%, тейк {('%g%%' % cfg['tp']) if cfg.get('tp') else 'нет'}, {cfg['hold']:g} мин")


# ---------------------------------------------------------------- признаки

def base_features(kl, m):
    """kl: свечи окна сканера [start, o, h, l, c, v, turnover] (строки или числа), m: результат analyze()."""
    o = [float(k[1]) for k in kl]
    h = [float(k[2]) for k in kl]
    l = [float(k[3]) for k in kl]
    c = [float(k[4]) for k in kl]
    n = len(kl)
    lo, hi = min(l), max(h)
    rng5 = [(h[i] - l[i]) / o[i] * 100 for i in range(max(0, n - 5), n)]
    return {
        "vol1m": m["vol1m"], "atr": m["atr"], "er": m["er"],
        "width": (m["upper"] - m["lower"]) / m["price"] * 100,
        "log_turnover": math.log10(max(m["turnover"], 1.0)),
        "ret_win": (c[-1] / c[0] - 1) * 100,
        "ret_5": (c[-1] / c[max(0, n - 6)] - 1) * 100,
        "pos_range": (c[-1] - lo) / (hi - lo) if hi > lo else 0.5,
        "atr_trend": (sum(rng5) / len(rng5)) / m["atr"] if m["atr"] > 0 else 1.0,
        "grids": m["grids"],
    }


def variant(base, cfg, max_levels):
    """Признаки рынка + настройки сетки. Возвращает (признаки, число уровней)."""
    levels = max(2, min(max_levels, int(round(base["grids"] / cfg["mult"]))))
    f = {k: base[k] for k in MARKET}
    f["levels"] = levels / 40.0
    f["step"] = base["width"] * cfg["wmult"] / levels
    for k in CONFIG_KEYS:
        f[k] = float(cfg[k])
    return f, levels


def vec(f):
    return [float(f[k]) for k in FEATURES]


# ---------------------------------------------------------------- симулятор сетки по свечам

def simulate_grid(future, price, lower, upper, n, fee_pct, leverage, max_loss_pct, shift_steps=1.0, trace=None,
                  geo=False, lotk=1.0, stop_pct=None, hold=None, tp_pct=0.0):
    """Прогон нейтральной сетки бота по минутным свечам future [(o,h,l,c), ...].

    Те же правила, что у бота: geo — равный шаг в %, lotk — рост лота на каждый уровень от цены,
    встречный ордер ставится тем же объёмом, что исполнился; сдвиг за ценой через shift_steps шагов;
    стоп stop_pct % от объёма (или max_loss_pct % маржи); держим hold минут.
    Внутри свечи цена идёт open→low→high→close (если свеча зелёная) или open→high→low→close.
    Возвращает доходность в % от объёма позиции (маржа × плечо).
    """
    width = upper - lower
    total = 1.0 / price            # объём всей сетки: 1 USDT по цене входа
    max_pos = total

    def build(lo, hi, px):
        if geo:
            r = (hi / lo) ** (1.0 / n)
            lv = [round(lo * r ** i, 10) for i in range(n + 1)]
        else:
            st = (hi - lo) / n
            lv = [round(lo + i * st, 10) for i in range(n + 1)]
        empty = min(range(n + 1), key=lambda i: abs(lv[i] - px))
        wts = {i: lotk ** abs(i - empty) for i in range(n + 1) if i != empty}
        tw = sum(wts.values())
        want = {i: ((1 if i < empty else -1), total * w / tw) for i, w in wts.items()}   # (1 Buy / -1 Sell, объём)
        return lv, want, (hi - lo) / n

    lv, want, step = build(lower, upper, price)
    net = avg = realized = 0.0
    if stop_pct is not None:
        stop_at = -stop_pct / 100
    else:
        stop_at = -(max_loss_pct / 100) / max(leverage, 1) if max_loss_pct > 0 else None
    last = price

    def fill(side, px, q):
        nonlocal net, avg, realized
        dq = q * side
        realized -= fee_pct / 100 * px * q
        if net == 0 or (net > 0) == (dq > 0):
            avg = (avg * abs(net) + px * q) / (abs(net) + q)
            net += dq
        else:
            c = min(q, abs(net))
            realized += (px - avg) * c * (1 if net > 0 else -1)
            net += dq
            if abs(net) < 1e-15:
                net, avg = 0.0, 0.0
            elif (net > 0) == (dq > 0):
                avg = px
        if trace is not None:
            trace.append((side, px, round(net / (total / n), 6)))

    def can(side, q):
        return (net * side) + q <= max_pos * 1.0001

    def move(a, b):
        if b < a:      # вниз: исполняются Buy
            for i in sorted([i for i, (s, _) in want.items() if s == 1 and b <= lv[i] < a], reverse=True):
                s, q = want.get(i, (0, 0))
                if s != 1 or not can(1, q):
                    continue
                fill(1, lv[i], q)
                del want[i]
                if i + 1 <= n:
                    want[i + 1] = (-1, q)
        elif b > a:    # вверх: исполняются Sell
            for i in sorted([i for i, (s, _) in want.items() if s == -1 and a < lv[i] <= b]):
                s, q = want.get(i, (0, 0))
                if s != -1 or not can(-1, q):
                    continue
                fill(-1, lv[i], q)
                del want[i]
                if i - 1 >= 0:
                    want[i - 1] = (1, q)

    steps = future if hold is None else future[:int(hold)]
    for (o, h, l, c) in steps:
        path = [o, l, h, c] if c >= o else [o, h, l, c]
        for a, b in zip(path, path[1:]):
            move(a, b)
        last = c
        pnl = realized + (c - avg) * net
        if stop_at is not None and pnl <= stop_at:
            break
        if tp_pct and pnl >= tp_pct / 100:
            break
        edge = step * shift_steps
        if c > lv[-1] + edge or c < lv[0] - edge:
            lv, want, step = build(c - width / 2, c + width / 2, c)
    pnl = realized + (last - avg) * net - fee_pct / 100 * abs(net) * last   # закрытие лимитками
    return pnl * 100


# ---------------------------------------------------------------- нейросеть (MLP на numpy)

class MLP:
    def __init__(self, n_in, hidden=(32, 16), seed=0):
        rng = np.random.default_rng(seed)
        sizes = [n_in, *hidden, 1]
        self.W = [rng.normal(0, math.sqrt(2 / sizes[i]), (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]

    def forward(self, X):
        acts = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            acts.append(np.maximum(z, 0) if i < len(self.W) - 1 else z)
        return acts

    def predict(self, X):
        return self.forward(X)[-1][:, 0]

    def copy(self):
        m = MLP.__new__(MLP)
        m.W = [W.copy() for W in self.W]
        m.b = [b.copy() for b in self.b]
        return m

    def fit(self, X, y, w, Xv, yv, epochs=60, lr=2e-3, batch=256, l2=5e-4, delta=1.0, seed=0, on_epoch=None,
            patience=12, restore_best=True):
        """Adam + Huber loss. Возвращает кривые ошибки по эпохам; оставляет лучшие по проверке веса.
        Ранняя остановка: если проверка не улучшалась `patience` эпох — дальше не учимся (это заучивание шума)."""
        rng = np.random.default_rng(seed)
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        t = 0
        curve = []
        best = (float("inf"), None)
        since_best = 0

        def huber(pred, target, wt):
            r = pred - target
            a = np.abs(r)
            loss = np.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))
            return float(np.sum(loss * wt) / np.sum(wt))

        for ep in range(epochs):
            idx = rng.permutation(len(X))
            for s in range(0, len(X), batch):
                bi = idx[s:s + batch]
                xb, yb, wb = X[bi], y[bi], w[bi]
                acts = self.forward(xb)
                r = acts[-1][:, 0] - yb
                g = np.clip(r, -delta, delta) * wb / np.sum(wb)
                grad = g[:, None]
                gW, gb = [None] * len(self.W), [None] * len(self.b)
                for i in range(len(self.W) - 1, -1, -1):
                    gW[i] = acts[i].T @ grad + l2 * self.W[i]
                    gb[i] = grad.sum(0)
                    if i > 0:
                        grad = (grad @ self.W[i].T) * (acts[i] > 0)
                t += 1
                for k, (p, gp) in enumerate(zip(params, gW + gb)):
                    m[k] = 0.9 * m[k] + 0.1 * gp
                    v[k] = 0.999 * v[k] + 0.001 * gp * gp
                    p -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
            tl = huber(self.predict(X), y, w)
            vl = huber(self.predict(Xv), yv, np.ones(len(yv))) if len(Xv) else tl
            curve.append({"epoch": ep + 1, "train": tl, "val": vl})
            if vl < best[0]:
                best = (vl, ([W.copy() for W in self.W], [b.copy() for b in self.b]))
                since_best = 0
            else:
                since_best += 1
            if on_epoch:
                on_epoch(ep + 1, epochs)
            if patience and since_best >= patience:
                break
        if restore_best and best[1]:
            self.W, self.b = best[1]
        return curve

    def to_json(self):
        return {"W": [W.tolist() for W in self.W], "b": [b.tolist() for b in self.b]}

    @classmethod
    def from_json(cls, d):
        m = cls.__new__(cls)
        m.W = [np.array(W) for W in d["W"]]
        m.b = [np.array(b) for b in d["b"]]
        return m


# ---------------------------------------------------------------- «мозг»: учёба на ходу, решения

class Brain:
    def __init__(self, path, a, market, analyze_fn, klines_fn=None, symbols_fn=None):
        self.path = path
        self.a = a
        self.market = market            # HTTP-клиент для рыночных данных (pybit, реальный рынок)
        self.analyze = analyze_fn
        self.symbols = symbols_fn or (lambda: [])
        self.lock = threading.RLock()
        nf = len(FEATURES)
        self.X = np.zeros((0, nf), np.float32)
        self.y = np.zeros(0, np.float32)
        self.w = np.zeros(0, np.float32)
        self.t = np.zeros(0, np.float64)
        self.kind = np.zeros(0, np.int8)            # 0 учебная, 1 реальная
        self.g = np.zeros(0, np.int64)              # номер момента рынка
        self.d = np.zeros(0, np.int8)               # 1 обычные настройки, 2 выбор сети, 0 случайные
        self._buf = []
        self.rng = random.Random(7)
        self.allowed = np.array([i for i, c in enumerate(ALL_CONFIGS) if c["hold"] <= a.nn_max_hold])
        self.K = {}                     # монета -> минутные свечи (n, 7) за последние часы
        self.made = {}                  # монета -> время последнего созданного момента
        self.pending = []               # моменты рынка, итог которых ещё не известен
        self.evals = collections.deque(maxlen=50000)   # онлайн-проверка: выбор сети vs обычные, по факту
        self.focus = []                 # монеты из последнего скана — учусь в первую очередь на них
        self.watch = None               # монета работающей сетки — учусь и на ней
        self._liquid = ([], 0.0)
        self.model, self.mu, self.sd = None, None, None
        self.trained_until = 0.0        # самое свежее время, которое модель уже видела
        self.active = False
        self.val_active = False
        self.history = []               # шаги обучения (раз в минуту)
        self.curve = []                 # кривая последнего полного переобучения
        self.status_text = "загрузка"
        self.progress = ""
        self.last_full = 0.0
        self.last_step = 0.0
        self.last_save = 0.0
        self.last_summary = time.time()
        self.cnt = collections.Counter()
        self.new_since_update = 0
        self.new_err = collections.deque(maxlen=3000)  # ошибка прогноза на ещё не виденных сетках
        self.regime = None              # текущее состояние рынка: волатильность и трендовость (медиана по монетам)
        self.regime_hold = False        # рынок сменился — сеть не решает, пока не наберёт итоги нового режима
        self.regime_start = 0.0         # когда сменился рынок (мс)
        self.regime_quiet = 0.0         # до какого времени не поднимать новую тревогу
        self.regime_retrained = True
        self.regime_streak = 0
        self.preds = []
        self.thoughts = collections.deque(maxlen=400)
        self.importance = []
        self._last_decision = ("", 0.0)
        os.makedirs(path, exist_ok=True)
        self.load()
        old = os.path.join(path, "hist")
        self.removed_hist = os.path.isdir(old)
        if self.removed_hist:
            shutil.rmtree(old, ignore_errors=True)   # кэш месячной истории больше не нужен

    # --- хранение
    def n(self):
        return len(self.y) + len(self._buf)

    def _flush(self):
        with self.lock:
            if not self._buf:
                return
            xs, ys, ws, ts, ks, gs, ds = zip(*self._buf)
            self._buf = []
            self.X = np.concatenate([self.X, np.array(xs, np.float32)])
            self.y = np.concatenate([self.y, np.array(ys, np.float32)])
            self.w = np.concatenate([self.w, np.array(ws, np.float32)])
            self.t = np.concatenate([self.t, np.array(ts, np.float64)])
            self.kind = np.concatenate([self.kind, np.array(ks, np.int8)])
            self.g = np.concatenate([self.g, np.array(gs, np.int64)])
            self.d = np.concatenate([self.d, np.array(ds, np.int8)])

    def _trim(self, now):
        """Только свежий рынок: учебные примеры — за последние --nn-window-hours часов, реальные сетки — за сутки."""
        with self.lock:
            keep = ((self.t >= now - self.a.nn_window_hours * 3_600_000) |
                    ((self.kind == 1) & (self.t >= now - 86_400_000)))
            if len(keep) - keep.sum() > 0:
                self.X, self.y, self.w, self.t, self.kind, self.g, self.d = (
                    self.X[keep], self.y[keep], self.w[keep], self.t[keep], self.kind[keep], self.g[keep], self.d[keep])
            cap = self.a.nn_max_samples
            if len(self.y) > cap:
                order = np.argsort(self.t)[-cap:]
                order.sort()
                self.X, self.y, self.w, self.t, self.kind, self.g, self.d = (
                    self.X[order], self.y[order], self.w[order], self.t[order], self.kind[order], self.g[order],
                    self.d[order])

    def save(self):
        try:
            self._flush()
            with self.lock:
                tmp = os.path.join(self.path, "data.npz.tmp")
                with open(tmp, "wb") as f:
                    np.savez_compressed(f, X=self.X, y=self.y, w=self.w, t=self.t, kind=self.kind, g=self.g, d=self.d)
                os.replace(tmp, os.path.join(self.path, "data.npz"))
                meta = {"history": self.history[-600:], "curve": self.curve, "active": self.active,
                        "val_active": self.val_active,
                        "model": self.model.to_json() if self.model else None,
                        "mu": self.mu.tolist() if self.mu is not None else None,
                        "sd": self.sd.tolist() if self.sd is not None else None, "features": FEATURES,
                        "thoughts": list(self.thoughts)[:200], "importance": self.importance,
                        "trained_until": self.trained_until, "evals": list(self.evals)[-6000:],
                        "made": self.made, "pending": self.pending, "regime_hold": self.regime_hold,
                        "regime_start": self.regime_start, "regime_quiet": self.regime_quiet,
                        "regime_retrained": self.regime_retrained}
            tmp = os.path.join(self.path, "brain.json.tmp")
            with open(tmp, "w") as f:
                json.dump(meta, f, default=float)
            os.replace(tmp, os.path.join(self.path, "brain.json"))
            self.last_save = time.time()
        except Exception as e:  # noqa: BLE001
            log.warning("Нейросеть: не сохранилась: %s", e)

    def load(self):
        try:
            with open(os.path.join(self.path, "brain.json")) as f:
                d = json.load(f)
        except FileNotFoundError:
            return
        except Exception as e:  # noqa: BLE001
            log.warning("Нейросеть: файл не прочитан: %s", e)
            return
        if d.get("features") != FEATURES:
            log.info("Нейросеть: признаки изменились — начинаю обучение заново")
            return
        try:
            f = os.path.join(self.path, "data.npz")
            if os.path.exists(f):
                z = np.load(f)
                self.X, self.y, self.w, self.t, self.kind = z["X"], z["y"], z["w"], z["t"], z["kind"]
                self.g, self.d = z["g"], z["d"]
            self.history = [h for h in d.get("history", []) if "src" in h]
            self.curve = d.get("curve", [])
            if d.get("model"):
                self.model = MLP.from_json(d["model"])
                self.mu, self.sd = np.array(d["mu"]), np.array(d["sd"])
            self.thoughts.extend(d.get("thoughts") or [])
            self.importance = d.get("importance") or []
            # старая модель без отметки «что уже видела» — считаем, что видела всё до сейчас (честная проверка)
            self.trained_until = float(d.get("trained_until") or time.time() * 1000)
            self.evals.extend(d.get("evals") or [])
            # моменты рынка, созданные до перезапуска: не создаём их заново (без дублей) и дожидаемся их итога
            self.made = {k: float(v) for k, v in (d.get("made") or {}).items()}
            self.pending = d.get("pending") or []
            self.regime_hold = bool(d.get("regime_hold", False))
            self.regime_start = float(d.get("regime_start") or 0.0)
            self.regime_quiet = float(d.get("regime_quiet") or 0.0)
            self.regime_retrained = bool(d.get("regime_retrained", True))
            self.status_text = f"загружено {len(self.y)} примеров"
            log.info("Нейросеть: загружено %d примеров", len(self.y))
        except Exception as e:  # noqa: BLE001
            log.warning("Нейросеть: данные не прочитаны (%s) — начинаю заново", e)

    # --- мысли
    def think(self, kind, title, lines=None):
        """Запись в журнал мыслей. kind: data | train | decide | result | plan | warn."""
        self.thoughts.appendleft({"t": int(time.time() * 1000), "kind": kind, "title": title, "lines": lines or []})
        log.info("Нейросеть [%s]: %s", kind, title)

    def explain(self, f):
        """Три признака, которые сильнее всего сдвинули прогноз: подставляем вместо признака
        его среднее значение и смотрим, насколько меняется прогноз."""
        if not self.model:
            return []
        x = (np.array([vec(f)]) - self.mu) / self.sd
        base = float(self.model.predict(x)[0])
        X = np.repeat(x, len(FEATURES), axis=0)
        for k in range(len(FEATURES)):
            X[k, k] = 0.0
        deltas = base - self.model.predict(X)
        order = np.argsort(-np.abs(deltas))[:3]
        return [(FEATURES[k], float(deltas[k]), float(f[FEATURES[k]])) for k in order]

    def plan_lines(self):
        a = self.a
        nxt_full = self.last_full + a.nn_full_every * 60
        out = ["Каждую минуту: дочитываю свежие свечи, фиксирую свой выбор в новых моментах рынка и дообучаюсь "
               "на сетках, итог которых только что стал известен",
               (f"Полное переобучение: ~{time.strftime('%H:%M UTC', time.gmtime(nxt_full))} (раз в {a.nn_full_every:g} мин)"
                if self.last_full else "Полное переобучение: как только наберу 300 примеров"),
               f"Память: только последние {a.nn_window_hours:g} ч рынка; свежие минуты важнее "
               f"(вес вдвое меньше каждые {a.nn_half_life_min:g} мин)",
               f"Итог сетки узнаю, когда она отработает: короткой — через 10 мин, длинной — через {a.nn_max_hold} мин; "
               f"сейчас жду итог {len(self.pending):,} моментов"]
        if self.regime_hold:
            n_post = sum(1 for e in self.evals if e["t"] >= self.regime_start and "net" in e)
            out.append(f"Рынок сменился в {time.strftime('%H:%M UTC', time.gmtime(self.regime_start / 1000))}: "
                       f"решает бот с обычными настройками; жду итогов своих решений в новом режиме — "
                       f"{n_post} из {a.nn_regime_min_n}")
        elif not self.active:
            out.append(f"Чтобы включиться: за последние {a.nn_eval_min:g} мин мой выбор по факту в плюсе и лучше "
                       f"обычных настроек, минимум {a.nn_eval_min_n} сделок")
        else:
            out.append("Сейчас я выбираю монету, все настройки сетки и маржу; если онлайн-проверка станет хуже "
                       "обычных настроек — сама отключусь")
        return out

    @staticmethod
    def setting_effects(X, y):
        """Средний итог по каждой настройке: какое значение лучше шло (для мыслей)."""
        out = []
        names = {"mult": "частота ×", "wmult": "ширина ×", "geo": "", "lotk": "лот ×", "shift": "сдвиг через ",
                 "stop": "стоп ", "hold": "", "tp": "тейк "}
        for k in CONFIG_KEYS:
            col = X[:, FEATURES.index(k)]
            best = None
            for v in CONFIG_SPACE[k]:
                sel = np.isclose(col, v)
                if sel.sum() >= 20:
                    m = float(y[sel].mean())
                    best = (v, m) if best is None or m > best[1] else best
            if best:
                v, m = best
                label = ("геометрическая" if v else "обычная") if k == "geo" else (
                    f"{v:g} мин" if k == "hold" else f"{v:g}%" if k in ("stop", "tp") and v else
                    "нет" if k == "tp" else f"{v:g}")
                out.append(f"{names[k]}{label} ({m:+.3f}%)")
        return out

    # --- данные
    def add(self, feats, target, kind, t_ms, weight, g=0, d=0):
        with self.lock:
            self._buf.append((vec(feats), float(max(-10.0, min(10.0, target))), float(weight), float(t_ms), kind,
                              int(g), int(d)))
            self.new_since_update += 1

    def add_real(self, feats, target, pred=None, sym=""):
        """Закрытая сетка на бирже: доходность в % от объёма позиции."""
        if not feats:
            return
        self.add(feats, target, 1, time.time() * 1000, 3.0, g=-int(time.time() * 1000), d=1)
        lines = [f"Итог: {target:+.3f}% от объёма позиции"]
        if pred is not None:
            err = target - pred
            lines.append(f"Мой прогноз был {pred:+.3f}% — ошиблась на {err:+.3f}% "
                         + ("(оказалось лучше, чем думала)" if err > 0 else "(оказалось хуже, чем думала)"))
        else:
            lines.append("Прогноза не было — сетку ставил бот с обычными настройками")
        self._flush()
        lines.append(f"Учу сразу, с весом ×3. Реальных сеток за сутки: {int((self.kind == 1).sum())}")
        self.think("result", f"{sym}: разбираю закрытую реальную сетку", lines)

    def _kline_page(self, sym, start, end):
        for i in range(4):
            try:
                r = self.market.get_kline(category="linear", symbol=sym, interval="1", limit=1000,
                                          start=int(start), end=int(end))
                return r["result"]["list"]
            except Exception as e:  # noqa: BLE001
                if i == 3:
                    raise
                log.warning("Нейросеть: свечи %s (%s), повтор", sym, str(e)[:100])
                time.sleep(1.0 * (i + 1))

    def _fetch(self, sym, start, end):
        """Минутные свечи [start, end] постранично (Bybit отдаёт до 1000 за раз, новые первыми)."""
        rows = []
        while end >= start:
            lst = self._kline_page(sym, start, end)
            if not lst:
                break
            rows.extend(lst)
            oldest = int(lst[-1][0])
            if oldest <= start or len(lst) < 1000:
                break
            end = oldest - 1
        if not rows:
            return np.zeros((0, 7))
        return np.array([[float(x) for x in k[:7]] for k in rows])

    def learn_symbols(self):
        """Монеты для учёбы: сначала те, что сейчас дал сканер, потом самые ликвидные."""
        now = time.time()
        if now - self._liquid[1] > 600:
            try:
                self._liquid = (self.symbols(), now)
            except Exception as e:  # noqa: BLE001
                log.warning("Нейросеть: список монет: %s", e)
        first = ([self.watch] if self.watch else []) + list(self.focus)
        return list(dict.fromkeys(first + list(self._liquid[0])))[: self.a.nn_symbols]

    def refresh_candles(self, sym, now):
        a = self.a
        keep_min = a.nn_window_hours * 60 + HORIZON_MIN + a.window + 10
        start = now - keep_min * 60_000
        K = self.K.get(sym)
        frm = start if K is None or not len(K) else K[-1, 0] + 60_000
        if frm <= now - 60_000:
            new = self._fetch(sym, frm, now)
            if len(new):
                K = new if K is None or not len(K) else np.concatenate([K, new])
        if K is None:
            K = np.zeros((0, 7))
        if len(K):
            _, idx = np.unique(K[:, 0], return_index=True)
            K = K[idx]
            K = K[(K[:, 0] >= start) & (K[:, 0] + 60_000 <= now)]       # только закрытые свечи
        self.K[sym] = K

    def make_moments(self, sym, now, model):
        """Новые моменты рынка по монете. В каждом сеть сразу фиксирует свой выбор (до того, как известен итог)."""
        a = self.a
        K = self.K.get(sym)
        win = a.window + 1
        if K is None or len(K) < win:
            return 0
        tm = K[:, 0]
        start_e = max(win, int(np.searchsorted(tm, self.made.get(sym, 0), side="right")) + 1)
        oldest = now - a.nn_window_hours * 3_600_000
        dflt = default_cfg(a)
        mdl, mu, sd = model
        made = 0
        for e in range(start_e, len(K) + 1):
            t_end = tm[e - 1]
            if t_end < oldest or int(t_end // 60_000) % a.nn_stride:
                continue
            if tm[e - 1] - tm[e - win] != (win - 1) * 60_000:
                continue
            window = K[e - win:e]
            m = self.analyze(window, a)
            if not m or m["vol1m"] < a.min_1m * 0.3:
                continue
            base = base_features(window, m)
            items = [{"cfg": dflt, "flag": 1}]
            items += [{"cfg": ALL_CONFIGS[i], "flag": 0} for i in self.rng.sample(list(self.allowed), a.nn_configs)]
            rec = {"sym": sym, "t": float(t_end), "gid": int(t_end // 60_000) * 65536 + (zlib.crc32(sym.encode()) & 0xFFFF),
                   "price": m["price"], "lower": m["lower"], "upper": m["upper"], "base": base, "items": items,
                   "chosen": None, "pred": None, "took": None}
            if mdl is not None and t_end > self.trained_until:
                X, _ = feature_matrix(base, self.allowed, a.max_levels)
                p = mdl.predict((X - mu) / sd)
                k = int(np.argmax(p))
                cfg = ALL_CONFIGS[int(self.allowed[k])]
                rec["pred"], rec["took"] = float(p[k]), bool(p[k] > a.nn_min_edge)
                if cfg == dflt:
                    rec["chosen"] = 0
                else:
                    items.append({"cfg": cfg, "flag": 2})
                    rec["chosen"] = len(items) - 1
            self.pending.append(rec)
            made += 1
        self.made[sym] = float(tm[-1])
        return made

    def resolve(self, now):
        """Итоги, которые уже известны: прогоняем сетки по свечам после момента и учим."""
        a = self.a
        keep, resolved = [], 0
        mdl = self.model
        for rec in self.pending:
            K = self.K.get(rec["sym"])
            if K is not None and len(K):
                tm = K[:, 0]
                i0 = int(np.searchsorted(tm, rec["t"] + 60_000))
                for it in rec["items"]:
                    if "r" in it:
                        continue
                    h = int(it["cfg"]["hold"])
                    if i0 + h > len(K):
                        continue
                    if tm[i0] != rec["t"] + 60_000 or tm[i0 + h - 1] - tm[i0] != (h - 1) * 60_000:
                        it["r"] = None            # дыра в свечах
                        continue
                    cfg = it["cfg"]
                    f, levels = variant(rec["base"], cfg, a.max_levels)
                    lo, hi = grid_bounds(rec["lower"], rec["upper"], cfg["wmult"])
                    r = simulate_grid(K[i0:i0 + h, 1:5], rec["price"], lo, hi, levels, a.fee, a.leverage, 0,
                                      cfg["shift"], geo=bool(cfg["geo"]), lotk=cfg["lotk"], stop_pct=cfg["stop"],
                                      tp_pct=cfg.get("tp", 0),
                                      hold=h)
                    it["r"] = r
                    if mdl is not None and rec["t"] > self.trained_until:
                        pz = float(mdl.predict((np.array([vec(f)]) - self.mu) / self.sd)[0])
                        self.new_err.append(abs(pz - r))
                    self.add(f, r, 0, rec["t"], 1.0, g=rec["gid"], d=it["flag"])
                    resolved += 1
            # онлайн-проверка: итог выбора сети засчитываем сразу, как только её сетка отработала
            # (10-минутная — через 10 мин), итог обычных настроек — когда отработали они
            c = rec["chosen"]
            if c is not None and not rec.get("nlog") and "r" in rec["items"][c]:
                rec["nlog"] = True
                if rec["items"][c]["r"] is not None:
                    self.evals.append({"t": rec["t"], "tr": now, "sym": rec["sym"], "took": rec["took"],
                                       "net": rec["items"][c]["r"], "pred": rec["pred"]})
                    self.cnt["evals"] += 1
            if not rec.get("dlog") and "r" in rec["items"][0]:
                rec["dlog"] = True
                if rec["items"][0]["r"] is not None:
                    self.evals.append({"t": rec["t"], "tr": now, "dflt": rec["items"][0]["r"]})
            if all("r" in it for it in rec["items"]):
                pass
            elif now - rec["t"] <= (HORIZON_MIN + 30) * 60_000:
                keep.append(rec)
        self.pending = keep
        return resolved

    def online_metrics(self, now, minutes, since=None):
        """Онлайн-проверка: что дал выбор сети по факту против обычных настроек за последние `minutes` мин.
        since — учитывать только решения, принятые после смены рынка."""
        W = [e for e in self.evals if e["tr"] >= now - minutes * 60_000 and (since is None or e["t"] >= since)]
        E = [e for e in W if "net" in e]            # итоги выбора сети
        D = [e for e in W if "dflt" in e]           # итоги обычных настроек
        taken = [e for e in E if e["took"]]
        out = {"n": len(E), "n_dflt": len(D), "sel_n": len(taken),
               "sel_avg": float(np.mean([e["net"] for e in taken])) if taken else 0.0,
               "base_avg": float(np.mean([e["dflt"] for e in D])) if D else 0.0,
               "policy": float(np.mean([e["net"] if e["took"] else 0.0 for e in E])) if E else 0.0, "corr": 0.0}
        if len(E) > 5:
            pr = np.array([e["pred"] for e in E])
            ac = np.array([e["net"] for e in E])
            if pr.std() > 0 and ac.std() > 0:
                out["corr"] = float(np.corrcoef(pr, ac)[0, 1])
        return out

    # --- режим рынка
    @staticmethod
    def _er(c):
        path = np.abs(np.diff(c)).sum()
        return float(abs(c[-1] - c[0]) / path) if path > 0 else 1.0

    def market_regime(self, syms):
        """Медиана по монетам: волатильность последних 15 мин к предыдущему часу и трендовость сейчас / раньше.
        Медиана — чтобы тревогу поднимала смена всего рынка, а не всплеск одной монеты (их и ищет сканер)."""
        vr, en, eb = [], [], []
        for sym in syms:
            K = self.K.get(sym)
            if K is None or len(K) < 90 or K[-1, 0] - K[-90, 0] != 89 * 60_000:
                continue
            o, h, l, c = K[-90:, 1], K[-90:, 2], K[-90:, 3], K[-90:, 4]
            rng = (h - l) / o * 100
            prev = rng[15:75].mean()                 # волатильность: последние 15 мин к часу перед ними
            if prev <= 0:
                continue
            vr.append(rng[75:].mean() / prev)
            en.append(self._er(c[60:]))              # трендовость: последние 30 мин к двум получасам перед ними
            eb.append(float(np.mean([self._er(c[0:30]), self._er(c[30:60])])))
        if len(vr) < 8:
            return None
        return {"vol_ratio": float(np.median(vr)), "er_now": float(np.median(en)), "er_before": float(np.median(eb)),
                "coins": len(vr)}

    def check_regime(self, now, syms):
        a = self.a
        r = self.market_regime(syms)
        self.regime = r
        if self.regime_hold:
            n_post = sum(1 for e in self.evals if e["t"] >= self.regime_start and "net" in e)
            if n_post >= a.nn_regime_min_n or now - self.regime_start >= a.nn_regime_max * 60_000:
                self.regime_hold = False
                self.regime_quiet = now + 10 * 60_000
                om = self.online_metrics(now, a.nn_eval_min, since=self.regime_start)
                self.think("decide", "Новый режим рынка изучен — возвращаю себе право решать по проверке",
                           [f"Итогов моих решений после смены рынка: {n_post}",
                            f"По ним мой выбор: {om['sel_avg']:+.3f}% против {om['base_avg']:+.3f}% у обычных настроек "
                            f"({om['sel_n']} сделок)" if om["n"] else "Итогов мало — включусь только если пройду проверку",
                            "Онлайн-проверка дальше считается только по новому режиму"])
            return
        if r is None or now < self.regime_quiet:
            self.regime_streak = 0
            return
        vr, de = r["vol_ratio"], r["er_now"] - r["er_before"]
        hit = vr >= a.nn_regime_vol or vr <= 1 / a.nn_regime_vol or abs(de) >= a.nn_regime_er
        self.regime_streak = self.regime_streak + 1 if hit else 0     # сигнал должен держаться N минут подряд
        if self.regime_streak >= a.nn_regime_confirm:
            self.regime_streak = 0
            self.regime_hold, self.regime_start, self.regime_retrained = True, now, False
            what = []
            if vr >= a.nn_regime_vol:
                what.append(f"волатильность выросла ×{vr:.2f}")
            elif vr <= 1 / a.nn_regime_vol:
                what.append(f"волатильность упала до ×{vr:.2f}")
            if abs(de) >= a.nn_regime_er:
                what.append(f"трендовость {r['er_before']:.2f} → {r['er_now']:.2f}")
            self.think("warn", "Рынок сменился — передаю решения обычным настройкам",
                       [f"По {r['coins']} монетам (медиана), держится {a.nn_regime_confirm} мин подряд: " + ", ".join(what),
                        "Мои знания о прошлом режиме могут не работать: пока решает бот с обычными настройками",
                        f"Учу новый режим с весом ×{a.nn_regime_boost:g}; полностью переобучусь, когда наберу 300 его примеров",
                        f"Вернусь, когда наберу минимум {a.nn_regime_min_n} итогов своих решений в новом режиме "
                        f"(не позже чем через {a.nn_regime_max:g} мин); первый час учусь вдвое быстрее"])

    # --- обучение
    def _arrays(self, now):
        self._flush()
        with self.lock:
            X, y, w, t = self.X.astype(float), self.y.astype(float), self.w.astype(float), self.t.copy()
            kind, g, dflt = self.kind.copy(), self.g.copy(), self.d.copy()
        age_min = np.maximum(0.0, (now - t) / 60_000)
        w = w * np.maximum(0.05, 0.5 ** (age_min / max(self.a.nn_half_life_min, 1)))
        if self.regime_start and now - self.regime_start < 2 * 3_600_000:
            w = np.where(t >= self.regime_start, w * self.a.nn_regime_boost, w)    # новый режим рынка важнее
        return X, y, w, t, kind, g, dflt

    def full_train(self, now):
        X, y, w, t, kind, g, dflt = self._arrays(now)
        order = np.argsort(t, kind="stable")
        X, y, w, t, kind, g, dflt = X[order], y[order], w[order], t[order], kind[order], g[order], dflt[order]
        n_val = max(100, int(len(y) * 0.2))
        Xt, yt, wt = X[:-n_val], y[:-n_val], w[:-n_val]
        Xv, yv = X[-n_val:], y[-n_val:]
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
        model = MLP(X.shape[1], seed=len(self.history))

        def prog(ep, n):
            self.progress = f"полное переобучение: эпоха {ep}/{n}"
        batch = 256 if len(yt) < 30_000 else 1024
        curve = model.fit((Xt - mu) / sd, yt, wt, (Xv - mu) / sd, yv, epochs=self.a.nn_epochs, batch=batch,
                          on_epoch=prog)
        # проверка на самом свежем куске, которого модель не видела: в каждом моменте — лучшая по прогнозу комбинация
        pv = model.predict((Xv - mu) / sd)
        gv, dv = g[-n_val:], dflt[-n_val:]
        o = np.lexsort((-pv, gv))
        _, first = np.unique(gv[o], return_index=True)
        best_rows = o[first]
        chosen = best_rows[pv[best_rows] > self.a.nn_min_edge]
        sel_avg = float(yv[chosen].mean()) if len(chosen) else 0.0
        base_avg = float(yv[dv == 1].mean()) if (dv == 1).any() else 0.0
        self.val_active = bool(len(chosen) >= 30 and sel_avg > 0 and sel_avg > base_avg)
        # важность признаков: насколько растёт ошибка, если перемешать признак на проверке
        Zv, yvi = (Xv - mu) / sd, yv
        if len(Zv) > 20000:
            Zv, yvi = Zv[-20000:], yv[-20000:]
        base_err = float(np.mean(np.abs(model.predict(Zv) - yvi)))
        rng = np.random.default_rng(1)
        imp = []
        for k in range(len(FEATURES)):
            Zs = Zv.copy()
            Zs[:, k] = rng.permutation(Zs[:, k])
            imp.append((FEATURES[k], float(np.mean(np.abs(model.predict(Zs) - yvi)) - base_err)))
        imp.sort(key=lambda x: -x[1])
        # для решений в бою — модель, дообученная и на свежем куске
        model.fit((X - mu) / sd, y, w, X[:0], y[:0], epochs=max(1, self.a.nn_online_epochs), lr=1e-3,
                  batch=batch, patience=0, restore_best=False)
        with self.lock:
            self.model, self.mu, self.sd = model, mu, sd
            self.curve = curve
            self.importance = imp
            self.trained_until = float(t.max())
        self.last_full = time.time()
        self.new_since_update = 0
        self.cnt["full"] += 1
        best_ep = min(curve, key=lambda c: c["val"])["epoch"]
        lines = [f"Примеров: {len(y):,} за последние {self.a.nn_window_hours:g} ч, реальных сеток: {int((kind == 1).sum())}",
                 f"Ошибка: на обучении {curve[-1]['train']:.3f}, на свежем куске {min(c['val'] for c in curve):.3f} "
                 f"(лучшая на проходе {best_ep} из {len(curve)})",
                 f"На свежем куске взяла бы {len(chosen):,} из {len(best_rows):,} моментов: в среднем {sel_avg:+.3f}% "
                 f"против {base_avg:+.3f}% у обычных настроек",
                 "Больше всего опираюсь на: " + ", ".join(FEATURE_RU[k] for k, v in imp[:4] if v > 0)]
        self.think("train", "Полное переобучение готово", lines)

    def online_update(self, now):
        """Дообучение каждую минуту: несколько проходов по свежей памяти, начиная с текущих весов."""
        X, y, w, t, _, _, _ = self._arrays(now)
        m2 = self.model.copy()
        fresh = self.regime_start and now - self.regime_start < 3_600_000     # первый час нового режима — учусь вдвое быстрее
        m2.fit((X - self.mu) / self.sd, y, w, X[:0], y[:0], epochs=self.a.nn_online_epochs * (2 if fresh else 1),
               lr=self.a.nn_online_lr * (2 if fresh else 1),
               batch=512, patience=0, restore_best=False, seed=self.cnt["updates"])
        with self.lock:
            self.model = m2
            self.trained_until = float(t.max())
        self.new_since_update = 0
        self.cnt["updates"] += 1

    # --- шаг учёбы (раз в минуту)
    def step(self):
        a = self.a
        now = time.time() * 1000
        first = not self.K
        syms = self.learn_symbols()
        if first:
            self.think("data", f"Беру свежий рынок: последние {a.nn_window_hours:g} ч минутных свечей по {len(syms)} монетам",
                       ["Старую историю не беру — для скальпинга важен только текущий рынок",
                        f"Каждую минуту — новый момент рынка по каждой монете; в нём я прогоняю обычные настройки, "
                        f"{a.nn_configs} случайные и свой выбор по свечам, которые идут после"])
        with self.lock:
            model = (self.model, self.mu, self.sd)
        need = set(syms) | {r["sym"] for r in self.pending}
        made = 0
        for i, sym in enumerate(sorted(need, key=lambda s: s not in syms)):
            self.progress = f"свечи: {sym} ({i + 1}/{len(need)})"
            try:
                self.refresh_candles(sym, now)
                if sym in syms:
                    made += self.make_moments(sym, now, model)
            except Exception as e:  # noqa: BLE001
                log.warning("Нейросеть: %s: %s", sym, e)
        for sym in [s for s in self.K if s not in need]:
            del self.K[sym]
        self.progress = "разбираю итоги"
        resolved = self.resolve(now)
        self.cnt["moments"] += made
        self.cnt["resolved"] += resolved
        self._flush()
        self._trim(now)
        self.check_regime(now, syms)
        if first and resolved:
            self.think("data", "Свежий рынок загружен",
                       [f"Монет: {len(self.K)}, моментов рынка: {made:,}, сеток с известным итогом: {resolved:,}",
                        "Что сейчас лучше шло: " + "; ".join(self.setting_effects(self.X.astype(float), self.y.astype(float)))])
        if self.n() < 300:
            self.status_text = f"собираю данные ({self.n()} примеров)"
        elif self.model is None or time.time() - self.last_full >= a.nn_full_every * 60:
            self.full_train(now)
        elif self.regime_hold and not self.regime_retrained and int((self.t >= self.regime_start).sum()) >= 300:
            self.think("train", "Набрала 300 примеров нового режима — переобучаюсь полностью")
            self.full_train(now)
            self.regime_retrained = True
        elif self.new_since_update >= 20:
            self.progress = "дообучаюсь"
            self.online_update(now)
        self.progress = ""
        # решение: онлайн-проверка за последний час, пока её мало — проверка при полном переобучении
        since = self.regime_start if self.regime_start and now - self.regime_start < a.nn_eval_min * 60_000 else None
        om = self.online_metrics(now, a.nn_eval_min, since=since)
        was = self.active
        if self.regime_hold:
            self.active, src = False, "regime"
        elif om["sel_n"] >= a.nn_eval_min_n and om["n_dflt"] >= a.nn_eval_min_n:
            self.active, src = bool(om["sel_avg"] > 0 and om["sel_avg"] > om["base_avg"]), "online"
        else:
            self.active, src = self.val_active, "full"
        if self.model is not None:
            mae = float(np.mean(self.new_err)) if self.new_err else None
            self.history.append({"time": int(now), "samples": self.n(), "real": int((self.kind == 1).sum()),
                                 "sel_avg": om["sel_avg"] if om["sel_n"] else None,
                                 "base_avg": om["base_avg"] if om["n_dflt"] else None, "sel_n": om["sel_n"],
                                 "val_n": om["n"], "corr": om["corr"], "active": self.active, "src": src, "mae": mae,
                                 "vol_ratio": self.regime["vol_ratio"] if self.regime else None})
            self.history = self.history[-600:]
        if self.active and not was:
            self.think("decide", "Включаюсь в принятие решений",
                       [f"За последние {a.nn_eval_min:g} мин мой выбор по факту: {om['sel_avg']:+.3f}% в {om['sel_n']} сделках "
                        f"против {om['base_avg']:+.3f}% у обычных настроек" if src == "online" else
                        "Онлайн-проверки пока мало — включилась по проверке на свежем куске при переобучении",
                        "Теперь выбираю монету, все настройки сетки и маржу сама"])
        elif was and not self.active and src != "regime":
            self.think("warn", "Отключаюсь от решений",
                       [f"За последние {a.nn_eval_min:g} мин мой выбор: {om['sel_avg']:+.3f}% против {om['base_avg']:+.3f}% "
                        f"у обычных — бот торгует обычными настройками, я учусь дальше"])
        self.status_text = "ПРИНИМАЕТ РЕШЕНИЯ" if self.active else (
            "рынок сменился — изучаю новый режим" if self.regime_hold else
            "учится на ходу" if self.model is not None else self.status_text)
        if time.time() - self.last_summary >= a.nn_summary_min * 60 and self.model is not None:
            self.summary(now, om)
        if time.time() - self.last_save >= 300:
            self.save()
        self.last_step = time.time()

    def summary(self, now, om):
        a = self.a
        mins = (time.time() - self.last_summary) / 60
        c = self.cnt
        mae = float(np.mean(self.new_err)) if self.new_err else None
        lines = [f"За {mins:.0f} мин: новых моментов рынка {c['moments']:,}, сеток с известным итогом {c['resolved']:,}, "
                 f"дообучений {c['updates']}" + (f", полных переобучений {c['full']}" if c['full'] else ""),
                 f"Учусь на {self.n():,} примерах за последние {a.nn_window_hours:g} ч"
                 + (f"; ошибка прогноза на ещё не виденных сетках {mae:.3f}%" if mae is not None else "")]
        if om["sel_n"]:
            lines.append(f"Онлайн-проверка за {a.nn_eval_min:g} мин: из {om['n']:,} моментов брала бы {om['sel_n']:,} — "
                         f"по факту {om['sel_avg']:+.3f}% против {om['base_avg']:+.3f}% у обычных настроек; "
                         f"связь прогноза с итогом {om['corr']:+.2f}")
        else:
            lines.append("Онлайн-проверка ещё копится: мои первые решения ждут, пока сетки отработают")
        with self.lock:
            recent = self.t >= now - 30 * 60_000 - HORIZON_MIN * 60_000
            Xr, yr = self.X[recent].astype(float), self.y[recent].astype(float)
        if len(yr) >= 100:
            lines.append("Сейчас лучше работает: " + "; ".join(self.setting_effects(Xr, yr)))
        self.think("train", "Учусь на ходу", lines)
        self.cnt.clear()
        self.last_summary = time.time()

    # --- решения
    def evaluate(self, cand):
        """Оценить все комбинации настроек для монеты. Возвращает словарь:
        best/default: (настройки, уровней, прогноз, признаки); top: 3 лучших; size: множитель маржи."""
        base = cand.get("nn_base")
        if not base:
            return None
        dflt = default_cfg(self.a)
        fd, ld = variant(base, dflt, self.a.max_levels)
        with self.lock:
            mdl, mu, sd = self.model, self.mu, self.sd
        if mdl is None:
            return {"best": (dflt, ld, None, fd), "default": (dflt, ld, None, fd), "top": [], "size": 1.0}
        X, levels = feature_matrix(base, self.allowed, self.a.max_levels)
        p = mdl.predict((np.vstack([X, [vec(fd)]]) - mu) / sd)
        pd_ = float(p[-1])
        p = p[:-1]
        order = np.argsort(-p)[:3]
        top = []
        for k in order:
            cfg = ALL_CONFIGS[int(self.allowed[k])]
            f, lv = variant(base, cfg, self.a.max_levels)
            top.append((cfg, lv, float(p[k]), f))
        best = top[0]
        return {"best": best, "default": (dflt, ld, pd_, fd), "top": top, "size": self.size_for(best[2])}

    def reassess(self, kl, cfg):
        """Переоценка работающей сетки: сколько дала бы сетка, начатая прямо сейчас, —
        с её текущими настройками и с лучшими. Это и есть прогноз на остаток пути."""
        with self.lock:
            mdl, mu, sd = self.model, self.mu, self.sd
        if mdl is None:
            return None
        m = self.analyze(kl, self.a)
        if not m:
            return None
        base = base_features(kl, m)
        cur = {**default_cfg(self.a), **(cfg or {})}
        fc, _ = variant(base, cur, self.a.max_levels)
        X, _ = feature_matrix(base, self.allowed, self.a.max_levels)
        p = mdl.predict((np.vstack([X, [vec(fc)]]) - mu) / sd)
        k = int(np.argmax(p[:-1]))
        best_cfg = ALL_CONFIGS[int(self.allowed[k])]
        fb, _ = variant(base, best_cfg, self.a.max_levels)
        return {"best": float(p[k]), "current": float(p[-1]), "best_cfg": cfg_short(best_cfg), "why": self.explain(fb)}

    def size_for(self, pred):
        """Маржа на сетку: больше при сильном прогнозе, меньше при слабом, в пределах --nn-size-min…max."""
        if pred is None or pred <= 0:
            return self.a.nn_size_min
        return float(min(self.a.nn_size_max, max(self.a.nn_size_min, pred / self.a.nn_size_ref)))

    def log_decisions(self, decided, use):
        """Мысли по скану: какие настройки выбрала для каждой монеты и почему."""
        if not decided or not self.model:
            return
        top = decided[0]
        key = f"{top['symbol']}:{top['take']}:{cfg_short(top['cfg'])}"
        if key == self._last_decision[0] and time.time() - self._last_decision[1] < 300:
            return
        self._last_decision = (key, time.time())
        lines = []
        for d in decided[:3]:
            lines.append(f"{d['symbol']}: лучшие настройки — {cfg_short(d['cfg'], d['levels'])} → прогноз {d['pred']:+.3f}%; "
                         f"обычные → {d['pred_default']:+.3f}%")
            if use:
                lines.append(f"   → {'беру, маржа ×%.2f' % d['size'] if d['take'] else 'пропускаю: прогноз ниже порога %+.2f%%' % self.a.nn_min_edge}")
            why = self.explain(d["feats"])
            if why:
                lines.append("   почему: " + "; ".join(
                    f"{FEATURE_RU[k]} = {v:.3g} {'тянет вверх' if dlt > 0 else 'тянет вниз'} {dlt:+.3f}%" for k, dlt, v in why))
        title = (f"Выбор по скану: {'беру ' + top['symbol'] if top['take'] else 'все кандидаты в минусе — пропускаю'}"
                 if use else "Рынок сменился — решает сканер с обычными настройками, я изучаю новый режим"
                 if self.regime_hold else "Оцениваю кандидатов (решает сканер с обычными настройками, я пока учусь)")
        self.think("decide", title, lines)

    def loop(self):
        """Фон: шаг учёбы раз в минуту, через несколько секунд после закрытия минутной свечи."""
        while True:
            try:
                self.step()
            except Exception as e:  # noqa: BLE001
                log.warning("Нейросеть: ошибка: %s", e)
                self.progress = f"ошибка: {e}"
            nxt = (int(time.time() // 60) + 1) * 60 + 3
            time.sleep(max(1.0, nxt - time.time()))

    def start(self):
        lines = ["Учусь на ходу: только последние часы рынка, дообучение каждую минуту",
                 "Пока я не прошла проверку, бот торгует обычными настройками"]
        if self.removed_hist:
            lines.append("Удалила кэш месячной истории — для скальпинга он не нужен")
        if self.model is not None:
            lines.insert(0, "Модель из памяти загружена, свежие данные дочитаю за минуту")
        self.think("plan", "Запускаюсь", lines + self.plan_lines())
        threading.Thread(target=self.loop, daemon=True).start()

    def status(self):
        now = time.time() * 1000
        since = (self.regime_start if self.regime_start and now - self.regime_start < self.a.nn_eval_min * 60_000
                 else None)
        om = self.online_metrics(now, self.a.nn_eval_min, since=since)
        n_post = sum(1 for e in self.evals if e["t"] >= self.regime_start and "net" in e) if self.regime_hold else 0
        with self.lock:
            return {
                "regime": self.regime, "regime_hold": self.regime_hold, "regime_start": self.regime_start,
                "regime_n": n_post, "regime_need": self.a.nn_regime_min_n, "regime_since": since,
                "regime_vol": self.a.nn_regime_vol,
                "active": self.active, "text": self.status_text, "progress": self.progress,
                "samples": self.n(), "real": int((self.kind == 1).sum()) + sum(1 for b in self._buf if b[4] == 1),
                "history": self.history[-600:], "curve": self.curve, "preds": self.preds[:12],
                "min_edge": self.a.nn_min_edge, "enabled": self.a.nn,
                "horizon": int(self.a.nn_max_hold),
                "thoughts": list(self.thoughts)[:80], "plan": self.plan_lines(),
                "importance": [{"name": FEATURE_RU[k], "value": v} for k, v in self.importance[:8]],
                "online": om, "pending": len(self.pending), "coins": len(self.K),
                "window_h": self.a.nn_window_hours, "eval_min": self.a.nn_eval_min,
                "mae": float(np.mean(self.new_err)) if self.new_err else None,
            }
