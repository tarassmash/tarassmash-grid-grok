"""Встроенная нейросеть сеточного бота: учится предсказывать доходность нейтральной сетки.

Откуда данные:
  1) учебные сетки — бот сам прогоняет свою сетку по минутным свечам ликвидных монет
     (те же правила: сдвиг за ценой, лимит позиции, стоп по убытку, комиссии): при первом запуске —
     по истории за --nn-history-days дней (кэш на диске), дальше — по новым свечам каждые --nn-refresh минут;
     свежие примеры весят больше старых (вес вдвое меньше каждые --nn-half-life дней);
  2) реальные сетки — каждая закрытая сетка на бирже, с весом ×3.

Что предсказывает: доходность сетки в % от объёма позиции (не зависит от маржи и плеча) за --max-minutes.
Для каждой монеты сеть сравнивает 3 варианта шага (частый / обычный / редкий) и выбирает лучший.

Честность: после обучения сеть проверяется на самых свежих примерах, которых не видела.
Решения она принимает, только если сетки, которые она отбирает, там в плюсе и лучше обычного отбора.
"""

import collections
import json
import logging
import math
import os
import threading
import time

import numpy as np

log = logging.getLogger("grid")
MULTS = (0.6, 1.0, 1.6)        # множитель шага: частая / обычная / редкая сетка
HORIZON_MIN = 60
FEATURES = ["vol1m", "atr", "er", "step", "levels", "width", "log_turnover", "ret_win", "ret_5",
            "pos_range", "atr_trend", "mult"]
# как признаки называть в «мыслях» нейросети
FEATURE_RU = {
    "vol1m": "размах за минуту", "atr": "средняя волатильность", "er": "трендовость (ER)", "step": "шаг сетки",
    "levels": "число уровней", "width": "ширина диапазона", "log_turnover": "объём торгов",
    "ret_win": "движение за 15 мин", "ret_5": "движение за 5 мин", "pos_range": "место цены в диапазоне",
    "atr_trend": "ускорение волатильности", "mult": "вариант шага",
}
MULT_RU = {0.6: "частая", 1.0: "обычная", 1.6: "редкая"}
MULT_ACC = {0.6: "частую", 1.0: "обычную", 1.6: "редкую"}


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


def variant(base, mult, max_levels):
    """Признаки для варианта шага: mult > 1 — шаг шире, уровней меньше."""
    levels = max(2, min(max_levels, int(round(base["grids"] / mult))))
    f = dict(base)
    f["levels"] = levels / 40.0
    f["step"] = base["width"] / levels
    f["mult"] = mult
    return f, levels


def vec(f):
    return [float(f[k]) for k in FEATURES]


# ---------------------------------------------------------------- симулятор сетки по свечам

def simulate_grid(future, price, lower, upper, n, fee_pct, leverage, max_loss_pct, shift_steps=1.0, trace=None):
    """Прогон нейтральной сетки бота по минутным свечам future [(o,h,l,c), ...].

    Внутри свечи цена идёт open→low→high→close (если свеча зелёная) или open→high→low→close.
    Возвращает доходность в % от объёма позиции (маржа × плечо).
    """
    width = upper - lower

    def build(lo, hi, px):
        step = (hi - lo) / n
        lv = [round(lo + i * step, 10) for i in range(n + 1)]
        empty = min(range(n + 1), key=lambda i: abs(lv[i] - px))
        want = {i: (1 if i < empty else -1) for i in range(n + 1) if i != empty}   # 1 = Buy, -1 = Sell
        return lv, want, step

    lv, want, step = build(lower, upper, price)
    q = (1.0 / n) / price          # объём одного уровня: вся позиция = 1 USDT объёма
    max_pos = q * n
    net = avg = realized = 0.0
    stop_at = -(max_loss_pct / 100) / max(leverage, 1) if max_loss_pct > 0 else None
    last = price

    def fill(side, px):
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
            trace.append((side, px, round(net / q, 6)))

    def can(side):
        return (net * side) + q <= max_pos * 1.0001

    def move(a, b):
        nonlocal want
        if b < a:      # вниз: исполняются Buy
            for i in sorted([i for i, s in want.items() if s == 1 and b <= lv[i] < a], reverse=True):
                if want.get(i) != 1 or not can(1):
                    continue
                fill(1, lv[i])
                del want[i]
                if i + 1 <= n:
                    want[i + 1] = -1
        elif b > a:    # вверх: исполняются Sell
            for i in sorted([i for i, s in want.items() if s == -1 and a < lv[i] <= b]):
                if want.get(i) != -1 or not can(-1):
                    continue
                fill(-1, lv[i])
                del want[i]
                if i - 1 >= 0:
                    want[i - 1] = 1

    for (o, h, l, c) in future:
        path = [o, l, h, c] if c >= o else [o, h, l, c]
        for a, b in zip(path, path[1:]):
            move(a, b)
        last = c
        pnl = realized + (c - avg) * net
        if stop_at is not None and pnl <= stop_at:
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

    def fit(self, X, y, w, Xv, yv, epochs=60, lr=2e-3, batch=256, l2=5e-4, delta=1.0, seed=0, on_epoch=None,
            patience=12):
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
        if best[1]:
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


# ---------------------------------------------------------------- «мозг»: данные, обучение, решения

class Brain:
    def __init__(self, path, a, market, analyze_fn, klines_fn, symbols_fn):
        self.path = path
        self.a = a
        self.market = market            # HTTP-клиент для рыночных данных (pybit, реальный рынок)
        self.analyze = analyze_fn
        self.klines = klines_fn
        self.symbols = symbols_fn       # () -> список ликвидных монет
        self.lock = threading.RLock()
        nf = len(FEATURES)
        # данные в numpy: сотни тысяч примеров занимают десятки МБ, а не гигабайты
        self.X = np.zeros((0, nf), np.float32)
        self.y = np.zeros(0, np.float32)
        self.w = np.zeros(0, np.float32)
        self.t = np.zeros(0, np.float64)
        self.kind = np.zeros(0, np.int8)            # 0 учебная, 1 реальная
        self._buf = []
        self.covered = {}               # монета -> время последней разобранной учебной сетки
        self.hist_dir = os.path.join(path, "hist")
        self.hist = {"symbols": 0, "candles": 0, "days": 0.0}
        self.model, self.mu, self.sd = None, None, None
        self.active = False
        self.history = []               # запуски обучения
        self.curve = []                 # кривая последнего обучения
        self.status_text = "загрузка"
        self.progress = ""
        self.last_boot = 0
        self.last_train_n = 0
        self.preds = []                 # прогнозы по последним кандидатам
        self.thoughts = collections.deque(maxlen=400)   # журнал мыслей: что делает, почему, что дальше
        self.importance = []
        self._last_decision = ("", 0.0)
        os.makedirs(self.hist_dir, exist_ok=True)
        self.load()

    # --- хранение
    def n(self):
        return len(self.y) + len(self._buf)

    def _flush(self):
        with self.lock:
            if not self._buf:
                return
            xs, ys, ws, ts, ks = zip(*self._buf)
            self._buf = []
            self.X = np.concatenate([self.X, np.array(xs, np.float32)])
            self.y = np.concatenate([self.y, np.array(ys, np.float32)])
            self.w = np.concatenate([self.w, np.array(ws, np.float32)])
            self.t = np.concatenate([self.t, np.array(ts, np.float64)])
            self.kind = np.concatenate([self.kind, np.array(ks, np.int8)])

    def _trim(self):
        """Оставляем самые свежие --nn-max-samples примеров; реальные сетки не выбрасываем никогда."""
        with self.lock:
            cap = self.a.nn_max_samples
            if len(self.y) <= cap:
                return
            real = np.where(self.kind == 1)[0]
            other = np.where(self.kind != 1)[0]
            keep_other = other[np.argsort(self.t[other])][-(cap - len(real)):]
            keep = np.sort(np.concatenate([real, keep_other]))
            self.X, self.y, self.w, self.t, self.kind = (self.X[keep], self.y[keep], self.w[keep],
                                                         self.t[keep], self.kind[keep])

    def save(self):
        try:
            self._flush()
            with self.lock:
                tmp = os.path.join(self.path, "data.npz.tmp")
                with open(tmp, "wb") as f:
                    np.savez_compressed(f, X=self.X, y=self.y, w=self.w, t=self.t, kind=self.kind)
                os.replace(tmp, os.path.join(self.path, "data.npz"))
                meta = {"history": self.history[-300:], "curve": self.curve, "active": self.active,
                        "model": self.model.to_json() if self.model else None,
                        "mu": self.mu.tolist() if self.mu is not None else None,
                        "sd": self.sd.tolist() if self.sd is not None else None, "features": FEATURES,
                        "covered": self.covered, "thoughts": list(self.thoughts)[:200],
                        "importance": self.importance, "last_train_n": self.last_train_n, "hist": self.hist}
            tmp = os.path.join(self.path, "brain.json.tmp")
            with open(tmp, "w") as f:
                json.dump(meta, f)
            os.replace(tmp, os.path.join(self.path, "brain.json"))
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
            if "X" in d:      # старый формат (списки в JSON) — переносим
                self.X = np.array(d["X"], np.float32).reshape(-1, len(FEATURES))
                self.y, self.w = np.array(d["y"], np.float32), np.array(d["w"], np.float32)
                self.t, self.kind = np.array(d["t"], np.float64), np.array(d["kind"], np.int8)
            elif os.path.exists(os.path.join(self.path, "data.npz")):
                z = np.load(os.path.join(self.path, "data.npz"))
                self.X, self.y, self.w, self.t, self.kind = z["X"], z["y"], z["w"], z["t"], z["kind"]
            self.history, self.curve, self.active = d.get("history", []), d.get("curve", []), d.get("active", False)
            if d.get("model"):
                self.model = MLP.from_json(d["model"])
                self.mu, self.sd = np.array(d["mu"]), np.array(d["sd"])
            self.covered = d.get("covered") or {}
            self.thoughts.extend(d.get("thoughts") or [])
            self.importance = d.get("importance") or []
            self.last_train_n = d.get("last_train_n", len(self.y))
            self.hist = d.get("hist") or self.hist
            self.status_text = f"загружено {len(self.y)} примеров"
            log.info("Нейросеть: загружено %d примеров, модель %s", len(self.y), "активна" if self.active else "учится")
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
            X[k, k] = 0.0       # 0 после нормировки = среднее значение
        deltas = base - self.model.predict(X)
        order = np.argsort(-np.abs(deltas))[:3]
        return [(FEATURES[k], float(deltas[k]), float(f[FEATURES[k]])) for k in order]

    def plan_lines(self):
        nxt_boot = self.last_boot + self.a.nn_refresh * 60
        need = max(0, self.a.nn_retrain_every - (self.n() - self.last_train_n))
        out = [f"Следующий сбор учебных сеток: ~{time.strftime('%H:%M UTC', time.gmtime(nxt_boot))}"
               if self.last_boot else "Сбор учебных сеток: сейчас",
               f"Переобучение: после ещё {need} новых примеров (обычно сразу после сбора)",
               f"История: {self.hist.get('symbols', 0)} монет, {self.hist.get('days', 0):.0f} дн. минутных свечей; "
               f"свежие дни важнее старых (вес вдвое меньше каждые {self.a.nn_half_life:g} дн.)"]
        if not self.active:
            out.append("Чтобы включиться, мои сетки на проверке должны быть в плюсе и лучше всех сеток сканера, "
                       "и таких сеток должно быть не меньше 30")
        else:
            out.append("Сейчас я выбираю монету и шаг сетки; если на проверке стану хуже сканера — сама отключусь")
        return out

    # --- данные
    def add(self, feats, target, kind, t_ms, weight):
        with self.lock:
            self._buf.append((vec(feats), float(max(-10.0, min(10.0, target))), float(weight), float(t_ms), kind))

    def add_real(self, feats, target, pred=None, sym=""):
        """Закрытая сетка на бирже: доходность в % от объёма позиции."""
        if not feats:
            return
        self.add(feats, target, 1, time.time() * 1000, 3.0)
        lines = [f"Итог: {target:+.3f}% от объёма позиции"]
        if pred is not None:
            err = target - pred
            lines.append(f"Мой прогноз был {pred:+.3f}% — ошиблась на {err:+.3f}% "
                         + ("(оказалось лучше, чем думала)" if err > 0 else "(оказалось хуже, чем думала)"))
        else:
            lines.append("Прогноза не было — сетку выбрал сканер, пока я училась")
        self._flush()
        lines.append(f"Запомнила с весом ×3. Реальных сеток в памяти: {int((self.kind == 1).sum())}")
        self.think("result", f"{sym}: разбираю закрытую реальную сетку", lines)
        self.save()

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
                time.sleep(1.5 * (i + 1))

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
            time.sleep(0.04)
        if not rows:
            return np.zeros((0, 7))
        return np.array([[float(x) for x in k[:7]] for k in rows])

    def candles(self, sym):
        """Свечи монеты за --nn-history-days: кэш на диске + докачка недостающего. Массив (n, 7) от старых к новым."""
        f = os.path.join(self.hist_dir, sym + ".npy")
        now = time.time() * 1000
        start = now - self.a.nn_history_days * 86_400_000
        arr = np.load(f) if os.path.exists(f) else np.zeros((0, 7))
        arr = arr[arr[:, 0] >= start] if len(arr) else arr
        parts = [arr]
        if not len(arr) or arr[0, 0] > start + 3_600_000:          # не хватает старой истории
            parts.append(self._fetch(sym, start, (arr[0, 0] - 1) if len(arr) else now))
        if len(arr):                                                 # докачка новых свечей
            parts.append(self._fetch(sym, arr[-1, 0] + 60_000, now))
        arr = np.concatenate([p for p in parts if len(p)]) if any(len(p) for p in parts) else np.zeros((0, 7))
        if len(arr):
            _, idx = np.unique(arr[:, 0], return_index=True)
            arr = arr[idx]
            arr = arr[arr[:, 0] + 60_000 <= now]                     # только закрытые свечи
        np.save(f, arr)
        return arr

    def bootstrap(self):
        """Учебные сетки по минутным свечам: при первом запуске — вся история за --nn-history-days,
        дальше — только новые свечи. Возвращает число новых примеров."""
        a = self.a
        syms = self.symbols()[: a.nn_symbols]
        H = min(HORIZON_MIN, int(a.max_minutes))
        win = a.window + 1
        first = not self.covered
        if first:
            self.think("data", f"Загружаю историю: {a.nn_history_days} дн. минутных свечей по {len(syms)} монетам",
                       ["Первая загрузка займёт несколько минут — Bybit отдаёт по 1000 свечей за запрос",
                        f"По истории ставлю учебную сетку каждые {a.nn_history_stride} мин., по последним суткам — "
                        f"каждые {a.nn_stride} мин.; на каждую — 3 варианта шага, итог через {H} мин."])
        else:
            self.think("data", f"Дочитываю новые свечи по {len(syms)} монетам",
                       [f"Учебная сетка каждые {a.nn_stride} мин. свежего рынка, 3 варианта шага, итог через {H} мин."])
        n0 = self.n()
        now = time.time() * 1000
        total_candles, oldest = 0, now
        for si, sym in enumerate(syms):
            self.progress = f"{'история' if first else 'новые свечи'}: {sym} ({si + 1}/{len(syms)})"
            try:
                KL = self.candles(sym)
            except Exception as e:  # noqa: BLE001
                log.warning("Нейросеть: свечи %s: %s", sym, e)
                continue
            n = len(KL)
            total_candles += n
            if n:
                oldest = min(oldest, KL[0, 0])
            if n < win + H:
                continue
            tm = KL[:, 0]
            last = self.covered.get(sym, 0)
            for end in range(win, n - H + 1):
                t_end = tm[end - 1]
                if t_end <= last:
                    continue
                stride = a.nn_stride if now - t_end < 86_400_000 else a.nn_history_stride
                if int(t_end // 60_000) % stride:
                    continue
                if tm[end + H - 1] - tm[end - win] != (win + H - 1) * 60_000:
                    continue                                         # дыра в свечах — пропускаем
                window = KL[end - win:end]
                m = self.analyze(window, a)
                if not m or m["vol1m"] < a.min_1m * 0.3:
                    continue
                base = base_features(window, m)
                fut = KL[end:end + H, 1:5]
                for mult in MULTS:
                    f, levels = variant(base, mult, a.max_levels)
                    r = simulate_grid(fut, m["price"], m["lower"], m["upper"], levels, a.fee, a.leverage,
                                      a.max_loss, a.shift_steps)
                    self.add(f, r, 0, t_end, 1.0)
            self.covered[sym] = max(last, float(tm[n - H]))
        self.hist = {"symbols": len(syms), "candles": int(total_candles), "days": (now - oldest) / 86_400_000}
        self._flush()
        self._trim()
        self.last_boot = time.time()
        added = self.n() - n0
        if added > 0:
            with self.lock:
                yn = self.y[-added:].astype(float)
                Xn = self.X[-added:].astype(float)
            lines = [f"Новых примеров: {added:,}. Прибыльных: {(yn > 0).mean() * 100:.0f}%, средний итог {yn.mean():+.3f}%"]
            if first:
                lines.insert(0, f"История: {total_candles:,} минутных свечей за {self.hist['days']:.0f} дн.")
            mcol = Xn[:, FEATURES.index("mult")]
            best = None
            for mlt in MULTS:
                sel = np.isclose(mcol, mlt)
                if sel.any():
                    avg = yn[sel].mean()
                    lines.append(f"{MULT_RU[mlt].capitalize()} сетка: в среднем {avg:+.3f}%, в плюсе {(yn[sel] > 0).mean() * 100:.0f}%")
                    best = (mlt, avg) if best is None or avg > best[1] else best
            ercol = Xn[:, FEATURES.index("er")]
            lo, hi = yn[ercol < 0.3], yn[ercol >= 0.3]
            if len(lo) and len(hi):
                lines.append(f"Боковик (ER < 0.3): {lo.mean():+.3f}%, трендовее (ER ≥ 0.3): {hi.mean():+.3f}%")
            if best:
                lines.append(f"Лучше всего шла {MULT_RU[best[0]]} сетка")
            lines.append(f"Всего в памяти: {self.n():,} примеров")
            self.think("data", "История загружена" if first else "Учебные сетки собраны", lines)
        else:
            self.think("data", "Новых учебных сеток нет", ["Новые свечи ещё не накопились — жду"])
        return added

    # --- обучение
    def train(self):
        self._flush()
        with self.lock:
            X = self.X.astype(float)
            y = self.y.astype(float)
            w = self.w.astype(float)
            t = self.t.copy()
            kind = self.kind.copy()
        if len(y) < 300:
            self.status_text = f"мало данных ({len(y)} примеров), жду"
            self.think("plan", "Мало данных для обучения", [f"Есть {len(y)} примеров, нужно хотя бы 300"] + self.plan_lines())
            return
        self.think("train", f"Начинаю обучение №{len(self.history) + 1}",
                   [f"Примеров: {len(y):,}, из них реальных сеток: {int((kind == 1).sum())}",
                    "Самые свежие 20% по времени отложу для проверки — их я не увижу при обучении",
                    f"Свежие примеры важнее: вес вдвое меньше каждые {self.a.nn_half_life:g} дн.",
                    f"До {self.a.nn_epochs} проходов; остановлюсь раньше, если проверка 12 проходов не улучшается"])
        order = np.argsort(t, kind="stable")
        X, y, w, t, kind = X[order], y[order], w[order], t[order], kind[order]
        age_days = (t.max() - t) / 86_400_000
        w = w * np.maximum(0.1, 0.5 ** (age_days / max(self.a.nn_half_life, 0.1)))
        n_val = max(100, int(len(y) * 0.2))
        Xt, yt, wt = X[:-n_val], y[:-n_val], w[:-n_val]
        Xv, yv = X[-n_val:], y[-n_val:]
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
        model = MLP(X.shape[1], seed=len(self.history))

        def prog(ep, n):
            self.progress = f"обучение: эпоха {ep}/{n}"
        batch = 256 if len(yt) < 50_000 else 1024
        curve = model.fit((Xt - mu) / sd, yt, wt, (Xv - mu) / sd, yv, epochs=self.a.nn_epochs, batch=batch,
                          on_epoch=prog)
        pv = model.predict((Xv - mu) / sd)
        # проверка на свежих данных: отбор сети против «брать всё, что дал сканер»
        sel = pv > self.a.nn_min_edge
        base_avg = float(yv[np.isclose(Xv[:, FEATURES.index("mult")], 1.0)].mean()) if len(yv) else 0.0
        sel_avg = float(yv[sel].mean()) if sel.any() else 0.0
        corr = float(np.corrcoef(pv, yv)[0, 1]) if len(yv) > 2 and np.std(pv) > 0 else 0.0
        active = bool(sel.sum() >= 30 and sel_avg > 0 and sel_avg > base_avg)
        # важность признаков: насколько растёт ошибка, если перемешать признак на проверке
        Zv = (Xv - mu) / sd
        if len(Zv) > 20000:
            Zv, yvi = Zv[-20000:], yv[-20000:]
        else:
            yvi = yv
        base_err = float(np.mean(np.abs(model.predict(Zv) - yvi)))
        rng = np.random.default_rng(1)
        imp = []
        for k in range(len(FEATURES)):
            Zs = Zv.copy()
            Zs[:, k] = rng.permutation(Zs[:, k])
            imp.append((FEATURES[k], float(np.mean(np.abs(model.predict(Zs) - yvi)) - base_err)))
        imp.sort(key=lambda x: -x[1])
        run = {
            "time": int(time.time() * 1000), "samples": int(len(y)), "real": int((kind == 1).sum()),
            "train_loss": curve[-1]["train"], "val_loss": min(c["val"] for c in curve), "corr": corr,
            "sel_avg": sel_avg, "base_avg": base_avg, "sel_n": int(sel.sum()), "val_n": int(len(yv)),
            "sel_share": float(sel.mean()) if len(yv) else 0.0, "active": active, "epochs": len(curve),
        }
        prev = self.history[-1] if self.history else None
        was_active = self.active
        with self.lock:
            self.model, self.mu, self.sd = model, mu, sd
            self.active = active
            self.curve = curve
            self.history.append(run)
            self.last_train_n = len(y)
            self.importance = imp
        best_ep = min(curve, key=lambda c: c["val"])["epoch"]
        lines = [f"Ошибка: на обучении {curve[-1]['train']:.3f}, на проверке {run['val_loss']:.3f} "
                 f"(лучшая на проходе {best_ep} из {len(curve)})"]
        if len(curve) < self.a.nn_epochs:
            lines.append(f"Остановилась на проходе {len(curve)}: дальше проверка не улучшалась — это было бы заучивание шума")
        lines.append(f"Связь прогноза с реальным итогом на проверке: {corr:+.2f} "
                     + ("(слабая)" if abs(corr) < 0.15 else "(заметная)" if abs(corr) < 0.4 else "(сильная)"))
        lines.append(f"На проверке выбрала бы {run['sel_n']:,} из {run['val_n']:,} сеток: в среднем {sel_avg:+.3f}% "
                     f"против {base_avg:+.3f}% у всех сеток сканера")
        if prev:
            d = sel_avg - prev["sel_avg"]
            lines.append(f"По сравнению с прошлым обучением мой отбор {'лучше' if d > 0 else 'хуже'} на {abs(d):.3f}%")
        lines.append("Больше всего опираюсь на: " + ", ".join(f"{FEATURE_RU[k]}" for k, v in imp[:4] if v > 0))
        self.think("train", f"Обучение №{len(self.history)} готово", lines)
        if active and not was_active:
            self.think("decide", "Включаюсь в принятие решений",
                       ["Проверка пройдена: мой отбор в плюсе и лучше сканера", "Теперь выбираю монету и шаг сетки сама"])
        elif not active and was_active:
            self.think("warn", "Отключаюсь от решений",
                       ["На свежей проверке мой отбор больше не лучше сканера — бот торгует по сканеру, я продолжаю учиться"])
        elif not active:
            why = ("мало сеток прошли бы мой отбор" if sel.sum() < 30 else
                   "мой отбор на проверке в минусе" if sel_avg <= 0 else "мой отбор не лучше всех сеток сканера")
            self.think("warn", "Пока не включаюсь", [f"Причина: {why}"])
        self.think("plan", "Что дальше", self.plan_lines())
        self.status_text = ("ПРИНИМАЕТ РЕШЕНИЯ" if active else "учится (на проверке пока не лучше сканера)")
        log.info("Нейросеть: обучена на %d примерах | проверка: отбор %+.3f%% vs все %+.3f%% (%d сеток) | %s",
                 len(y), sel_avg, base_avg, run["sel_n"], "активна" if active else "не активна")
        self.save()

    # --- решения
    def evaluate(self, cand):
        """Все 3 варианта шага: [(множитель, уровней, прогноз, признаки)]."""
        base = cand.get("nn_base")
        if not base:
            return []
        rows = [variant(base, mlt, self.a.max_levels) for mlt in MULTS]
        if not self.model:
            return [(mlt, lv, None, f) for mlt, (f, lv) in zip(MULTS, rows)]
        with self.lock:
            X = (np.array([vec(f) for f, _ in rows]) - self.mu) / self.sd
            p = self.model.predict(X)
        return [(mlt, lv, float(pp), f) for mlt, (f, lv), pp in zip(MULTS, rows, p)]

    def log_decisions(self, decided, use):
        """Мысли по скану: прогнозы по вариантам шага и почему беру/пропускаю."""
        if not decided or not self.model:
            return
        top = decided[0]
        key = f"{top['symbol']}:{top['take']}:{top['mult']}"
        if key == self._last_decision[0] and time.time() - self._last_decision[1] < 300:
            return
        self._last_decision = (key, time.time())
        lines = []
        for d in decided[:3]:
            vs = " | ".join(f"{MULT_RU[m]} {p:+.3f}%" for m, _, p, _ in d["variants"])
            lines.append(f"{d['symbol']}: {vs}")
            if use:
                verdict = (f"→ беру {MULT_ACC[d['mult']]} сетку ({d['levels']} уровней), прогноз {d['pred']:+.3f}%"
                           if d["take"] else f"→ пропускаю: лучший прогноз {d['pred']:+.3f}% ниже порога {self.a.nn_min_edge:+.2f}%")
                lines.append(verdict)
            why = self.explain(d["feats"])
            if why:
                lines.append("   почему: " + "; ".join(
                    f"{FEATURE_RU[k]} = {v:.3g} {'тянет вверх' if dlt > 0 else 'тянет вниз'} {dlt:+.3f}%" for k, dlt, v in why))
        title = (f"Выбор по скану: {'беру ' + top['symbol'] if top['take'] else 'все кандидаты в минусе — пропускаю'}"
                 if use else "Оцениваю кандидатов (решает сканер, я пока учусь)")
        self.think("decide", title, lines)

    def choose(self, cand):
        """Лучший вариант шага для кандидата: (уровней, прогноз %, признаки) или прогноз None, если модели нет."""
        base = cand.get("nn_base")
        if not base:
            return cand["grids"], None, None
        rows = [variant(base, mlt, self.a.max_levels) for mlt in MULTS]
        if not self.model:
            f, levels = rows[1]
            return levels, None, f
        with self.lock:
            X = (np.array([vec(f) for f, _ in rows]) - self.mu) / self.sd
            p = self.model.predict(X)
        k = int(np.argmax(p))
        return rows[k][1], float(p[k]), rows[k][0]

    def loop(self):
        """Фон: учебные сетки раз в --nn-refresh минут, дообучение при новых данных."""
        while True:
            try:
                if time.time() - self.last_boot > self.a.nn_refresh * 60:
                    self.status_text = "собираю учебные сетки" if not self.model else self.status_text
                    self.last_boot = time.time()      # даже если сбор упадёт, не долбим биржу каждые 20 сек
                    added = self.bootstrap()
                    log.info("Нейросеть: +%d учебных примеров (всего %d)", added, self.n())
                    self.save()
                if self.n() - self.last_train_n >= self.a.nn_retrain_every or (self.model is None and self.n() >= 300):
                    self.train()
                self.progress = ""
            except Exception as e:  # noqa: BLE001
                log.warning("Нейросеть: ошибка: %s", e)
                self.progress = f"ошибка: {e}"
            time.sleep(20)

    def start(self):
        if self.model:
            self.think("plan", "Запускаюсь с сохранённой памятью",
                       [f"В памяти {self.n():,} примеров, обучений было {len(self.history)}",
                        "Сейчас: " + ("принимаю решения" if self.active else "учусь, решает сканер")] + self.plan_lines())
        else:
            self.think("plan", "Запускаюсь с нуля",
                       ["Памяти нет — сначала соберу учебные сетки по свежим свечам, потом обучусь",
                        "Пока я не прошла проверку, монеты выбирает обычный сканер"])
        threading.Thread(target=self.loop, daemon=True).start()

    def status(self):
        with self.lock:
            return {
                "active": self.active, "text": self.status_text, "progress": self.progress,
                "samples": self.n(), "real": int((self.kind == 1).sum()) + sum(1 for b in self._buf if b[4] == 1),
                "history": self.history[-200:], "curve": self.curve, "preds": self.preds[:12],
                "min_edge": self.a.nn_min_edge, "enabled": self.a.nn,
                "horizon": int(min(HORIZON_MIN, self.a.max_minutes)),
                "thoughts": list(self.thoughts)[:80], "plan": self.plan_lines(),
                "importance": [{"name": FEATURE_RU[k], "value": v} for k, v in self.importance[:8]],
                "hist": self.hist,
            }
