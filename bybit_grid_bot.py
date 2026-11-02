#!/usr/bin/env python3
"""
Bybit: сканер волатильности + авто нейтральная сетка (USDT perpetual, one-way mode).

Что делает:
  1. Сканирует все USDT-перпетуалы Bybit, ищет пары с большим размахом за последнюю минуту
     и без тренда (низкий Efficiency Ratio).
  2. Берёт лучшую пару, считает границы и уровни сетки.
  3. Ставит лимитки: Buy ниже цены, Sell выше, один уровень у цены пустой.
  4. Каждую секунду проверяет исполнения: Buy на уровне i исполнен -> Sell на i+1,
     Sell на уровне i исполнен -> Buy на i-1. Сетка сразу обновляется.
  5. Цена ушла за край сетки -> сетка той же ширины перестраивается вокруг цены (идёт за ценой).
     Позиция не растёт больше маржи × плечо.
  6. Закрывает сетку, если: достигнут --tp-usdt, убыток больше --max-loss %% маржи,
     прошло --max-minutes, или (по желанию) тренд ER > --exit-er.
     Закрытие ТОЛЬКО лимитками с догоном: мейкер у лучшей цены, перестановка каждые
     --chase-sec, затем быстрая лимитка через спред. Маркет-ордеров нет.
  7. Пауза для этой пары и поиск следующей.

Установка:
  pip install pybit

Ключи (создай API key на Bybit только с правом Contract/Derivatives trade, БЕЗ вывода,
лучше с привязкой к IP). Положи в файл .env рядом со скриптом:
  BYBIT_API_KEY=xxxx
  BYBIT_API_SECRET=yyyy

Запуск:
  python bybit_grid_bot.py --dry            # только показать, что бы поставил (ключи не нужны)
  python bybit_grid_bot.py                  # РЕАЛЬНЫЙ счёт (по умолчанию)
  python bybit_grid_bot.py --testnet        # учебный счёт, ключи от testnet.bybit.com

Режим позиций One-Way и Hedge определяется сам.
Ctrl+C / остановка сервиса: бот снимет ордера и закроет позицию лимитками (до --term-timeout сек).
"""

import argparse
import logging
import os
import shlex
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal

from pybit.unified_trading import HTTP

import nn
import status_web as web

BRAIN = None   # встроенная нейросеть (nn.Brain), создаётся в main()

log = logging.getLogger("grid")
CAT = "linear"


# ---------------------------------------------------------------- utils

def load_env(path=".env"):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.exists(p):
        return
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def D(x):
    return Decimal(str(x))


def round_step(x, step, mode=ROUND_DOWN):
    x, step = D(x), D(step)
    # quantize — чтобы в API уходило "10", а не "1E+1"
    return ((x / step).to_integral_value(rounding=mode) * step).quantize(step)


def api(fn, *args, retries=3, **kw):
    """Вызов pybit с повтором при сетевых ошибках / rate limit."""
    for i in range(retries):
        try:
            return fn(*args, **kw)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            # ошибки "ничего не изменилось" — не ошибки
            if "110043" in msg or "leverage not modified" in msg.lower():
                return None
            if i == retries - 1:
                raise
            log.warning("API ошибка (%s), повтор…", msg[:160])
            time.sleep(1.5 * (i + 1))


# ---------------------------------------------------------------- market data

def load_instruments(s):
    out, cursor = {}, None
    while True:
        kw = {"category": CAT, "limit": 1000}
        if cursor:
            kw["cursor"] = cursor
        r = api(s.get_instruments_info, **kw)["result"]
        for x in r["list"]:
            if x.get("quoteCoin") != "USDT" or x.get("status") != "Trading":
                continue
            if x.get("contractType") != "LinearPerpetual":
                continue
            lot = x["lotSizeFilter"]
            out[x["symbol"]] = {
                "tick": D(x["priceFilter"]["tickSize"]),
                "qty_step": D(lot["qtyStep"]),
                "min_qty": D(lot["minOrderQty"]),
                "min_notional": D(lot.get("minNotionalValue") or "5"),
                "max_lev": float(x["leverageFilter"]["maxLeverage"]),
            }
        cursor = r.get("nextPageCursor")
        if not cursor:
            break
    return out


def analyze(kl, a):
    """kl — свечи Bybit от старой к новой: [start, open, high, low, close, volume, turnover]."""
    if len(kl) < 5:
        return None
    o = [float(k[1]) for k in kl]
    h = [float(k[2]) for k in kl]
    l = [float(k[3]) for k in kl]
    c = [float(k[4]) for k in kl]
    tv = [float(k[6]) for k in kl]
    n = len(kl)
    vol1m = (max(h[-2:]) - min(l[-2:])) / o[-2] * 100
    atr = sum((h[i] - l[i]) / o[i] * 100 for i in range(n)) / n
    path = sum(abs(c[i] - c[i - 1]) for i in range(1, n))
    er = abs(c[-1] - c[0]) / path if path > 0 else 1.0
    price = c[-1]
    lower = min(l) * (1 - a.pad / 100)
    upper = max(h) * (1 + a.pad / 100)
    width = (upper - lower) / price * 100
    step_pct = max(a.fee * 2 * 3, atr / 2)
    grids = max(2, min(a.max_levels, int(width / step_pct)))
    return {
        "price": price, "vol1m": vol1m, "atr": atr, "er": er, "turnover": sum(tv),
        "score": vol1m * (1 - er), "lower": lower, "upper": upper, "grids": grids,
        "step_pct": width / grids,
    }


def klines(s, sym, n):
    r = api(s.get_kline, category=CAT, symbol=sym, interval="1", limit=n)
    return list(reversed(r["result"]["list"]))


def scan(s, inst, a):
    tick = api(s.get_tickers, category=CAT)["result"]["list"]
    syms = [t["symbol"] for t in tick
            if t["symbol"] in inst and float(t.get("turnover24h") or 0) >= a.min_vol]

    def one(sym):
        try:
            kl = klines(s, sym, a.window + 1)
            m = analyze(kl, a)
        except Exception:  # noqa: BLE001
            return None
        if m:
            m["symbol"] = sym
            m["nn_base"] = nn.base_features(kl, m)
        return m

    with ThreadPoolExecutor(max_workers=6) as ex:
        res = [r for r in ex.map(one, syms) if r]
    res = [r for r in res if r["er"] <= a.max_er and r["turnover"] >= a.min_vol_win
           and r["vol1m"] >= a.min_1m]
    res.sort(key=lambda r: r["score"], reverse=True)
    res = apply_brain(res, a)
    log.info("Скан: %d пар по объёму, %d прошли фильтры", len(syms), len(res))
    web.update(scan={"time": time.time(), "checked": len(syms), "passed": len(res),
                     "top": [{**{k: r[k] for k in ("symbol", "vol1m", "atr", "er", "score")},
                              "pred": r["nn"]["pred"], "levels": r["grids"]} for r in res[:10]]})
    for r in res[:5]:
        log.info("  %-14s 1m=%.2f%% ATR=%.2f%% ER=%.2f score=%.2f",
                 r["symbol"], r["vol1m"], r["atr"], r["er"], r["score"])
    return res


def apply_brain(res, a):
    """Нейросеть оценивает каждого кандидата и 3 варианта шага сетки.
    Активна (прошла проверку) — выбирает шаг, сортирует по прогнозу и отсекает ожидаемый минус.
    Не активна — бот работает по сканеру, а сеть запоминает признаки для обучения и пишет свои оценки."""
    if not BRAIN:
        return res
    out, decided = [], []
    use = a.nn and BRAIN.active
    for c in res:
        vs = BRAIN.evaluate(c)
        if not vs:
            continue
        scored = [v for v in vs if v[2] is not None]
        best = max(scored, key=lambda v: v[2]) if scored else vs[1]
        normal = vs[1]
        pick = best if use else normal
        mult, levels, pred, feats = pick
        take = (not use) or (pred is not None and pred > a.nn_min_edge)
        c["nn"] = {"pred": pred, "feats": feats, "levels": levels}
        if use:
            c["grids"] = levels
        decided.append({"symbol": c["symbol"], "pred": best[2], "levels": best[1], "mult": best[0], "feats": best[3],
                        "take": take, "variants": vs})
        if take:
            out.append(c)
    if use:
        out.sort(key=lambda c: c["nn"]["pred"], reverse=True)
    decided.sort(key=lambda x: x["pred"] if x["pred"] is not None else -99, reverse=True)
    BRAIN.preds = [{k: d[k] for k in ("symbol", "pred", "levels", "mult", "take")} for d in decided]
    try:
        BRAIN.log_decisions(decided, use)
    except Exception as e:  # noqa: BLE001
        log.warning("Нейросеть: объяснение не получилось: %s", e)
    return out


# ---------------------------------------------------------------- grid

def positions(s, sym):
    """Все открытые ноги позиции: [(сторона, размер, positionIdx)]. В Hedge их может быть две."""
    r = api(s.get_positions, category=CAT, symbol=sym)
    out = []
    for p in r["result"]["list"]:
        size = D(p.get("size") or 0)
        if size > 0:
            out.append((p["side"], size, int(p.get("positionIdx") or 0)))
    return out


def position(s, sym):
    """Первая открытая нога (сторона, размер, positionIdx) или (None, 0, 0)."""
    legs = positions(s, sym)
    return legs[0] if legs else (None, D(0), 0)


def is_hedge(s, sym):
    """Режим позиций монеты: в Hedge Bybit отдаёт две записи с positionIdx 1 и 2."""
    r = api(s.get_positions, category=CAT, symbol=sym)
    return any(int(p.get("positionIdx") or 0) in (1, 2) for p in r["result"]["list"])


def chase_flat(s, sym, ins, a, timeout):
    """Закрыть позицию ТОЛЬКО лимитками с догоном (без маркета).

    Сначала мейкер-лимитка у лучшей цены (PostOnly), через --chase-sec снимаем и
    переставляем к новой цене. После --chase-tries попыток — быстрая лимитка через спред
    (IOC, цена ограничена --chase-slip тиками). Повтор, пока позиция не станет 0.
    """
    tick = ins["tick"]
    t_end = time.time() + timeout
    tries = 0
    while time.time() < t_end:
        side, size, idx = position(s, sym)
        if size <= 0:
            return True
        close_side = "Sell" if side == "Buy" else "Buy"
        ob = api(s.get_orderbook, category=CAT, symbol=sym, limit=1)["result"]
        bid, ask = D(ob["b"][0][0]), D(ob["a"][0][0])
        if tries < a.chase_tries:
            if close_side == "Sell":
                px = ask - tick if ask - tick > bid else ask
            else:
                px = bid + tick if bid + tick < ask else bid
            tif = "PostOnly"
        else:
            px = bid - tick * a.chase_slip if close_side == "Sell" else ask + tick * a.chase_slip
            tif = "IOC"
        tries += 1
        try:
            r = api(s.place_order, category=CAT, symbol=sym, side=close_side, orderType="Limit",
                    qty=str(size), price=str(px), timeInForce=tif, reduceOnly=True, positionIdx=idx)
            oid = r["result"]["orderId"]
        except Exception as e:  # noqa: BLE001
            log.warning("%s: лимитка закрытия не встала (%s), повтор", sym, str(e)[:120])
            time.sleep(0.5)
            continue
        log.info("%s: закрываю %s %s лимиткой %s @ %s (попытка %d)", sym, close_side, size, tif, px, tries)
        if tif == "PostOnly":
            time.sleep(a.chase_sec)
            try:
                s.cancel_order(category=CAT, symbol=sym, orderId=oid)
            except Exception:  # noqa: BLE001
                pass  # уже исполнена или отклонена
        else:
            time.sleep(0.3)
    return position(s, sym)[1] <= 0


class Grid:
    """Нейтральная сетка, которая идёт за ценой. Только лимитные ордера."""

    def __init__(self, s, a, cand, ins):
        self.s, self.a, self.ins = s, a, ins
        self.sym = cand["symbol"]
        self.n = cand["grids"]
        self.tick = ins["tick"]
        price = D(cand["price"])
        self.lev = min(a.leverage, ins["max_lev"])
        notional = D(a.budget) * D(self.lev) / D(self.n)
        self.qty = round_step(notional / price, ins["qty_step"])
        if self.qty < ins["min_qty"] or self.qty * price < ins["min_notional"]:
            raise ValueError(f"{self.sym}: бюджет мал для {self.n} уровней "
                             f"(qty {self.qty} < min {ins['min_qty']} или < {ins['min_notional']}$)")
        # максимум позиции в одну сторону = вся маржа × плечо
        self.max_pos = float(self.qty) * self.n
        self.set_levels(cand["lower"], cand["upper"])
        self.want = {}      # уровень -> (сторона, positionIdx): что должно стоять
        self.live = {}      # orderId -> (уровень, сторона, positionIdx)
        self.hedge = False  # режим Hedge: лонг через positionIdx=1, шорт через 2
        self.long, self.lavg = 0.0, 0.0
        self.short, self.savg = 0.0, 0.0
        self.realized = 0.0
        self.fills = 0
        self.shifts = 0
        self.t0 = time.time()
        self.out_since = None
        self.price = float(price)
        self.er_checked = 0.0
        self.closing = ""
        self.nn = cand.get("nn")
        self.notional = float(self.qty) * self.n * float(price)

    # --- уровни
    def set_levels(self, lo, hi):
        step = (D(hi) - D(lo)) / D(self.n)
        self.levels = [round_step(D(lo) + i * step, self.tick, ROUND_HALF_UP) for i in range(self.n + 1)]
        self.lower, self.upper = float(self.levels[0]), float(self.levels[-1])
        self.step_price = float(self.levels[-1] - self.levels[0]) / self.n

    def layout(self, price):
        """Buy ниже цены, Sell выше, уровень у цены пустой.
        Hedge: ближайшие к цене уровни сначала закрывают уже открытую ногу
        (Buy закрывает шорт, Sell закрывает лонг), остальные открывают новую."""
        empty = min(range(self.n + 1), key=lambda i: abs(float(self.levels[i]) - price))
        self.want = {}
        q = float(self.qty)
        close_short = int(round(self.short / q)) if self.hedge else 0
        close_long = int(round(self.long / q)) if self.hedge else 0
        for i in range(empty - 1, -1, -1):          # вниз от цены
            if not self.hedge:
                self.want[i] = ("Buy", 0)
            elif close_short > 0:
                self.want[i] = ("Buy", 2)
                close_short -= 1
            else:
                self.want[i] = ("Buy", 1)
        for i in range(empty + 1, self.n + 1):      # вверх от цены
            if not self.hedge:
                self.want[i] = ("Sell", 0)
            elif close_long > 0:
                self.want[i] = ("Sell", 1)
                close_long -= 1
            else:
                self.want[i] = ("Sell", 2)

    @staticmethod
    def closes(side, idx):
        """Ордер закрывает ногу в Hedge: Sell по лонгу (1) или Buy по шорту (2)."""
        return (side == "Sell" and idx == 1) or (side == "Buy" and idx == 2)

    @property
    def net(self):
        return self.long - self.short

    # --- учёт позиции и прибыли
    def book(self, side, idx, px, qty):
        """Учёт исполнения. One-Way: одна нетто-позиция; Hedge: лонг и шорт отдельно."""
        self.realized -= self.a.fee / 100 * px * qty
        if idx == 1 or (idx == 0 and side == "Buy" and self.short == 0) or (idx == 0 and side == "Sell" and self.long > 0):
            leg = "long"
        else:
            leg = "short"
        opening = (side == "Buy") == (leg == "long")
        if leg == "long":
            if opening:
                self.lavg = (self.lavg * self.long + px * qty) / (self.long + qty)
                self.long += qty
            else:
                c = min(qty, self.long)
                self.realized += (px - self.lavg) * c
                self.long -= c
                rest = qty - c
                if self.long <= 1e-12:
                    self.long, self.lavg = 0.0, 0.0
                if rest > 1e-12 and idx == 0:          # One-Way: переворот в шорт
                    self.short, self.savg = rest, px
        else:
            if opening:
                self.savg = (self.savg * self.short + px * qty) / (self.short + qty)
                self.short += qty
            else:
                c = min(qty, self.short)
                self.realized += (self.savg - px) * c
                self.short -= c
                rest = qty - c
                if self.short <= 1e-12:
                    self.short, self.savg = 0.0, 0.0
                if rest > 1e-12 and idx == 0:          # One-Way: переворот в лонг
                    self.long, self.lavg = rest, px

    def pnl(self):
        return self.realized + (self.price - self.lavg) * self.long + (self.savg - self.price) * self.short

    def publish(self):
        web.update(grid={
            "symbol": self.sym, "lower": self.lower, "upper": self.upper, "levels": self.n,
            "step_pct": self.step_price / ((self.lower + self.upper) / 2) * 100, "qty": str(self.qty),
            "open_orders": len(self.live), "fills": self.fills, "profit": self.pnl(),
            "net": (f"L {self.long:g} / S {self.short:g}" if self.hedge else self.net), "shifts": self.shifts, "started": self.t0, "closing": self.closing})

    # --- ордера
    def can_place(self, side, idx):
        """Не даём позиции вырасти больше маржи × плечо, даже если сетка ушла за ценой."""
        q = float(self.qty)
        if self.hedge:
            if self.closes(side, idx):
                return True
            same = sum(1 for _, sd, ix in self.live.values() if sd == side and ix == idx)
            cur = self.long if idx == 1 else self.short
            return cur + q * (same + 1) <= self.max_pos * 1.0001
        same = sum(1 for _, sd, _ix in self.live.values() if sd == side)
        proj = (self.net if side == "Buy" else -self.net) + q * (same + 1)
        return proj <= self.max_pos * 1.0001

    def place(self, i, side, idx):
        if self.a.dry:
            return
        kw = {}
        if self.closes(side, idx):
            kw["reduceOnly"] = True
        try:
            r = api(self.s.place_order, category=CAT, symbol=self.sym, side=side, orderType="Limit",
                    qty=str(self.qty), price=str(self.levels[i]), timeInForce="GTC", positionIdx=idx, **kw)
            self.live[r["result"]["orderId"]] = (i, side, idx)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if self.closes(side, idx) and ("110017" in msg or "reduce" in msg.lower()):
                # закрывать нечего (нога уже 0) — уровень переводим на открытие
                self.want[i] = (side, 2 if side == "Sell" else 1)
                return
            log.error("%s: не удалось поставить %s @ %s: %s", self.sym, side, self.levels[i], msg)

    def refill(self):
        """Поставить всё, чего не хватает в сетке (после исполнения, сдвига, снятия)."""
        placed = {i for i, _, _ in self.live.values()}
        for i, (side, idx) in sorted(self.want.items()):
            if i not in placed and self.can_place(side, idx):
                self.place(i, side, idx)

    def order_info(self, oid):
        r = api(self.s.get_order_history, category=CAT, symbol=self.sym, orderId=oid)
        lst = r["result"]["list"]
        return lst[0] if lst else None

    def on_done(self, oid, info):
        """Ордер сетки ушёл из открытых: исполнен -> встречный на соседний уровень."""
        i, side, idx = self.live.pop(oid)
        st = info["orderStatus"]
        done = float(info.get("cumExecQty") or 0)
        if done > 0:
            self.book(side, idx, float(info.get("avgPrice") or self.levels[i]), done)
        if st == "Filled":
            self.fills += 1
            # уровень освобождается, только если там всё ещё ждал этот ордер: если за ту же секунду
            # исполнился соседний и уже поставил сюда встречный, его не трогаем
            if self.want.get(i) == (side, idx):
                self.want.pop(i, None)
            # встречная лимитка на соседний уровень, по той же ноге (positionIdx)
            if side == "Buy" and i + 1 <= self.n:
                self.want[i + 1] = ("Sell", idx)
            elif side == "Sell" and i - 1 >= 0:
                self.want[i - 1] = ("Buy", idx)
            pos = f"L {self.long:g} / S {self.short:g}" if self.hedge else f"{self.net:g}"
            log.info("%s: исполнен %s @ %s | позиция %s | PnL≈%.3f$",
                     self.sym, side, self.levels[i], pos, self.pnl())

    def sync(self):
        r = api(self.s.get_open_orders, category=CAT, symbol=self.sym, limit=50)
        open_ids = {o["orderId"] for o in r["result"]["list"]}
        for oid in [x for x in self.live if x not in open_ids]:
            info = self.order_info(oid)
            if info is not None:
                self.on_done(oid, info)
        self.refill()
        self.publish()

    def cancel_grid(self):
        api(self.s.cancel_all_orders, category=CAT, symbol=self.sym)
        for oid in list(self.live):
            info = None
            try:
                info = self.order_info(oid)
            except Exception:  # noqa: BLE001
                pass
            if info:
                self.on_done(oid, info)
            else:
                self.live.pop(oid, None)

    def last_price(self):
        t = api(self.s.get_tickers, category=CAT, symbol=self.sym)["result"]["list"][0]
        self.price = float(t["lastPrice"])
        return self.price

    def follow(self, price):
        """Цена ушла за край сетки -> перестроить сетку той же ширины вокруг цены."""
        edge = self.step_price * self.a.shift_steps
        if self.lower - edge <= price <= self.upper + edge:
            self.out_since = None
            return
        if self.out_since is None:
            self.out_since = time.time()
        if time.time() - self.out_since < self.a.shift_sec:
            return
        half = (self.upper - self.lower) / 2
        self.cancel_grid()
        self.set_levels(price - half, price + half)
        self.layout(price)
        self.shifts += 1
        self.out_since = None
        log.info("%s: сетка пошла за ценой (сдвиг %d): %s … %s", self.sym, self.shifts, self.lower, self.upper)
        self.refill()

    # --- запуск / выход
    def start(self, price):
        if not self.a.dry:
            self.hedge = is_hedge(self.s, self.sym)
            log.info("%s: режим позиций %s", self.sym, "Hedge" if self.hedge else "One-Way")
        self.layout(price)
        log.info("СЕТКА %s: %s … %s, уровней %d, шаг %.3f%%, qty %s, плечо %sx",
                 self.sym, self.lower, self.upper, self.n, self.step_price / price * 100, self.qty, self.lev)
        if self.a.dry:
            for i, (side, idx) in sorted(self.want.items()):
                log.info("  [dry] %s %s @ %s", side, self.qty, self.levels[i])
            return
        api(self.s.set_leverage, category=CAT, symbol=self.sym,
            buyLeverage=str(self.lev), sellLeverage=str(self.lev))
        api(self.s.cancel_all_orders, category=CAT, symbol=self.sym)
        self.refill()
        log.info("%s: поставлено %d ордеров", self.sym, len(self.live))
        self.publish()

    def check_exit(self, price):
        if (time.time() - self.t0) / 60 >= self.a.max_minutes:
            return "истекло время"
        pnl = self.pnl()
        if self.a.tp_usdt > 0 and pnl >= self.a.tp_usdt:
            return f"тейк-профит {pnl:.2f}$"
        if self.a.max_loss > 0 and pnl <= -self.a.budget * self.a.max_loss / 100:
            return f"стоп по убытку {pnl:.2f}$"
        if self.a.exit_er > 0 and time.time() - self.er_checked > 30:
            self.er_checked = time.time()
            m = analyze(klines(self.s, self.sym, self.a.window + 1), self.a)
            if m and m["er"] > self.a.exit_er:
                return f"тренд (ER={m['er']:.2f})"
        return None

    def close(self, reason, timeout=None):
        """Снять сетку и закрыть позицию лимитками с догоном. True, если позиция закрыта."""
        if not self.closing:
            log.info("%s: ЗАКРЫТИЕ — %s", self.sym, reason)
        self.closing = reason
        self.publish()
        if self.a.dry:
            return True
        self.cancel_grid()
        ok = chase_flat(self.s, self.sym, self.ins, self.a, timeout or self.a.close_timeout)
        if not ok:
            log.error("%s: позиция ещё не закрыта, продолжаю догон лимитками", self.sym)
            return False
        log.info("%s: закрыто. Исполнений %d, сдвигов %d, PnL≈%.3f$", self.sym, self.fills, self.shifts, self.pnl())
        if BRAIN and self.nn and self.notional > 0:
            pnl = self.pnl() - self.a.fee / 100 * abs(self.net) * self.price
            BRAIN.add_real(self.nn["feats"], pnl / self.notional * 100, self.nn.get("pred"), self.sym)
            log.info("%s: результат записан в обучение нейросети (%+.3f%% от объёма)", self.sym, pnl / self.notional * 100)
        with web.LOCK:
            web.STATUS["history"].append({"time": time.strftime("%d.%m %H:%M UTC", time.gmtime()), "symbol": self.sym,
                                          "fills": self.fills, "profit": self.pnl(), "reason": reason})
            web.STATUS["history"] = web.STATUS["history"][-100:]
            web.STATUS["grid"] = None
        return True

    def has_position(self):
        """Есть ли уже позиция или ордера на этой монете (например, от другого бота на счёте)."""
        if positions(self.s, self.sym):
            return True
        o = api(self.s.get_open_orders, category=CAT, symbol=self.sym, limit=1)
        return bool(o["result"]["list"])


# ---------------------------------------------------------------- main

def normalize_args(text, parser):
    """BOT_ARGS из переменной Railway: понимает 'budget 15', '—budget 15', 'budget=15', 'tp-usdt 1'."""
    known = {o.lstrip("-") for act in parser._actions for o in act.option_strings}
    text = text.replace("—", "--").replace("–", "--").replace("−", "-")
    out = []
    for tok in shlex.split(text):
        name, eq, val = tok.lstrip("-").partition("=")
        name = name.replace("_", "-").lower()
        if name in known:
            out.append("--" + name)
            if eq:
                out.append(val)
        else:
            out.append(tok.replace(",", "."))   # 15,5 -> 15.5
    return out


def check_access(s, a):
    """Проверка, что Bybit доступен с этого сервера и ключи рабочие. Возвращает текст ошибки или ''."""
    try:
        s.get_server_time()
    except Exception as e:  # noqa: BLE001
        return ("Bybit недоступен с этого сервера: %s\n"
                "Скорее всего IP облака заблокирован Bybit (403). Задай переменную HTTPS_PROXY "
                "(http:// или socks5:// прокси в разрешённой стране) или смени регион Railway." % e)
    if a.dry:
        return ""
    try:
        bal = balance(s)
        log.info("Ключи OK, баланс USDT: %s", bal)
    except Exception as e:  # noqa: BLE001
        return "Ключи не работают: %s (проверь ключ, права Contract trade, привязку IP, testnet/mainnet)" % e
    return ""


def balance(s):
    r = s.get_wallet_balance(accountType="UNIFIED", coin="USDT")
    coins = r["result"]["list"][0]["coin"]
    bal = float(next((c.get("walletBalance") for c in coins if c["coin"] == "USDT"), 0) or 0)
    web.update(balance=bal)
    return bal


def pause_wait(sec):
    """Ждать sec секунд, но выйти сразу, если нажали «Стоп»."""
    end = time.time() + sec
    while time.time() < end and not web.CONTROL["paused"]:
        time.sleep(1)


def wait_forever(msg):
    """Фатальная ошибка: не выходим (иначе пропадёт страница), показываем её и ждём."""
    log.error(msg)
    web.update(error=msg, state="остановлен — нужна правка настроек")
    while True:
        time.sleep(3600)


def main():
    p = argparse.ArgumentParser(description="Bybit volatility scanner + neutral grid bot")
    p.add_argument("--testnet", action="store_true", help="учебный счёт (по умолчанию реальный)")
    p.add_argument("--live", action="store_true", help="оставлено для совместимости: реальный счёт и так по умолчанию")
    p.add_argument("--dry", action="store_true", help="без ордеров, только показать план")
    p.add_argument("--budget", type=float, default=50, help="маржа на одну сетку, USDT")
    p.add_argument("--leverage", type=float, default=3)
    p.add_argument("--max-levels", type=int, default=20, help="макс. уровней сетки (≤40)")
    p.add_argument("--max-loss", type=float, default=30, help="стоп: убыток в %% от маржи (0 = выкл)")
    p.add_argument("--tp-usdt", type=float, default=0, help="закрыть при прибыли ≥ X$ (0 = выкл)")
    p.add_argument("--max-minutes", type=float, default=60, help="макс. жизнь сетки, мин")
    p.add_argument("--exit-er", type=float, default=0, help="закрыть, если ER выше (тренд); 0 = выкл")
    p.add_argument("--poll", type=float, default=1, help="опрос ордеров, сек")
    # сетка идёт за ценой
    p.add_argument("--shift-steps", type=float, default=1, help="сдвиг, когда цена дальше края на N шагов")
    p.add_argument("--shift-sec", type=float, default=0, help="сколько сек цена должна быть за краем до сдвига")
    # закрытие только лимитками с догоном
    p.add_argument("--chase-sec", type=float, default=1.5, help="сколько ждать мейкер-лимитку перед перестановкой")
    p.add_argument("--chase-tries", type=int, default=5, help="мейкер-попыток, потом быстрая лимитка через спред")
    p.add_argument("--chase-slip", type=int, default=5, help="быстрая лимитка: макс. тиков за лучшей ценой")
    p.add_argument("--close-timeout", type=float, default=120, help="сек на одну серию догона")
    p.add_argument("--cooldown", type=float, default=30, help="пауза для пары после закрытия, мин")
    # сканер
    p.add_argument("--window", type=int, default=15, help="окно, 1m свечей")
    p.add_argument("--min-vol", type=float, default=10e6, help="мин. оборот 24ч, USDT")
    p.add_argument("--min-vol-win", type=float, default=300e3, help="мин. оборот за окно, USDT")
    p.add_argument("--min-1m", type=float, default=0.5, help="мин. размах за минуту, %%")
    p.add_argument("--max-er", type=float, default=0.5)
    p.add_argument("--fee", type=float, default=0.055, help="комиссия, %% (taker Bybit 0.055)")
    p.add_argument("--pad", type=float, default=0.3, help="запас к границам, %%")
    p.add_argument("--keep-on-exit", action="store_true", help="Ctrl+C не закрывает сетку")
    p.add_argument("--term-timeout", type=float, default=25, help="сек на догон при остановке сервиса")
    # нейросеть
    p.add_argument("--no-nn", dest="nn", action="store_false", help="не давать нейросети принимать решения (учится всё равно)")
    p.add_argument("--nn-min-edge", type=float, default=0.02, help="брать сетку, если прогноз выше X %% от объёма")
    p.add_argument("--nn-symbols", type=int, default=30, help="монет для учебных сеток")
    p.add_argument("--nn-stride", type=int, default=5, help="шаг (мин) между учебными сетками")
    p.add_argument("--nn-refresh", type=float, default=60, help="раз в N минут — новые учебные сетки")
    p.add_argument("--nn-retrain-every", type=int, default=500, help="переобучать после N новых примеров")
    p.add_argument("--nn-epochs", type=int, default=60)
    p.add_argument("--nn-max-samples", type=int, default=60000)
    p.add_argument("--nn-path", default=os.getenv("NN_PATH") or ("/data/nn" if os.path.isdir("/data") else "data/nn"))
    # На Railway параметры удобно задавать переменной BOT_ARGS, например:
    #   BOT_ARGS=--budget 30 --leverage 3 --tp-usdt 1
    argv = sys.argv[1:] or normalize_args(os.getenv("BOT_ARGS", ""), p)
    a, unknown = p.parse_known_args(argv)
    a.max_levels = max(2, min(40, a.max_levels))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout, force=True)
    load_env()
    web.start()
    if unknown:
        log.warning("BOT_ARGS: не понял и пропускаю: %s", " ".join(unknown))

    if os.getenv("BYBIT_ENV", "").lower() == "testnet":
        a.testnet = True
    live = not a.testnet
    web.update(mode=("РЕАЛЬНЫЙ СЧЁТ" if live else "TESTNET") + (" · dry-run" if a.dry else ""),
               budget=a.budget, leverage=a.leverage, state="проверка доступа")

    key, sec = os.getenv("BYBIT_API_KEY"), os.getenv("BYBIT_API_SECRET")
    if not a.dry and not (key and sec):
        wait_forever("Нет ключей: добавь в Railway → Variables переменные BYBIT_API_KEY и BYBIT_API_SECRET")

    # Railway при редеплое/остановке шлёт SIGTERM -> закрываем сетку как по Ctrl+C
    def _term(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)

    s = HTTP(testnet=not live, api_key=key, api_secret=sec)
    log.info("Режим: %s%s | маржа на сетку %s$, плечо %sx", "РЕАЛЬНЫЙ СЧЁТ" if live else "TESTNET",
             " (dry-run)" if a.dry else "", a.budget, a.leverage)
    err = check_access(s, a)
    if err:
        wait_forever(err)
    inst = load_instruments(s)
    log.info("Инструментов: %d", len(inst))

    global BRAIN
    market = HTTP(testnet=False)   # учебные сетки — по реальным свечам

    def liquid():
        t = api(market.get_tickers, category=CAT)["result"]["list"]
        t = [x for x in t if x["symbol"] in inst and float(x.get("turnover24h") or 0) >= a.min_vol]
        t.sort(key=lambda x: float(x.get("turnover24h") or 0), reverse=True)
        return [x["symbol"] for x in t]
    BRAIN = nn.Brain(a.nn_path, a, market, analyze, klines, liquid)
    BRAIN.start()
    web.BRAIN = BRAIN
    last_bal = time.time()

    cooldown = {}
    grid = None
    try:
        while True:
            if web.CONTROL["paused"]:
                web.update(state="на паузе — нажми «Старт» на странице")
                if web.CONTROL["msg"].startswith("Остановка"):
                    web.CONTROL["msg"] = "Бот на паузе, ордера сняты, позиция закрыта"
                time.sleep(1)
                continue
            web.update(state="сканирую рынок", error="")
            try:
                cands = scan(s, inst, a)
            except Exception as e:  # noqa: BLE001
                log.error("Ошибка скана: %s", e)
                web.update(error="Ошибка скана: %s" % e)
                time.sleep(30)
                continue
            if web.CONTROL["paused"]:
                continue
            now = time.time()
            cands = [c for c in cands if cooldown.get(c["symbol"], 0) < now]
            grid = None
            for c in cands:
                try:
                    g = Grid(s, a, c, inst[c["symbol"]])
                    if not a.dry and g.has_position():
                        log.info("%s: уже есть позиция или ордера, пропускаю", c["symbol"])
                        continue
                    grid = g
                    break
                except ValueError as e:
                    log.info("%s", e)
            if not grid:
                log.info("Подходящих пар нет, жду 60 сек")
                web.update(state="подходящих монет нет, жду")
                pause_wait(60)
                continue

            web.update(state="ставлю сетку " + grid.sym)
            grid.start(next(c["price"] for c in cands if c["symbol"] == grid.sym))
            web.update(state="сетка работает: " + grid.sym)
            if a.dry:
                cooldown[grid.sym] = now + a.cooldown * 60
                grid = None
                pause_wait(60)
                continue

            while True:
                time.sleep(a.poll)
                reason = "стоп со страницы" if web.CONTROL["paused"] else grid.closing
                if not reason:
                    try:
                        grid.sync()
                        price = grid.last_price()
                        grid.follow(price)
                        reason = grid.check_exit(price)
                        if time.time() - last_bal > 60:
                            last_bal = time.time()
                            balance(s)
                    except Exception as e:  # noqa: BLE001
                        log.error("Ошибка цикла: %s", e)
                        continue
                if reason:
                    try:
                        if grid.close(reason):
                            break
                    except Exception as e:  # noqa: BLE001
                        log.error("Ошибка закрытия: %s — повтор", e)
            cooldown[grid.sym] = time.time() + a.cooldown * 60
            grid = None
    except KeyboardInterrupt:
        log.info("Остановка…")
        if grid and not a.keep_on_exit and not a.dry:
            web.update(state="перезапуск: закрываю позицию лимитками")
            grid.close("перезапуск/остановка сервиса", timeout=a.term_timeout)


if __name__ == "__main__":
    main()
