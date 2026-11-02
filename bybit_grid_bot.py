#!/usr/bin/env python3
"""
Bybit: сканер волатильности + авто нейтральная сетка (USDT perpetual, one-way mode).

Что делает:
  1. Сканирует все USDT-перпетуалы Bybit, ищет пары с большим размахом за последнюю минуту
     и без тренда (низкий Efficiency Ratio).
  2. Берёт лучшую пару, считает границы и уровни сетки.
  3. Ставит лимитки: Buy ниже цены, Sell выше, один уровень у цены пустой.
  4. Каждые N секунд проверяет исполнения: Buy на уровне i исполнен -> Sell на i+1,
     Sell на уровне i исполнен -> Buy на i-1. Это и есть нейтральная сетка.
  5. Закрывает сетку (отмена ордеров + закрытие позиции по рынку), если:
       - цена ушла за границу дальше чем на --stop %
       - достигнут --tp-usdt (оценка прибыли)
       - прошло --max-minutes
       - пара перешла в тренд (ER > --exit-er)
  6. Пауза для этой пары и поиск следующей.

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

Аккаунт должен быть в One-Way Mode (не Hedge) для USDT perpetual.
Ctrl+C: бот отменит ордера и закроет позицию текущей сетки (если не указан --keep-on-exit).
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
    return (x / step).to_integral_value(rounding=mode) * step


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
            m = analyze(klines(s, sym, a.window + 1), a)
        except Exception:  # noqa: BLE001
            return None
        if m:
            m["symbol"] = sym
        return m

    with ThreadPoolExecutor(max_workers=6) as ex:
        res = [r for r in ex.map(one, syms) if r]
    res = [r for r in res if r["er"] <= a.max_er and r["turnover"] >= a.min_vol_win
           and r["vol1m"] >= a.min_1m]
    res.sort(key=lambda r: r["score"], reverse=True)
    log.info("Скан: %d пар по объёму, %d прошли фильтры", len(syms), len(res))
    for r in res[:5]:
        log.info("  %-14s 1m=%.2f%% ATR=%.2f%% ER=%.2f score=%.2f",
                 r["symbol"], r["vol1m"], r["atr"], r["er"], r["score"])
    return res


# ---------------------------------------------------------------- grid

class Grid:
    def __init__(self, s, a, cand, ins):
        self.s, self.a, self.ins = s, a, ins
        self.sym = cand["symbol"]
        n = cand["grids"]
        lo, hi = cand["lower"], cand["upper"]
        step = (hi - lo) / n
        self.levels = [round_step(lo + i * step, ins["tick"], ROUND_HALF_UP) for i in range(n + 1)]
        self.lower, self.upper = float(self.levels[0]), float(self.levels[-1])
        self.step_price = (self.levels[-1] - self.levels[0]) / n
        price = D(cand["price"])

        self.lev = min(a.leverage, ins["max_lev"])
        notional = D(a.budget) * D(self.lev) / D(n)
        self.qty = round_step(notional / price, ins["qty_step"])
        if self.qty < ins["min_qty"] or self.qty * price < ins["min_notional"]:
            raise ValueError(f"{self.sym}: бюджет мал для {n} уровней "
                             f"(qty {self.qty} < min {ins['min_qty']} или < {ins['min_notional']}$)")

        self.orders = {}          # orderId -> (level_idx, side, is_pair)
        self.realized_est = 0.0
        self.fills = 0
        self.t0 = time.time()

    # --- orders
    def place(self, i, side, is_pair):
        price = self.levels[i]
        if self.a.dry:
            return
        try:
            r = api(self.s.place_order, category=CAT, symbol=self.sym, side=side, orderType="Limit",
                    qty=str(self.qty), price=str(price), timeInForce="GTC", positionIdx=0)
            self.orders[r["result"]["orderId"]] = (i, side, is_pair)
        except Exception as e:  # noqa: BLE001
            log.error("%s: не удалось поставить %s @ %s: %s", self.sym, side, price, e)

    def start(self, price):
        log.info("СЕТКА %s: %s … %s, уровней %d, шаг %.3f%%, qty %s, плечо %sx",
                 self.sym, self.lower, self.upper, len(self.levels) - 1,
                 float(self.step_price) / price * 100, self.qty, self.lev)
        empty = min(range(len(self.levels)), key=lambda i: abs(float(self.levels[i]) - price))
        plan = [(i, "Buy" if i < empty else "Sell") for i in range(len(self.levels)) if i != empty]
        if self.a.dry:
            for i, side in plan:
                log.info("  [dry] %s %s @ %s", side, self.qty, self.levels[i])
            return
        api(self.s.set_leverage, category=CAT, symbol=self.sym,
            buyLeverage=str(self.lev), sellLeverage=str(self.lev))
        api(self.s.cancel_all_orders, category=CAT, symbol=self.sym)
        for i, side in plan:
            self.place(i, side, is_pair=False)
        log.info("%s: поставлено %d ордеров", self.sym, len(self.orders))

    def order_status(self, oid):
        r = api(self.s.get_order_history, category=CAT, symbol=self.sym, orderId=oid)
        lst = r["result"]["list"]
        return lst[0]["orderStatus"] if lst else None

    def sync(self):
        r = api(self.s.get_open_orders, category=CAT, symbol=self.sym, limit=50)
        open_ids = {o["orderId"] for o in r["result"]["list"]}
        for oid in [x for x in self.orders if x not in open_ids]:
            st = self.order_status(oid)
            if st is None:
                continue                      # история ещё не обновилась
            i, side, is_pair = self.orders.pop(oid)
            if st == "Filled":
                self.fills += 1
                if is_pair:
                    self.realized_est += float(self.qty * self.step_price) \
                        - 2 * self.a.fee / 100 * float(self.qty * self.levels[i])
                log.info("%s: исполнен %s @ %s (сделок %d, прибыль≈%.3f$)",
                         self.sym, side, self.levels[i], self.fills, self.realized_est)
                if side == "Buy" and i + 1 < len(self.levels):
                    self.place(i + 1, "Sell", is_pair=True)
                elif side == "Sell" and i - 1 >= 0:
                    self.place(i - 1, "Buy", is_pair=True)
            elif st in ("Cancelled", "Rejected", "Deactivated"):
                log.warning("%s: ордер %s @ %s %s, ставлю заново", self.sym, side, self.levels[i], st)
                self.place(i, side, is_pair)

    def unrealised(self):
        r = api(self.s.get_positions, category=CAT, symbol=self.sym)
        return sum(float(p.get("unrealisedPnl") or 0) for p in r["result"]["list"])

    def check_exit(self):
        price = float(api(self.s.get_tickers, category=CAT, symbol=self.sym)["result"]["list"][0]["lastPrice"])
        if price < self.lower * (1 - self.a.stop / 100) or price > self.upper * (1 + self.a.stop / 100):
            return f"цена {price} вышла за диапазон"
        if (time.time() - self.t0) / 60 >= self.a.max_minutes:
            return "истекло время"
        if self.a.tp_usdt > 0:
            total = self.realized_est + self.unrealised()
            if total >= self.a.tp_usdt:
                return f"тейк-профит {total:.2f}$"
        m = analyze(klines(self.s, self.sym, self.a.window + 1), self.a)
        if m and m["er"] > self.a.exit_er:
            return f"тренд (ER={m['er']:.2f})"
        return None

    def close(self, reason):
        log.info("%s: ЗАКРЫТИЕ — %s", self.sym, reason)
        if self.a.dry:
            return
        api(self.s.cancel_all_orders, category=CAT, symbol=self.sym)
        r = api(self.s.get_positions, category=CAT, symbol=self.sym)
        for p in r["result"]["list"]:
            if float(p.get("size") or 0) > 0:
                side = "Sell" if p["side"] == "Buy" else "Buy"
                api(self.s.place_order, category=CAT, symbol=self.sym, side=side, orderType="Market",
                    qty=p["size"], reduceOnly=True, positionIdx=int(p.get("positionIdx", 0)))
                log.info("%s: позиция %s %s закрыта по рынку", self.sym, p["side"], p["size"])
        log.info("%s: итог по оценке: сделок %d, сеточная прибыль≈%.3f$", self.sym, self.fills, self.realized_est)

    def has_position(self):
        """Есть ли уже позиция или ордера на этой монете (например, от другого бота на счёте)."""
        r = api(self.s.get_positions, category=CAT, symbol=self.sym)
        if any(float(p.get("size") or 0) > 0 for p in r["result"]["list"]):
            return True
        o = api(self.s.get_open_orders, category=CAT, symbol=self.sym, limit=1)
        return bool(o["result"]["list"])


# ---------------------------------------------------------------- main

def check_access(s, a):
    """Проверка, что Bybit доступен с этого сервера и ключи рабочие."""
    try:
        s.get_server_time()
    except Exception as e:  # noqa: BLE001
        sys.exit("Bybit недоступен с этого сервера: %s\n"
                 "Скорее всего IP облака заблокирован Bybit (403). Задай переменную HTTPS_PROXY "
                 "(http:// или socks5:// прокси в разрешённой стране) или смени регион/хостинг." % e)
    if a.dry:
        return
    try:
        r = s.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        coins = r["result"]["list"][0]["coin"]
        bal = next((c.get("walletBalance") for c in coins if c["coin"] == "USDT"), "0")
        log.info("Ключи OK, баланс USDT: %s", bal)
    except Exception as e:  # noqa: BLE001
        sys.exit("Ключи не работают: %s (проверь ключ, права Contract trade, привязку IP, testnet/mainnet)" % e)


def main():
    p = argparse.ArgumentParser(description="Bybit volatility scanner + neutral grid bot")
    p.add_argument("--testnet", action="store_true", help="учебный счёт (по умолчанию реальный)")
    p.add_argument("--live", action="store_true", help="оставлено для совместимости: реальный счёт и так по умолчанию")
    p.add_argument("--dry", action="store_true", help="без ордеров, только показать план")
    p.add_argument("--budget", type=float, default=50, help="маржа на одну сетку, USDT")
    p.add_argument("--leverage", type=float, default=3)
    p.add_argument("--max-levels", type=int, default=20, help="макс. уровней сетки (≤40)")
    p.add_argument("--stop", type=float, default=0.5, help="стоп: %% за границей сетки")
    p.add_argument("--tp-usdt", type=float, default=0, help="закрыть при прибыли ≥ X$ (0 = выкл)")
    p.add_argument("--max-minutes", type=float, default=60, help="макс. жизнь сетки, мин")
    p.add_argument("--exit-er", type=float, default=0.75, help="закрыть, если ER выше (тренд)")
    p.add_argument("--poll", type=float, default=3, help="опрос ордеров, сек")
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
    # На Railway параметры удобно задавать переменной BOT_ARGS, например:
    #   BOT_ARGS=--budget 30 --leverage 3 --tp-usdt 1
    argv = sys.argv[1:] or shlex.split(os.getenv("BOT_ARGS", ""))
    a = p.parse_args(argv)
    a.max_levels = max(2, min(40, a.max_levels))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout, force=True)
    load_env()

    key, sec = os.getenv("BYBIT_API_KEY"), os.getenv("BYBIT_API_SECRET")
    if not a.dry and not (key and sec):
        sys.exit("Нет ключей: создай .env с BYBIT_API_KEY и BYBIT_API_SECRET (или запусти с --dry)")
    if os.getenv("BYBIT_ENV", "").lower() == "testnet":
        a.testnet = True
    live = not a.testnet

    # Railway при редеплое/остановке шлёт SIGTERM -> закрываем сетку как по Ctrl+C
    def _term(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)

    s = HTTP(testnet=not live, api_key=key, api_secret=sec)
    log.info("Режим: %s%s | маржа на сетку %s$, плечо %sx", "РЕАЛЬНЫЙ СЧЁТ" if live else "TESTNET",
             " (dry-run)" if a.dry else "", a.budget, a.leverage)
    check_access(s, a)
    inst = load_instruments(s)
    log.info("Инструментов: %d", len(inst))

    cooldown = {}
    grid = None
    try:
        while True:
            cands = scan(s, inst, a)
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
                time.sleep(60)
                continue

            grid.start(next(c["price"] for c in cands if c["symbol"] == grid.sym))
            if a.dry:
                cooldown[grid.sym] = now + a.cooldown * 60
                grid = None
                time.sleep(60)
                continue

            while True:
                time.sleep(a.poll)
                try:
                    grid.sync()
                    reason = grid.check_exit()
                except Exception as e:  # noqa: BLE001
                    log.error("Ошибка цикла: %s", e)
                    continue
                if reason:
                    grid.close(reason)
                    break
            cooldown[grid.sym] = time.time() + a.cooldown * 60
            grid = None
    except KeyboardInterrupt:
        log.info("Остановка…")
        if grid and not a.keep_on_exit and not a.dry:
            grid.close("ручная остановка")


if __name__ == "__main__":
    main()
