#!/usr/bin/env python3
"""NY Breakout Autopilot для Bybit (USDT perpetual): BTC, ETH, SOL.

Каждый будний день в окно открытия Нью-Йорка (16:30–17:30 по Киеву) бот ждёт пробой и ретест
вчерашнего максимума/минимума (PDH/PDL) с сильной сигнальной свечой, входит лимитками с догоном,
сразу ставит на биржу тейк (лимитка, 3R) и стоп-лимит (−1R), на +2R переносит стоп в безубыток.
Маркет-ордеров нет. Риск — % от баланса на сделку.

Запуск:  python ny_bot.py            (реальный счёт)
         python ny_bot.py --testnet  (учебный счёт)
Страница бота: http://localhost:8080 (на Railway — ссылка сервиса).
"""

import argparse
import collections
import json
import logging
import math
import os
import signal as sig
import sys
import threading
import time
from datetime import datetime, timezone

from pybit.unified_trading import HTTP

import strategy as st
import web
from bybit_util import (CAT, D, api, best_prices, cancel, chase_close, instrument, is_hedge, klines, last_price,
                        load_env, normalize_args, order_info, positions, round_step, wallet)

log = logging.getLogger("bot")
UTC = timezone.utc


def now_ms():
    return int(time.time() * 1000)


class Bot:
    def __init__(self, s, a, p, live):
        self.s, self.a, self.p, self.live = s, a, p, live
        self.syms = [x.strip().upper() for x in a.symbols.split(",") if x.strip()]
        self.public = HTTP(testnet=False)        # рыночные данные для бэктеста — всегда реальные
        self.ins = {}
        self.lv = {x: {"day": None, "pdh": 0.0, "pdl": 0.0} for x in self.syms}
        self.price = {x: 0.0 for x in self.syms}
        self.last_t = {x: 0 for x in self.syms}
        self.check = {x: None for x in self.syms}
        self.trades = {}                          # открытые сделки: symbol -> dict
        self.journal = []                         # закрытые сделки
        self.taken = {}                           # (symbol, день NY) -> сделок
        self.events = collections.deque(maxlen=80)
        self.lock = threading.RLock()
        self.cmds = collections.deque()
        self.paused = os.getenv("START_PAUSED", "") == "1"
        self.balance = self.avail = 0.0
        self.error = ""
        self.state_path = a.state_path
        self.bt_cache = {}
        self.min_cache = {}
        self.load()

    # ------------------------------------------------------------ состояние
    def event(self, text, kind="info"):
        with self.lock:
            self.events.appendleft({"t": now_ms(), "text": text, "kind": kind})
        (log.warning if kind == "warn" else log.info)(text)

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
            tmp = self.state_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"trades": self.trades, "journal": self.journal[-500:],
                           "taken": [[k[0], k[1], v] for k, v in self.taken.items()]}, f, default=str)
            os.replace(tmp, self.state_path)
        except Exception as e:  # noqa: BLE001
            log.warning("Не удалось сохранить состояние: %s", e)

    def load(self):
        try:
            with open(self.state_path, encoding="utf-8") as f:
                data = json.load(f)
            self.trades = data.get("trades") or {}
            self.journal = data.get("journal") or []
            self.taken = {(k[0], k[1]): k[2] for k in data.get("taken") or []}
            if self.trades:
                log.info("Восстановлены открытые сделки: %s", ", ".join(self.trades))
        except FileNotFoundError:
            pass
        except Exception as e:  # noqa: BLE001
            log.warning("Состояние не прочитано: %s", e)

    # ------------------------------------------------------------ данные
    def refresh_wallet(self):
        try:
            self.balance, self.avail = wallet(self.s)
        except Exception as e:  # noqa: BLE001
            log.warning("Баланс не получен: %s", e)

    def refresh_levels(self, sym):
        day = datetime.now(UTC).date().isoformat()
        if self.lv[sym]["day"] == day:
            return
        d = klines(self.s, sym, "D", limit=3)
        prev = d[-2]
        self.lv[sym] = {"day": day, "pdh": prev["h"], "pdl": prev["l"]}
        self.event(f"{sym}: уровни на сегодня PDH {prev['h']:g} / PDL {prev['l']:g}")

    def new_closed_candle(self, sym):
        """Последняя закрытая 1m свеча, если она новая."""
        minute = now_ms() // 60_000 * 60_000
        if self.last_t[sym] >= minute - 60_000:
            return None
        ks = klines(self.s, sym, "1", limit=3)
        closed = [k for k in ks if k["t"] + 60_000 <= now_ms()]
        if not closed:
            return None
        c = closed[-1]
        if c["t"] <= self.last_t[sym]:
            return None
        self.last_t[sym] = c["t"]
        lv = self.lv[sym]
        self.check[sym] = {"candle": c, "items": st.checklist(c, lv["pdh"], lv["pdl"], self.p)}
        return c

    # ------------------------------------------------------------ вход
    def idx_for(self, sym, side):
        if is_hedge(self.s, sym):
            return 1 if side == "Buy" else 2
        return 0

    def enter(self, sym, side, c):
        a, p, ins = self.a, self.p, self.ins[sym]
        ref = c["c"]
        if positions(self.s, sym):
            self.event(f"{sym}: сигнал {side}, но на монете уже есть позиция — пропускаю", "warn")
            return
        self.refresh_wallet()
        risk_usd = self.balance * p.risk_pct / 100
        sl_dist = ref * p.sl_pct / 100
        qty = round_step(risk_usd / sl_dist, ins["qty_step"])
        if qty < ins["min_qty"] or qty * D(ref) < ins["min_notional"]:
            self.event(f"{sym}: риск {risk_usd:.2f}$ слишком мал для минимального объёма — пропуск", "warn")
            return
        notional = float(qty) * ref
        lev = max(1, math.ceil(notional / max(self.avail * 0.9, 1e-9)) + 1)
        lev = min(lev, int(ins["max_lev"]), a.max_leverage)
        if notional / lev > self.avail * 0.95:
            self.event(f"{sym}: не хватает маржи на {notional:.0f}$ при плече до x{a.max_leverage} — пропуск", "warn")
            return
        idx = self.idx_for(sym, side)
        api(self.s.set_leverage, category=CAT, symbol=sym, buyLeverage=str(lev), sellLeverage=str(lev))
        self.event(f"{sym}: СИГНАЛ {'LONG' if side == 'Buy' else 'SHORT'} @ {ref:g}, риск {risk_usd:.2f}$, "
                   f"объём {qty}, плечо x{lev}", "signal")

        # вход лимитками с догоном; дальше чем entry_slip_r × риск от цены сигнала не гонимся
        tick = ins["tick"]
        cap = D(ref) + D(sl_dist * a.entry_slip_r) * (1 if side == "Buy" else -1)
        filled, cost, tries = D(0), D(0), 0
        t_end = time.time() + a.entry_timeout
        while filled < qty and time.time() < t_end:
            bid, ask = best_prices(self.s, sym)
            if (side == "Buy" and bid > cap) or (side == "Sell" and ask < cap):
                break
            rest = round_step(qty - filled, ins["qty_step"])
            if rest < ins["min_qty"]:
                break
            if tries < a.entry_maker_tries:
                px, tif = (bid if side == "Buy" else ask), "PostOnly"
            else:
                px = min(ask + tick * a.chase_slip, cap) if side == "Buy" else max(bid - tick * a.chase_slip, cap)
                tif = "IOC"
            tries += 1
            try:
                r = api(self.s.place_order, category=CAT, symbol=sym, side=side, orderType="Limit", qty=str(rest),
                        price=str(round_step(px, tick)), timeInForce=tif, positionIdx=idx)
            except Exception as e:  # noqa: BLE001
                log.warning("%s: лимитка входа не встала: %s", sym, str(e)[:120])
                time.sleep(0.4)
                continue
            oid = r["result"]["orderId"]
            time.sleep(a.chase_sec if tif == "PostOnly" else 0.3)
            if tif == "PostOnly":
                cancel(self.s, sym, oid)
            info = order_info(self.s, sym, oid) or {}
            q = D(info.get("cumExecQty") or 0)
            if q > 0:
                filled += q
                cost += q * D(info.get("avgPrice") or px)
        if filled <= 0:
            self.event(f"{sym}: не успел войти лимиткой — цена ушла дальше {a.entry_slip_r}R, сигнал пропущен", "warn")
            return
        entry = float(cost / filled)
        lv = st.levels_for(side, entry, p)
        tr = {
            "symbol": sym, "side": side, "idx": idx, "qty": str(filled), "entry": entry, "risk_usd": risk_usd,
            "sl": lv["sl"], "tp": lv["tp"], "be_trigger": lv["be_trigger"], "be_stop": lv["be_stop"],
            "stop": lv["sl"], "be": False, "t_open": now_ms(), "signal_t": c["t"], "lev": lev,
            "pdh": self.lv[sym]["pdh"], "pdl": self.lv[sym]["pdl"], "tp_oid": None, "sl_oid": None,
            "out_since": None, "close_reason": "",
        }
        self.place_tp(tr)
        self.place_stop(tr, lv["sl"])
        with self.lock:
            self.trades[sym] = tr
        day = st.ny_day(c["t"]).isoformat()
        self.taken[(sym, day)] = self.taken.get((sym, day), 0) + 1
        self.save()
        self.event(f"{sym}: ВХОД {'LONG' if side == 'Buy' else 'SHORT'} {filled} @ {entry:g} | SL {lv['sl']:g} | "
                   f"Б/У на {lv['be_trigger']:g} | TP {lv['tp']:g}", "trade")

    # ------------------------------------------------------------ защитные ордера
    def close_side(self, tr):
        return "Sell" if tr["side"] == "Buy" else "Buy"

    def place_tp(self, tr):
        ins = self.ins[tr["symbol"]]
        try:
            r = api(self.s.place_order, category=CAT, symbol=tr["symbol"], side=self.close_side(tr),
                    orderType="Limit", qty=tr["qty"], price=str(round_step(tr["tp"], ins["tick"])),
                    timeInForce="GTC", reduceOnly=True, positionIdx=tr["idx"])
            tr["tp_oid"] = r["result"]["orderId"]
        except Exception as e:  # noqa: BLE001
            self.event(f"{tr['symbol']}: тейк не встал: {e}", "warn")

    def place_stop(self, tr, stop_px):
        """Стоп-лимит на бирже: срабатывает по последней цене, исполняется лимиткой."""
        ins = self.ins[tr["symbol"]]
        long = tr["side"] == "Buy"
        trig = round_step(stop_px, ins["tick"])
        slip = D(self.a.stop_limit_slip) / 100
        px = round_step(trig * (1 - slip) if long else trig * (1 + slip), ins["tick"])
        try:
            r = api(self.s.place_order, category=CAT, symbol=tr["symbol"], side=self.close_side(tr),
                    orderType="Limit", qty=tr["qty"], price=str(px), triggerPrice=str(trig),
                    triggerDirection=2 if long else 1, triggerBy="LastPrice", timeInForce="GTC",
                    reduceOnly=True, positionIdx=tr["idx"])
            tr["sl_oid"] = r["result"]["orderId"]
            tr["stop"] = float(trig)
        except Exception as e:  # noqa: BLE001
            tr["sl_oid"] = None
            tr["stop"] = float(trig)
            self.event(f"{tr['symbol']}: стоп на бирже не встал ({e}) — стоп держит бот", "warn")

    # ------------------------------------------------------------ сопровождение
    def leg(self, tr):
        for side, size, idx, avg in positions(self.s, tr["symbol"]):
            if idx == tr["idx"] and (idx != 0 or side == tr["side"]):
                return size
        return D(0)

    def finish(self, tr, reason=None):
        sym = tr["symbol"]
        for oid in (tr.get("tp_oid"), tr.get("sl_oid")):
            if oid:
                cancel(self.s, sym, oid)
        tp_info = order_info(self.s, sym, tr["tp_oid"]) if tr.get("tp_oid") else None
        if reason:
            outcome = reason
        elif tp_info and tp_info.get("orderStatus") == "Filled":
            outcome = "TP"
        else:
            outcome = "BE" if tr["be"] else "SL"
        pnl = None
        try:
            r = api(self.s.get_closed_pnl, category=CAT, symbol=sym, startTime=tr["t_open"] - 1000, limit=50)
            rows = [x for x in r["result"]["list"] if int(x.get("updatedTime") or x.get("createdTime") or 0) >= tr["t_open"] - 1000]
            if rows:
                pnl = sum(float(x["closedPnl"]) for x in rows)
                exit_px = float(rows[0].get("avgExitPrice") or 0)
        except Exception as e:  # noqa: BLE001
            log.warning("closed pnl: %s", e)
        if pnl is None:
            exit_px = {"TP": tr["tp"], "SL": tr["stop"], "BE": tr["stop"]}.get(outcome, self.price.get(sym) or tr["entry"])
            d = 1 if tr["side"] == "Buy" else -1
            pnl = (exit_px - tr["entry"]) * d * float(tr["qty"]) - self.p.fee_pct * 2 / 100 * tr["entry"] * float(tr["qty"])
        r_mult = pnl / tr["risk_usd"] if tr["risk_usd"] else 0
        rec = {**tr, "outcome": outcome, "pnl": pnl, "r": r_mult, "exit": exit_px, "t_close": now_ms()}
        with self.lock:
            self.journal.append(rec)
            self.trades.pop(sym, None)
        self.save()
        self.event(f"{sym}: ЗАКРЫТО {outcome} | PnL {pnl:+.2f}$ ({r_mult:+.2f}R)", "win" if pnl > 0 else "loss")

    def manage(self, tr):
        sym = tr["symbol"]
        size = self.leg(tr)
        price = self.price[sym]
        long = tr["side"] == "Buy"
        if size <= 0:
            self.finish(tr)
            return
        if str(size) != tr["qty"]:
            tr["qty"] = str(size)  # тейк исполнился частично
        if tr.get("close_reason"):
            ok = chase_close(self.s, sym, self.ins[sym], tr["idx"], self.close_side(tr), self.a.chase_sec,
                             self.a.chase_tries, self.a.chase_slip, timeout=self.a.close_timeout)
            if ok:
                self.finish(tr, tr["close_reason"])
            return
        # перенос в безубыток на +2R
        if not tr["be"] and price and ((price >= tr["be_trigger"]) if long else (price <= tr["be_trigger"])):
            if tr.get("sl_oid"):
                cancel(self.s, sym, tr["sl_oid"])
            tr["be"] = True
            self.place_stop(tr, tr["be_stop"])
            self.save()
            self.event(f"{sym}: +{self.p.be_r:g}R — стоп перенесён в безубыток {tr['stop']:g}", "trade")
        # страховка: цена за стопом, а позиция ещё открыта (стоп-лимит не исполнился или его нет)
        beyond = price and ((price < tr["stop"] * (1 - 0.0005)) if long else (price > tr["stop"] * (1 + 0.0005)))
        if beyond:
            tr["out_since"] = tr.get("out_since") or time.time()
            if time.time() - tr["out_since"] >= self.a.stop_grace:
                self.event(f"{sym}: цена за стопом, позиция открыта — закрываю лимитками с догоном", "warn")
                if tr.get("tp_oid"):
                    cancel(self.s, sym, tr["tp_oid"])
                if tr.get("sl_oid"):
                    cancel(self.s, sym, tr["sl_oid"])
                tr["close_reason"] = "BE" if tr["be"] else "SL"
        else:
            tr["out_since"] = None

    # ------------------------------------------------------------ команды со страницы
    def command(self, action, symbol=None):
        self.cmds.append((action, (symbol or "").upper()))

    def run_commands(self):
        while self.cmds:
            action, sym = self.cmds.popleft()
            if action == "start":
                self.paused = False
                self.event("Автопилот запущен")
            elif action == "pause":
                self.paused = True
                self.event("Автопилот на паузе: новые сделки не открываются, открытые остаются с тейком и стопом")
            elif action in ("close", "close_all"):
                targets = list(self.trades) if action == "close_all" else [sym]
                if action == "close_all":
                    self.paused = True
                for t in targets:
                    tr = self.trades.get(t)
                    if tr:
                        for oid in (tr.get("tp_oid"), tr.get("sl_oid")):
                            if oid:
                                cancel(self.s, t, oid)
                        tr["close_reason"] = "Вручную"
                        self.event(f"{t}: закрываю вручную лимитками с догоном")

    # ------------------------------------------------------------ главный цикл
    def recover(self):
        for sym, tr in list(self.trades.items()):
            if self.leg(tr) <= 0:
                self.finish(tr)
                continue
            info = order_info(self.s, sym, tr["sl_oid"]) if tr.get("sl_oid") else None
            if not info or info.get("orderStatus") not in ("Untriggered", "New", "PartiallyFilled"):
                self.place_stop(tr, tr["stop"])
            info = order_info(self.s, sym, tr["tp_oid"]) if tr.get("tp_oid") else None
            if not info or info.get("orderStatus") not in ("New", "PartiallyFilled"):
                self.place_tp(tr)
            self.event(f"{sym}: продолжаю вести сделку после перезапуска")
        self.save()

    def tick(self):
        self.run_commands()
        for sym in self.syms:
            try:
                self.refresh_levels(sym)
                self.price[sym] = last_price(self.s, sym)
            except Exception as e:  # noqa: BLE001
                log.warning("%s: данные: %s", sym, e)
                continue
            tr = self.trades.get(sym)
            if tr:
                try:
                    self.manage(tr)
                except Exception as e:  # noqa: BLE001
                    log.error("%s: сопровождение: %s", sym, e)
                continue
            try:
                c = self.new_closed_candle(sym)
            except Exception as e:  # noqa: BLE001
                log.warning("%s: свечи: %s", sym, e)
                continue
            if not c or self.paused or not st.in_session(c["t"], self.p):
                continue
            day = st.ny_day(c["t"]).isoformat()
            if self.taken.get((sym, day), 0) >= self.p.trades_per_day:
                continue
            lv = self.lv[sym]
            side = st.signal(c, lv["pdh"], lv["pdl"], self.p)
            if side and not self.a.dry:
                try:
                    self.enter(sym, side, c)
                except Exception as e:  # noqa: BLE001
                    self.event(f"{sym}: ошибка входа: {e}", "warn")
            elif side:
                self.event(f"{sym}: [dry] сигнал {side} @ {c['c']:g}", "signal")

    def run(self):
        for sym in self.syms:
            self.ins[sym] = instrument(self.s, sym)
        self.refresh_wallet()
        self.recover()
        self.event(f"Автопилот готов: {', '.join(self.syms)} | риск {self.p.risk_pct:g}% | окно "
                   f"{st.session_info(now_ms(), self.p)['window_kyiv']} по Киеву"
                   + (" | НА ПАУЗЕ" if self.paused else ""))
        last_w = time.time()
        while True:
            t0 = time.time()
            self.tick()
            if time.time() - last_w > 30:
                last_w = time.time()
                self.refresh_wallet()
            time.sleep(max(0.2, self.a.poll - (time.time() - t0)))

    # ------------------------------------------------------------ для страницы
    def stats(self):
        j = self.journal
        wins = [t for t in j if t["pnl"] > 0]
        losses = [t for t in j if t["pnl"] < 0]
        gp, gl = sum(t["pnl"] for t in wins), -sum(t["pnl"] for t in losses)
        return {"count": len(j), "pnl": sum(t["pnl"] for t in j),
                "winrate": len(wins) / len(j) * 100 if j else 0, "pf": gp / gl if gl else (99.9 if gp else 0)}

    def snapshot(self):
        with self.lock:
            sess = st.session_info(now_ms(), self.p)
            day = st.ny_day(now_ms()).isoformat()
            open_tr = []
            for sym, tr in self.trades.items():
                px = self.price.get(sym) or tr["entry"]
                d = 1 if tr["side"] == "Buy" else -1
                risk_px = tr["entry"] * self.p.sl_pct / 100
                r_now = (px - tr["entry"]) * d / risk_px if risk_px else 0
                open_tr.append({**tr, "price": px, "r_now": r_now, "pnl_now": r_now * tr["risk_usd"]})
            return {
                "mode": "РЕАЛЬНИЙ РАХУНОК" if self.live else "TESTNET", "dry": self.a.dry,
                "paused": self.paused, "error": self.error, "balance": self.balance, "avail": self.avail,
                "params": self.p.as_dict(), "symbols": self.syms, "session": sess, "server_ms": now_ms(),
                "control_key_required": bool(web.CONTROL_KEY),
                "markets": {s: {"price": self.price.get(s), **self.lv[s], "check": self.check[s],
                                "taken_today": self.taken.get((s, day), 0)} for s in self.syms},
                "trades": open_tr, "journal": self.journal[-100:][::-1], "stats": self.stats(),
                "events": list(self.events)[:50],
            }

    def minutes_for(self, sym, d):
        key = (sym, d.isoformat())
        if key not in self.min_cache:
            s_ms, _ = st.session_bounds(d, self.p)
            end = st.utc_day_start(s_ms) + 86_400_000 - 1
            self.min_cache[key] = klines(self.public, sym, "1", limit=1000, start=s_ms - 20 * 60_000, end=end)
        return self.min_cache[key]

    def backtest(self, sym, days, balance, risk):
        days = max(5, min(60, int(days)))
        key = (sym, days, round(balance, 2), round(risk, 3), st.ny_day(now_ms()).isoformat())
        if key in self.bt_cache:
            return self.bt_cache[key]
        p = st.Params(**{**self.p.as_dict(), "risk_pct": risk})
        daily = klines(self.public, sym, "D", limit=days + 10)
        res = st.backtest(sym, daily, lambda d: self.minutes_for(sym, d), p, days=days, balance=balance)
        self.bt_cache[key] = res
        return res


def main():
    ap = argparse.ArgumentParser(description="NY Breakout Autopilot (Bybit)")
    ap.add_argument("--testnet", action="store_true")
    ap.add_argument("--dry", action="store_true", help="только сигналы, без ордеров")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    ap.add_argument("--risk", type=float, default=1.0, help="риск на сделку, %% баланса")
    ap.add_argument("--sl", type=float, default=0.3, help="стоп, %% от входа")
    ap.add_argument("--tp-r", type=float, default=3.0)
    ap.add_argument("--be-r", type=float, default=2.0)
    ap.add_argument("--touch", type=float, default=0.12, help="касание уровня, %%")
    ap.add_argument("--strong-body", type=float, default=0.5, help="тело сигнальной свечи, доля диапазона (0 = выкл)")
    ap.add_argument("--trades-per-day", type=int, default=1)
    ap.add_argument("--session-start", default="09:30", help="начало окна по Нью-Йорку")
    ap.add_argument("--session-min", type=int, default=60)
    ap.add_argument("--fee", type=float, default=0.055, help="комиссия одной стороны, %%")
    ap.add_argument("--max-leverage", type=int, default=20)
    ap.add_argument("--entry-timeout", type=float, default=15, help="сек на вход лимитками")
    ap.add_argument("--entry-maker-tries", type=int, default=2)
    ap.add_argument("--entry-slip-r", type=float, default=0.3, help="не входить дальше X × риска от цены сигнала")
    ap.add_argument("--stop-limit-slip", type=float, default=0.15, help="стоп-лимит: цена лимитки за триггером, %%")
    ap.add_argument("--stop-grace", type=float, default=3, help="сек за стопом до догона ботом")
    ap.add_argument("--chase-sec", type=float, default=1.5)
    ap.add_argument("--chase-tries", type=int, default=5)
    ap.add_argument("--chase-slip", type=int, default=5)
    ap.add_argument("--close-timeout", type=float, default=60)
    ap.add_argument("--poll", type=float, default=1)
    ap.add_argument("--state-path", default=os.getenv("STATE_PATH") or ("/data/state.json" if os.path.isdir("/data") else "data/state.json"))
    argv = sys.argv[1:] or normalize_args(os.getenv("BOT_ARGS", ""), ap)
    a, unknown = ap.parse_known_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout, force=True)
    load_env()
    if os.getenv("BYBIT_ENV", "").lower() == "testnet":
        a.testnet = True
    live = not a.testnet
    p = st.Params(sl_pct=a.sl, tp_r=a.tp_r, be_r=a.be_r, touch_pct=a.touch, strong_body=a.strong_body,
                  risk_pct=a.risk, fee_pct=a.fee, session_start=a.session_start, session_min=a.session_min,
                  trades_per_day=a.trades_per_day)
    key, sec = os.getenv("BYBIT_API_KEY"), os.getenv("BYBIT_API_SECRET")
    s = HTTP(testnet=not live, api_key=key, api_secret=sec)
    bot = Bot(s, a, p, live)
    web.start(bot)
    if unknown:
        bot.event("BOT_ARGS: не понял и пропускаю: " + " ".join(unknown), "warn")
    log.info("Режим: %s | риск %s%% | символы %s", "РЕАЛЬНЫЙ СЧЁТ" if live else "TESTNET", a.risk, a.symbols)

    def fail(msg):
        bot.error = msg
        bot.event(msg, "warn")
        while True:
            time.sleep(3600)

    if not a.dry and not (key and sec):
        fail("Нет ключей: добавь в Railway → Variables BYBIT_API_KEY и BYBIT_API_SECRET")
    try:
        s.get_server_time()
    except Exception as e:  # noqa: BLE001
        fail(f"Bybit недоступен с этого сервера: {e}. Задай HTTPS_PROXY или смени регион Railway.")
    if not a.dry:
        try:
            wallet(s)
        except Exception as e:  # noqa: BLE001
            fail(f"Ключи не работают: {e}")

    # При перезапуске Railway позиции НЕ закрываются: на бирже стоят тейк и стоп-лимит,
    # после старта бот продолжит их вести (состояние в STATE_PATH).
    sig.signal(sig.SIGTERM, lambda *_: (bot.save(), os._exit(0)))
    while True:
        try:
            bot.run()
        except Exception as e:  # noqa: BLE001
            bot.event(f"Ошибка цикла: {e} — перезапуск через 5 сек", "warn")
            time.sleep(5)


if __name__ == "__main__":
    main()
