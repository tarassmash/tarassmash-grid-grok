"""Общие функции для работы с Bybit (pybit unified_trading)."""

import logging
import os
import shlex
import time
from decimal import ROUND_DOWN, Decimal

log = logging.getLogger("bot")
CAT = "linear"


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
            if "110043" in msg or "leverage not modified" in msg.lower():
                return None
            if i == retries - 1:
                raise
            log.warning("API ошибка (%s), повтор…", msg[:160])
            time.sleep(1.0 * (i + 1))


def normalize_args(text, parser):
    """BOT_ARGS из переменной Railway: понимает 'risk 1', '—risk 1', 'risk=1', запятые в числах."""
    known = {o.lstrip("-") for act in parser._actions for o in act.option_strings}
    text = text.replace("—", "--").replace("–", "--").replace("−", "-")
    out = []
    for tok in shlex.split(text):
        name, eq, val = tok.lstrip("-").partition("=")
        name = name.replace("_", "-").lower()
        if name in known:
            out.append("--" + name)
            if eq:
                out.append(val.replace(",", "."))
        else:
            out.append(tok.replace(",", "."))
    return out


def instrument(s, sym):
    x = api(s.get_instruments_info, category=CAT, symbol=sym)["result"]["list"][0]
    lot = x["lotSizeFilter"]
    return {
        "tick": D(x["priceFilter"]["tickSize"]),
        "qty_step": D(lot["qtyStep"]),
        "min_qty": D(lot["minOrderQty"]),
        "min_notional": D(lot.get("minNotionalValue") or "5"),
        "max_lev": float(x["leverageFilter"]["maxLeverage"]),
    }


def klines(s, sym, interval, limit=200, start=None, end=None):
    """Свечи от старых к новым: [{t,o,h,l,c,v}]."""
    kw = {"category": CAT, "symbol": sym, "interval": interval, "limit": limit}
    if start is not None:
        kw["start"] = int(start)
    if end is not None:
        kw["end"] = int(end)
    r = api(s.get_kline, **kw)["result"]["list"]
    return [{"t": int(k[0]), "o": float(k[1]), "h": float(k[2]), "l": float(k[3]), "c": float(k[4]),
             "v": float(k[5])} for k in reversed(r)]


def last_price(s, sym):
    return float(api(s.get_tickers, category=CAT, symbol=sym)["result"]["list"][0]["lastPrice"])


def best_prices(s, sym):
    ob = api(s.get_orderbook, category=CAT, symbol=sym, limit=1)["result"]
    return D(ob["b"][0][0]), D(ob["a"][0][0])


def wallet(s):
    """(баланс, доступно) в USDT."""
    r = s.get_wallet_balance(accountType="UNIFIED", coin="USDT")
    acc = r["result"]["list"][0]
    coin = next((c for c in acc["coin"] if c["coin"] == "USDT"), {})
    bal = float(coin.get("walletBalance") or acc.get("totalWalletBalance") or 0)
    avail = acc.get("totalAvailableBalance") or coin.get("availableToWithdraw") or bal
    try:
        avail = float(avail)
    except (TypeError, ValueError):
        avail = bal
    return bal, avail


def positions(s, sym):
    """Открытые ноги позиции: [(сторона, размер, positionIdx, avgPrice)]."""
    r = api(s.get_positions, category=CAT, symbol=sym)
    out = []
    for p in r["result"]["list"]:
        size = D(p.get("size") or 0)
        if size > 0:
            out.append((p["side"], size, int(p.get("positionIdx") or 0), float(p.get("avgPrice") or 0)))
    return out


def is_hedge(s, sym):
    r = api(s.get_positions, category=CAT, symbol=sym)
    return any(int(p.get("positionIdx") or 0) in (1, 2) for p in r["result"]["list"])


def order_info(s, sym, oid):
    r = api(s.get_order_history, category=CAT, symbol=sym, orderId=oid)
    lst = r["result"]["list"]
    if lst:
        return lst[0]
    r = api(s.get_open_orders, category=CAT, symbol=sym, orderId=oid)
    lst = r["result"]["list"]
    return lst[0] if lst else None


def cancel(s, sym, oid):
    try:
        s.cancel_order(category=CAT, symbol=sym, orderId=oid)
        return True
    except Exception:  # noqa: BLE001
        return False  # уже исполнен или снят


def chase_close(s, sym, ins, idx, close_side, chase_sec=1.5, tries_maker=5, slip_ticks=5, timeout=120):
    """Закрыть ногу позиции ТОЛЬКО лимитками с догоном: мейкер у лучшей цены
    (перестановка каждые chase_sec), затем быстрая лимитка через спред (IOC, цена ограничена)."""
    tick = ins["tick"]
    t_end = time.time() + timeout
    tries = 0
    while time.time() < t_end:
        leg = next((p for p in positions(s, sym) if p[2] == idx and (idx != 0 or p[0] != close_side)), None)
        if not leg:
            return True
        size = leg[1]
        bid, ask = best_prices(s, sym)
        if tries < tries_maker:
            px = (ask - tick if ask - tick > bid else ask) if close_side == "Sell" else (bid + tick if bid + tick < ask else bid)
            tif = "PostOnly"
        else:
            px = bid - tick * slip_ticks if close_side == "Sell" else ask + tick * slip_ticks
            tif = "IOC"
        tries += 1
        try:
            r = api(s.place_order, category=CAT, symbol=sym, side=close_side, orderType="Limit", qty=str(size),
                    price=str(px), timeInForce=tif, reduceOnly=True, positionIdx=idx)
        except Exception as e:  # noqa: BLE001
            log.warning("%s: лимитка закрытия не встала (%s)", sym, str(e)[:120])
            time.sleep(0.5)
            continue
        log.info("%s: закрываю %s %s лимиткой %s @ %s (попытка %d)", sym, close_side, size, tif, px, tries)
        if tif == "PostOnly":
            time.sleep(chase_sec)
            cancel(s, sym, r["result"]["orderId"])
        else:
            time.sleep(0.3)
    return not any(p[2] == idx for p in positions(s, sym))
