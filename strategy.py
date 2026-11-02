"""NY Breakout (стратегия из дашборда «NY Breakout Terminal»): чистая логика без биржи.

Правила:
  - Торгуем только в окно открытия Нью-Йорка: 9:30–10:30 по Нью-Йорку (16:30–17:30 по Киеву).
  - Уровни: PDH / PDL — максимум и минимум вчерашней дневной свечи (UTC, как на бирже).
  - LONG: закрытая 1m свеча коснулась PDH (|low − PDH| < touch %) и закрылась выше PDH,
    свеча бычья и сильная (тело ≥ strong_body × диапазона).
  - SHORT: зеркально от PDL.
  - Стоп: sl_pct % от входа. Тейк: tp_r × риск (3R). На +be_r R (2R) стоп в безубыток
    (вход ± комиссии, чтобы безубыток был действительно ≈ 0).
  - Не больше trades_per_day сделок на монету в день.

Эти же функции использует и бот, и бэктест — результаты бэктеста считаются по тем же правилам.
"""

from dataclasses import asdict, dataclass
from datetime import date, datetime, time as dtime, timedelta, timezone
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
KYIV = ZoneInfo("Europe/Kyiv")
UTC = timezone.utc


@dataclass
class Params:
    sl_pct: float = 0.3          # стоп, % от входа
    tp_r: float = 3.0            # тейк в R
    be_r: float = 2.0            # перенос в безубыток на +R
    touch_pct: float = 0.12      # касание уровня, % от уровня
    strong_body: float = 0.5     # тело сигнальной свечи ≥ доля диапазона (0 = не проверять)
    risk_pct: float = 1.0        # риск на сделку, % баланса
    fee_pct: float = 0.055       # комиссия за одну сторону, %
    session_start: str = "09:30"  # Нью-Йорк
    session_min: int = 60
    trades_per_day: int = 1

    def as_dict(self):
        return asdict(self)


# ---------------------------------------------------------------- время

def session_bounds(day_ny, p):
    """(начало, конец) окна в UTC-мс для торгового дня day_ny (дата по Нью-Йорку)."""
    hh, mm = map(int, p.session_start.split(":"))
    start = datetime.combine(day_ny, dtime(hh, mm), NY)
    end = start + timedelta(minutes=p.session_min)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def ny_day(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000, NY).date()


def in_session(ts_ms, p):
    """Минутная свеча, начавшаяся в ts_ms, целиком лежит внутри окна."""
    s, e = session_bounds(ny_day(ts_ms), p)
    return s <= ts_ms and ts_ms + 60_000 <= e


def session_info(now_ms, p):
    """Для страницы: идёт ли окно, сколько до начала/конца, время Киев/Нью-Йорк."""
    d = ny_day(now_ms)
    s, e = session_bounds(d, p)
    if now_ms >= e:
        s, e = session_bounds(d + timedelta(days=1), p)
    active = s <= now_ms < e
    now = datetime.fromtimestamp(now_ms / 1000, UTC)
    return {
        "active": active,
        "start_ms": s, "end_ms": e,
        "seconds_to_start": max(0, (s - now_ms) // 1000),
        "seconds_to_end": max(0, (e - now_ms) // 1000) if active else 0,
        "kyiv": now.astimezone(KYIV).strftime("%H:%M:%S"),
        "ny": now.astimezone(NY).strftime("%H:%M:%S"),
        "window_kyiv": "%s–%s" % (datetime.fromtimestamp(s / 1000, KYIV).strftime("%H:%M"),
                                   datetime.fromtimestamp(e / 1000, KYIV).strftime("%H:%M")),
    }


def utc_day_start(ts_ms):
    d = datetime.fromtimestamp(ts_ms / 1000, UTC).date()
    return int(datetime.combine(d, dtime(0, 0), UTC).timestamp() * 1000)


# ---------------------------------------------------------------- сигнал и сделка

def checklist(c, pdh, pdl, p):
    """Пункты чеклиста по закрытой свече c = {o,h,l,c}. Для страницы и для сигнала."""
    rng = c["h"] - c["l"]
    body = abs(c["c"] - c["o"])
    strong = p.strong_body <= 0 or (rng > 0 and body >= p.strong_body * rng)
    long_touch = pdh > 0 and abs(c["l"] - pdh) < pdh * p.touch_pct / 100
    short_touch = pdl > 0 and abs(c["h"] - pdl) < pdl * p.touch_pct / 100
    return {
        "long": {"break": c["c"] > pdh, "retest": long_touch, "strong": strong and c["c"] > c["o"]},
        "short": {"break": c["c"] < pdl, "retest": short_touch, "strong": strong and c["c"] < c["o"]},
    }


def signal(c, pdh, pdl, p):
    """'Buy' / 'Sell' / None по закрытой 1m свече."""
    ch = checklist(c, pdh, pdl, p)
    if all(ch["long"].values()):
        return "Buy"
    if all(ch["short"].values()):
        return "Sell"
    return None


def levels_for(side, entry, p):
    """Стоп, тейк, цена переноса в безубыток и сам безубыток от фактического входа."""
    risk = entry * p.sl_pct / 100
    d = 1 if side == "Buy" else -1
    fee_buf = entry * p.fee_pct * 2 / 100
    return {
        "risk": risk,
        "sl": entry - d * risk,
        "tp": entry + d * p.tp_r * risk,
        "be_trigger": entry + d * p.be_r * risk,
        "be_stop": entry + d * fee_buf,
    }


def fee_r(p):
    """Комиссия за вход и выход в долях R."""
    return 2 * p.fee_pct / p.sl_pct


def simulate(side, entry, candles, p):
    """Прогон сделки по свечам после входа. Если в одной свече задеты и стоп, и тейк —
    считаем, что первым был стоп (консервативно)."""
    lv = levels_for(side, entry, p)
    d = 1 if side == "Buy" else -1
    stop, be = lv["sl"], False
    exit_px, outcome, exit_t = None, "OPEN", None
    for c in candles:
        lo, hi = c["l"], c["h"]
        hit_stop = lo <= stop if d > 0 else hi >= stop
        hit_tp = hi >= lv["tp"] if d > 0 else lo <= lv["tp"]
        if hit_stop:
            exit_px, outcome, exit_t = stop, ("BE" if be else "SL"), c["t"]
            break
        if hit_tp:
            exit_px, outcome, exit_t = lv["tp"], "TP", c["t"]
            break
        if not be and (hi >= lv["be_trigger"] if d > 0 else lo <= lv["be_trigger"]):
            be, stop = True, lv["be_stop"]
    if exit_px is None:
        exit_px = candles[-1]["c"] if candles else entry
        exit_t = candles[-1]["t"] if candles else None
    gross_r = (exit_px - entry) * d / lv["risk"]
    return {**lv, "side": side, "entry": entry, "exit": exit_px, "outcome": outcome, "exit_t": exit_t,
            "be_moved": be, "r_gross": gross_r, "r": gross_r - fee_r(p)}


# ---------------------------------------------------------------- бэктест

def backtest(symbol, daily, minutes_for_day, p, days=30, balance=10000.0, now_ms=None):
    """Честный бэктест по свечам.

    daily: дневные свечи [{t,o,h,l,c}] (UTC), от старых к новым.
    minutes_for_day(day_ny) -> 1m свечи от начала окна до конца UTC-суток.
    """
    now_ms = now_ms or int(datetime.now(UTC).timestamp() * 1000)
    by_day = {c["t"]: c for c in daily}
    today = ny_day(now_ms)
    trades, equity = [], [{"label": "Старт", "balance": balance}]
    start_balance = peak = balance
    max_dd = 0.0
    for k in range(days, 0, -1):
        d = today - timedelta(days=k)
        if d.weekday() >= 5:
            continue  # NY сессия только в будни
        s_ms, e_ms = session_bounds(d, p)
        prev = by_day.get(utc_day_start(s_ms) - 86_400_000)
        if not prev:
            continue
        pdh, pdl = prev["h"], prev["l"]
        mins = minutes_for_day(d)
        taken = 0
        i = 0
        while i < len(mins) and taken < p.trades_per_day:
            c = mins[i]
            if not in_session(c["t"], p):
                i += 1
                continue
            side = signal(c, pdh, pdl, p)
            if not side:
                i += 1
                continue
            res = simulate(side, c["c"], mins[i + 1:], p)
            risk_usd = balance * p.risk_pct / 100
            pnl = res["r"] * risk_usd
            balance += pnl
            peak = max(peak, balance)
            max_dd = max(max_dd, (peak - balance) / peak * 100 if peak else 0)
            j_end = next((j for j in range(i, len(mins)) if mins[j]["t"] == res["exit_t"]), len(mins) - 1)
            window = mins[max(0, i - 15): min(len(mins), j_end + 6)]
            trades.append({
                "symbol": symbol, "date": d.isoformat(), "t": c["t"], "pdh": pdh, "pdl": pdl,
                "pnl": pnl, "balance": balance, "candles": window, **res,
            })
            equity.append({"label": d.strftime("%d.%m"), "balance": balance})
            taken += 1
            i = j_end + 1
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    gp, gl = sum(t["pnl"] for t in wins), -sum(t["pnl"] for t in losses)
    streak_w = streak_l = cw = cl = 0
    for t in trades:
        if t["pnl"] > 0:
            cw, cl = cw + 1, 0
        elif t["pnl"] < 0:
            cl, cw = cl + 1, 0
        streak_w, streak_l = max(streak_w, cw), max(streak_l, cl)
    return {
        "symbol": symbol, "days": days, "params": p.as_dict(), "trades": trades, "equity": equity,
        "stats": {
            "count": len(trades), "tp": sum(t["outcome"] == "TP" for t in trades),
            "be": sum(t["outcome"] == "BE" for t in trades), "sl": sum(t["outcome"] == "SL" for t in trades),
            "open": sum(t["outcome"] == "OPEN" for t in trades),
            "winrate": len(wins) / len(trades) * 100 if trades else 0,
            "pf": gp / gl if gl else (99.9 if gp else 0),
            "net": balance - start_balance, "net_pct": (balance / start_balance - 1) * 100,
            "max_dd": max_dd, "max_wins": streak_w, "max_losses": streak_l,
            "fee_r": fee_r(p), "avg_r": sum(t["r"] for t in trades) / len(trades) if trades else 0,
        },
    }
