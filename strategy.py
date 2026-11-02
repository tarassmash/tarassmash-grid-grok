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
    be_r: float = 2.0            # перенос в безубыток на +R (0 = без переноса)
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


# Дни, когда биржа США закрыта (NYSE). В эти дни окна NY нет — бот не торгует.
US_HOLIDAYS = {date.fromisoformat(x) for x in (
    "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26", "2025-06-19",
    "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03",
    "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18", "2027-07-05",
    "2027-09-06", "2027-11-25", "2027-12-24",
)}


def trading_day(d):
    return d.weekday() < 5 and d not in US_HOLIDAYS


def prev_trading_day(d):
    d -= timedelta(days=1)
    while not trading_day(d):
        d -= timedelta(days=1)
    return d


def rth_bounds(d):
    """Основная сессия США 9:30–16:00 по Нью-Йорку, UTC-мс."""
    s = datetime.combine(d, dtime(9, 30), NY)
    e = datetime.combine(d, dtime(16, 0), NY)
    return int(s.timestamp() * 1000), int(e.timestamp() * 1000)


def rth_levels(candles, d):
    """Максимум и минимум основной сессии дня d по свечам (любого таймфрейма ≤ 30 мин)."""
    s, e = rth_bounds(d)
    xs = [c for c in candles if s <= c["t"] < e]
    if not xs:
        return None
    return max(c["h"] for c in xs), min(c["l"] for c in xs)


def in_session(ts_ms, p):
    """Минутная свеча, начавшаяся в ts_ms, целиком лежит внутри окна торгового дня США."""
    d = ny_day(ts_ms)
    if not trading_day(d):
        return False
    s, e = session_bounds(d, p)
    return s <= ts_ms and ts_ms + 60_000 <= e


def session_info(now_ms, p):
    """Для страницы: идёт ли окно, сколько до начала/конца, время Киев/Нью-Йорк."""
    d = ny_day(now_ms)
    s, e = session_bounds(d, p)
    if now_ms >= e or not trading_day(d):
        d += timedelta(days=1)
        while not trading_day(d):
            d += timedelta(days=1)
        s, e = session_bounds(d, p)
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
        "be_trigger": entry + d * p.be_r * risk if p.be_r > 0 else None,
        "be_stop": entry + d * fee_buf,
    }


def fee_r(p):
    """Комиссия за вход и выход в долях R."""
    return 2 * p.fee_pct / p.sl_pct


def simulate(side, entry, candles, p, start=0):
    """Прогон сделки по свечам candles[start:]. Если в одной свече задеты и стоп, и тейк —
    считаем, что первым был стоп (консервативно)."""
    lv = levels_for(side, entry, p)
    d = 1 if side == "Buy" else -1
    stop, be = lv["sl"], False
    trig = lv["be_trigger"]
    exit_px, outcome, exit_t = None, "OPEN", None
    for k in range(start, len(candles)):
        c = candles[k]
        lo, hi = c["l"], c["h"]
        hit_stop = lo <= stop if d > 0 else hi >= stop
        hit_tp = hi >= lv["tp"] if d > 0 else lo <= lv["tp"]
        if hit_stop:
            exit_px, outcome, exit_t = stop, ("BE" if be else "SL"), c["t"]
            break
        if hit_tp:
            exit_px, outcome, exit_t = lv["tp"], "TP", c["t"]
            break
        if trig is not None and not be and (hi >= trig if d > 0 else lo <= trig):
            be, stop = True, lv["be_stop"]
    if exit_px is None:
        last = candles[-1] if len(candles) > start else None
        exit_px = last["c"] if last else entry
        exit_t = last["t"] if last else None
    gross_r = (exit_px - entry) * d / lv["risk"]
    return {**lv, "side": side, "entry": entry, "exit": exit_px, "outcome": outcome, "exit_t": exit_t,
            "be_moved": be, "r_gross": gross_r, "r": gross_r - fee_r(p)}


# ---------------------------------------------------------------- бэктест

def day_levels(d, by_day, minutes_for_day, rth, p):
    """PDH/PDL для торгового дня d.
    Крипта: вчерашняя суточная свеча UTC. S&P 500 и др. TradFi: основная сессия прошлого торгового дня США."""
    if rth:
        pd = prev_trading_day(d)
        return rth_levels(minutes_for_day(pd), pd)
    s_ms, _ = session_bounds(d, p)
    prev = by_day.get(utc_day_start(s_ms) - 86_400_000)
    return (prev["h"], prev["l"]) if prev else None


def backtest(symbol, daily, minutes_for_day, p, days=30, balance=10000.0, now_ms=None, rth=False):
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
        if not trading_day(d):
            continue  # NY сессия только в торговые дни США
        lv = day_levels(d, by_day, minutes_for_day, rth, p)
        if not lv:
            continue
        pdh, pdl = lv
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
            res = simulate(side, c["c"], mins, p, start=i + 1)
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


# ---------------------------------------------------------------- автоподбор настроек (walk-forward)

# Какие настройки перебираем. Сигнал зависит от touch/strong/окна, выход — от sl/tp/be.
GRID = {
    "sl_pct": [0.2, 0.3, 0.5, 0.8],
    "tp_r": [1.5, 2.0, 3.0],
    "be_r": [1.0, 2.0, 0.0],
    "touch_pct": [0.08, 0.15, 0.3],
    "strong_body": [0.0, 0.5],
    "session_min": [30, 60, 90],
}
# S&P 500 (SPY) за час ходит меньше крипты — стопы и касания уже
GRID_TRADFI = {
    "sl_pct": [0.1, 0.15, 0.2, 0.3],
    "tp_r": [1.5, 2.0, 3.0],
    "be_r": [1.0, 2.0, 0.0],
    "touch_pct": [0.03, 0.06, 0.12],
    "strong_body": [0.0, 0.5],
    "session_min": [30, 60, 90],
}


def collect_days(daily, minutes_for_day, days, now_ms, session_start="09:30", progress=None, rth=False):
    """Торговые дни США за последние `days` дней: уровни прошлого дня и 1m свечи от окна до конца суток."""
    by_day = {c["t"]: c for c in daily}
    today = ny_day(now_ms)
    out = []
    ks = list(range(days, 0, -1))
    for n, k in enumerate(ks):
        d = today - timedelta(days=k)
        if progress:
            progress(n + 1, len(ks))
        if not trading_day(d):
            continue
        p0 = Params(session_start=session_start, session_min=60)
        s_ms, _ = session_bounds(d, p0)
        lv = day_levels(d, by_day, minutes_for_day, rth, p0)
        if not lv:
            continue
        mins = minutes_for_day(d)
        if mins:
            out.append({"d": d, "s_ms": s_ms, "pdh": lv[0], "pdl": lv[1], "mins": mins})
    return out


def first_signal(day, p):
    """Первый сигнал дня в окне: (индекс свечи, сторона) или None."""
    end = day["s_ms"] + p.session_min * 60_000
    for i, c in enumerate(day["mins"]):
        if c["t"] < day["s_ms"]:
            continue
        if c["t"] + 60_000 > end:
            break
        side = signal(c, day["pdh"], day["pdl"], p)
        if side:
            return i, side
    return None


def r_stats(rs):
    """Статистика по списку результатов сделок в R."""
    if not rs:
        return {"n": 0, "sum_r": 0.0, "avg_r": 0.0, "winrate": 0.0, "pf": 0.0, "max_dd_r": 0.0}
    gp = sum(r for r in rs if r > 0)
    gl = -sum(r for r in rs if r < 0)
    eq = peak = dd = 0.0
    for r in rs:
        eq += r
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    return {"n": len(rs), "sum_r": sum(rs), "avg_r": sum(rs) / len(rs),
            "winrate": sum(r > 0 for r in rs) / len(rs) * 100,
            "pf": gp / gl if gl else (99.9 if gp else 0.0), "max_dd_r": dd}


def optimize(days_data, base, test_days=20, grid=None, min_train=8, min_test=3, now_ms=None, progress=None):
    """Подбор на старых днях (train), проверка на последних test_days днях (test), которых подбор не видел.

    Настройки применяются, только если на проверке результат > 0 R и сделок хватает.
    """
    grid = grid or GRID
    now_ms = now_ms or int(datetime.now(UTC).timestamp() * 1000)
    cut = ny_day(now_ms) - timedelta(days=test_days)
    train = [x for x in days_data if x["d"] <= cut]
    test = [x for x in days_data if x["d"] > cut]
    sig_keys = [(t, sb, sm) for t in grid["touch_pct"] for sb in grid["strong_body"] for sm in grid["session_min"]]
    exit_keys = [(sl, tp, be) for sl in grid["sl_pct"] for tp in grid["tp_r"] for be in grid["be_r"]]
    total = len(sig_keys) * len(exit_keys)
    rows = []
    done = 0
    for (t, sb, sm) in sig_keys:
        ps = Params(**{**base.as_dict(), "touch_pct": t, "strong_body": sb, "session_min": sm, "trades_per_day": 1})
        sigs_train = [(x, first_signal(x, ps)) for x in train]
        sigs_test = [(x, first_signal(x, ps)) for x in test]
        for (sl, tp, be) in exit_keys:
            p = Params(**{**ps.as_dict(), "sl_pct": sl, "tp_r": tp, "be_r": be})

            def run(sigs):
                out = []
                for x, sg in sigs:
                    if sg:
                        i, side = sg
                        out.append(simulate(side, x["mins"][i]["c"], x["mins"], p, start=i + 1)["r"])
                return out
            rows.append({"params": p.as_dict(), "train": r_stats(run(sigs_train)), "test": r_stats(run(sigs_test))})
            done += 1
            if progress and done % 50 == 0:
                progress(done, total)
    ok = [r for r in rows if r["train"]["n"] >= min_train and r["train"]["pf"] >= 1.1]
    ok.sort(key=lambda r: (r["train"]["sum_r"], r["train"]["pf"]), reverse=True)
    base_row = next((r for r in rows if all(r["params"][k] == base.as_dict()[k] for k in grid)), None)
    if base_row is None:
        def run_base(xs):
            out = []
            for x in xs:
                sg = first_signal(x, base)
                if sg:
                    out.append(simulate(sg[1], x["mins"][sg[0]]["c"], x["mins"], base, start=sg[0] + 1)["r"])
            return out
        base_row = {"params": base.as_dict(), "train": r_stats(run_base(train)), "test": r_stats(run_base(test))}
    best = ok[0] if ok else None
    if not best:
        applied, reason = False, (f"На {len(train)} днях подбора нет настроек с ≥{min_train} сделками и PF ≥ 1.1 — "
                                  "монету не торгую")
    elif best["test"]["n"] < min_test:
        applied, reason = False, f"На проверке ({len(test)} дней) всего {best['test']['n']} сделок — мало, не торгую"
    elif best["test"]["sum_r"] <= 0:
        applied, reason = False, (f"Лучшие настройки подбора на проверке в минусе ({best['test']['sum_r']:+.1f}R) — "
                                  "похоже на подгонку, не торгую")
    else:
        applied, reason = True, (f"Подбор {best['train']['sum_r']:+.1f}R за {best['train']['n']} сделок, "
                                 f"проверка {best['test']['sum_r']:+.1f}R за {best['test']['n']} — применяю")
    return {
        "applied": applied, "reason": reason, "best": best, "baseline": base_row,
        "top": ok[:10], "tested": total, "passed_train": len(ok),
        "train_days": len(train), "test_days": len(test),
        "train_from": train[0]["d"].isoformat() if train else None, "test_from": test[0]["d"].isoformat() if test else None,
    }
