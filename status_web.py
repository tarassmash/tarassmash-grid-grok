"""Страница состояния бота: Railway даёт сервису ссылку, по ней видно, что бот делает.

/            — страница (обновляется сама каждые 5 сек)
/api/status  — то же в JSON
"""

import collections
import hmac
import json
import logging
import os
import threading
import time
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

STATUS = {
    "started": time.time(),
    "mode": "запуск…",
    "budget": None,
    "leverage": None,
    "balance": None,
    "state": "запуск",
    "error": "",
    "scan": {"time": None, "checked": 0, "passed": 0, "top": []},
    "grid": None,
    "history": [],
}
LOGS = collections.deque(maxlen=400)
LOCK = threading.Lock()

# Управление со страницы. paused=True: бот закрывает текущую сетку и не открывает новые.
CONTROL = {"paused": os.getenv("START_PAUSED", "") == "1", "msg": ""}
CONTROL_KEY = os.getenv("CONTROL_KEY", "")
BRAIN = None   # нейросеть бота (nn.Brain), задаёт bybit_grid_bot.main()


def update(**kw):
    with LOCK:
        STATUS.update(kw)


class RingHandler(logging.Handler):
    def emit(self, record):
        try:
            LOGS.append(time.strftime("%H:%M:%S", time.gmtime(record.created)) + " UTC  " + record.getMessage())
        except Exception:  # noqa: BLE001
            pass


def snapshot():
    with LOCK:
        data = json.loads(json.dumps(STATUS, default=str))
    data["paused"] = CONTROL["paused"]
    data["control_msg"] = CONTROL["msg"]
    data["uptime_min"] = round((time.time() - STATUS["started"]) / 60, 1)
    data["logs"] = list(LOGS)[-200:]
    data["nn"] = BRAIN.status() if BRAIN else None
    return data


def svg_chart(series, w=560, h=190, xlabels=None, unit="", zero=False):
    """Простой линейный график в SVG. series: [(название, цвет, [значения])]."""
    vals = [v for _, _, ys in series for v in ys if v is not None]
    if not vals:
        return '<p class="muted">Данных для графика пока нет</p>'
    lo, hi = min(vals), max(vals)
    if zero:
        lo, hi = min(lo, 0), max(hi, 0)
    if hi - lo < 1e-12:
        hi, lo = hi + 1, lo - 1
    pad_l, pad_r, pad_t, pad_b = 58, 14, 10, 22

    def num(v):
        a = abs(v)
        if a >= 1e6:
            return f"{v / 1e6:.1f}M"
        if a >= 1e3:
            return f"{v / 1e3:.1f}k"
        if a >= 100:
            return f"{v:.0f}"
        t = f"{v:.3f}".rstrip("0").rstrip(".")
        return "0" if t in ("-0", "") else t
    n = max(len(ys) for _, _, ys in series)
    X = lambda i: pad_l + (w - pad_l - pad_r) * (i / max(n - 1, 1))  # noqa: E731
    Y = lambda v: pad_t + (h - pad_t - pad_b) * (1 - (v - lo) / (hi - lo))  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        y = Y(v)
        out.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                   f'<text x="{pad_l - 6}" y="{y + 4:.1f}" text-anchor="end" class="ax">{num(v)}{unit}</text>')
    if zero and lo < 0 < hi:
        out.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" class="zero"/>')
    if xlabels:
        for i in sorted({0, n // 2, n - 1}):
            if 0 <= i < len(xlabels):
                anchor = "start" if i == 0 else ("end" if i == n - 1 else "middle")
                out.append(f'<text x="{X(i):.1f}" y="{h - 5}" text-anchor="{anchor}" class="ax">{escape(str(xlabels[i]))}</text>')
    for name, color, ys in series:
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(ys) if v is not None)
        out.append(f'<polyline points="{pts}" fill="none" style="stroke:{color}" stroke-width="2" stroke-linejoin="round"/>')
        if ys and ys[-1] is not None:
            out.append(f'<circle cx="{X(len(ys) - 1):.1f}" cy="{Y(ys[-1]):.1f}" r="3.5" style="fill:{color}"/>')
    out.append("</svg>")
    legend = "".join(f'<span class="lg"><i style="background:{c}"></i>{escape(nm)}</span>' for nm, c, _ in series)
    return "".join(out) + f'<div class="legend">{legend}</div>'


def render_nn(nn):
    if not nn:
        return ""
    hist = nn["history"]
    curve = nn["curve"]
    pill = ('<span class="nnpill on">ПРИНИМАЕТ РЕШЕНИЯ</span>' if nn["active"]
            else '<span class="nnpill off">УЧИТСЯ</span>')
    if not nn["enabled"]:
        pill += ' <span class="nnpill off">решения выключены (--no-nn)</span>'
    om = nn.get("online") or {}
    rg = nn.get("regime")
    if nn.get("regime_hold"):
        regime_txt = (f"сменился в {time.strftime('%H:%M', time.gmtime(nn['regime_start'] / 1000))} UTC — "
                      f"итогов {nn.get('regime_n', 0)}/{nn.get('regime_need', 20)}")
    elif rg:
        regime_txt = f"стабильный (волат. ×{rg['vol_ratio']:.2f}, тренд {rg['er_now']:.2f})"
    else:
        regime_txt = "считаю…"
    em = nn.get("eval_min", 60)
    def pct(v, n):
        return f'<b class="{"pos" if v > 0 else "neg"}">{v:+.3f}%</b>' if n else '<b class="muted">ещё нет итогов</b>'
    kv = f"""<div class="kv">
      <div><span>Состояние</span><b>{escape(nn['text'])}</b></div>
      <div><span>Примеров за {nn.get('window_h', 4):g} ч</span><b>{nn['samples']:,}</b></div>
      <div><span>Реальных сеток за сутки</span><b>{nn['real']}</b></div>
      <div><span>Монет в учёбе</span><b>{nn.get('coins', 0)}</b></div>
      <div><span>Ждут итога</span><b>{nn.get('pending', 0):,}</b></div>
      <div><span>Онлайн {em:g} мин: выбор сети</span>{pct(om.get('sel_avg', 0), om.get('sel_n', 0))}</div>
      <div><span>Онлайн {em:g} мин: обычные</span>{pct(om.get('base_avg', 0), om.get('n', 0))}</div>
      <div><span>Сделок сети в проверке</span><b>{om.get('sel_n', 0)} из {om.get('n', 0)}</b></div>
      {f'<div><span>Ошибка на новых сетках</span><b>{nn["mae"]:.3f}%</b></div>' if nn.get("mae") is not None else ''}
      <div><span>Режим рынка</span><b class="{'neg' if nn.get('regime_hold') else ''}">{regime_txt}</b></div>
    </div>"""
    prog = f'<p class="muted small">Сейчас: {escape(nn["progress"])}</p>' if nn["progress"] else ""
    ch1 = svg_chart([("ошибка на обучении", "var(--acc)", [c["train"] for c in curve]),
                     ("ошибка на проверке", "var(--warn)", [c["val"] for c in curve])],
                    xlabels=[f"эп. {c['epoch']}" for c in curve])
    labels = [time.strftime("%H:%M", time.gmtime(r["time"] / 1000)) for r in hist]
    ch2 = svg_chart([("выбор сети по факту", "var(--pos)", [r["sel_avg"] for r in hist]),
                     ("обычные настройки", "var(--muted)", [r["base_avg"] for r in hist])],
                    xlabels=labels, unit="%", zero=True)
    ch3 = svg_chart([("ошибка прогноза на ещё не виденных сетках", "var(--warn)", [r.get("mae") for r in hist])],
                    xlabels=labels, unit="%")
    ch4 = svg_chart([("примеров в памяти", "var(--acc)", [r["samples"] for r in hist])], xlabels=labels)
    ch5 = svg_chart([("волатильность рынка (медиана по монетам)", "var(--acc)", [r.get("vol_ratio") for r in hist]),
                     ("порог смены рынка", "var(--neg)", [nn.get("regime_vol", 1.7) if r.get("vol_ratio") is not None else None
                                                          for r in hist])],
                    xlabels=labels, unit="×")
    preds = "".join(
        f"<tr><td>{escape(x['symbol'])}</td><td>{'—' if x['pred'] is None else '%+.3f%%' % x['pred']}</td>"
        f"<td>{'%+.3f%%' % x.get('pred_default', 0)}</td><td class=\"cfgcell\">{escape(x.get('cfg', ''))}</td>"
        f"<td>×{x.get('size', 1):.2f}</td><td class=\"{'pos' if x['take'] else 'neg'}\">{'беру' if x['take'] else 'пропуск'}</td></tr>"
        for x in nn["preds"]) or '<tr><td colspan="6" class="muted">Прогнозов ещё нет — дождись скана</td></tr>'
    kinds = {"data": ("ДАННЫЕ", "k-data"), "train": ("ОБУЧЕНИЕ", "k-train"), "decide": ("РЕШЕНИЕ", "k-decide"),
             "result": ("ИТОГ СЕТКИ", "k-result"), "plan": ("ПЛАН", "k-plan"), "warn": ("ВНИМАНИЕ", "k-warn")}
    th = "".join(
        f'<div class="th"><div class="th-h"><span class="tag {kinds.get(x["kind"], ("", ""))[1]}">'
        f'{kinds.get(x["kind"], (x["kind"], ""))[0]}</span><b>{escape(x["title"])}</b>'
        f'<span class="th-t">{time.strftime("%d.%m %H:%M:%S", time.gmtime(x["t"] / 1000))} UTC</span></div>'
        + ("<ul>" + "".join(f"<li>{escape(l)}</li>" for l in x["lines"]) + "</ul>" if x["lines"] else "")
        + "</div>" for x in nn.get("thoughts", [])) or '<p class="muted">Мыслей пока нет — нейросеть запускается</p>'
    plan = "".join(f"<li>{escape(l)}</li>" for l in nn.get("plan", []))
    imp = nn.get("importance") or []
    mx = max([i["value"] for i in imp] + [1e-9])
    imp_html = "".join(
        f'<div class="imp"><span>{escape(i["name"])}</span><div class="bar"><i style="width:{max(0, i["value"]) / mx * 100:.0f}%"></i></div></div>'
        for i in imp) or '<p class="muted">Появится после первого обучения</p>'
    return f"""
<div class="panel"><h2>Нейросеть {pill}</h2>{kv}{prog}
  <p class="muted small">Учится на ходу: в памяти только последние {nn.get('window_h', 4):g} ч рынка. Каждую минуту — новые моменты
  рынка, в каждом сеть сразу фиксирует свой выбор настроек; когда сетка отработает (20–{escape(str(nn.get('horizon', 90)))} мин),
  итог идёт в обучение и в онлайн-проверку. Решения принимает, только если за последние {em:g} мин её выбор по факту в плюсе
  и лучше обычных настроек. Реальные сетки на бирже учат её с весом ×3.</p>
</div>
<div class="charts">
  <div class="panel"><h2>Онлайн-проверка за {em:g} мин: выбор сети vs обычные</h2>{ch2}</div>
  <div class="panel"><h2>Ошибка прогноза на новых сетках</h2>{ch3}</div>
  <div class="panel"><h2>Полное переобучение: ошибка по эпохам</h2>{ch1}</div>
  <div class="panel"><h2>Память нейросети</h2>{ch4}</div>
  <div class="panel"><h2>Режим рынка: волатильность 15 мин к прошлому часу</h2>{ch5}</div>
  <div class="panel"><h2>План нейросети</h2><ul class="plan">{plan}</ul>
    <h2 style="margin-top:14px">На что она опирается</h2>{imp_html}</div>
  <div class="panel" style="grid-column:1/-1"><h2>Прогнозы по текущим кандидатам</h2>
    <div class="scroll"><table><tr><th>Монета</th><th>С настройками сети</th><th>С обычными</th><th>Настройки сети</th><th>Маржа</th><th>Решение</th></tr>{preds}</table></div></div>
</div>
<div class="panel"><h2>Мысли нейросети (новое сверху)</h2><div class="thoughts">{th}</div></div>"""


def _fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}g}" if abs(v) < 1 else f"{v:,.{min(nd, 4)}f}".rstrip("0").rstrip(".")
    return escape(str(v))


def render():
    s = snapshot()
    g = s.get("grid")
    if g:
        grid_html = f"""
        <div class="kv">
          <div><span>Монета</span><b>{escape(g['symbol'])}</b></div>
          <div><span>Диапазон</span><b>{_fmt(g['lower'])} … {_fmt(g['upper'])}</b></div>
          <div><span>Уровней</span><b>{g['levels']}</b></div>
          <div><span>Шаг</span><b>{g['step_pct']:.3f}%</b></div>
          <div><span>Маржа</span><b>{_fmt(g.get('budget'))} $</b></div>
          <div><span>Средний лот</span><b>{escape(str(g['qty']))}</b></div>
          <div><span>Ордеров открыто</span><b>{g.get('open_orders', '—')}</b></div>
          <div><span>Исполнений</span><b>{g.get('fills', 0)}</b></div>
          <div><span>PnL сетки ≈</span><b class="{'pos' if g.get('profit', 0) >= 0 else 'neg'}">{g.get('profit', 0):.3f} $</b></div>
          <div><span>Позиция</span><b>{_fmt(g.get('net', 0))}</b></div>
          <div><span>Сдвигов за ценой</span><b>{g.get('shifts', 0)}</b></div>
          <div><span>Работает</span><b>{(time.time() - g['started']) / 60:.0f} мин</b></div>
          {f'<div><span>Закрываю лимитками</span><b class="neg">{escape(g["closing"])}</b></div>' if g.get('closing') else ''}
        </div>"""
        if g.get("cfg"):
            grid_html += f'<p class="muted small">Настройки от нейросети: {escape(g["cfg"])}</p>'
    else:
        grid_html = '<p class="muted">Сейчас сетки нет — бот ищет монету.</p>'

    rows = "".join(
        f"<tr><td>{escape(c['symbol'])}</td><td>{c['vol1m']:.2f}%</td><td>{c['atr']:.2f}%</td>"
        f"<td>{c['er']:.2f}</td><td>{c['score']:.2f}</td>"
        f"<td>{'—' if c.get('pred') is None else '%+.3f%%' % c['pred']}</td></tr>"
        for c in s["scan"]["top"]
    ) or '<tr><td colspan="6" class="muted">Скан ещё не прошёл</td></tr>'

    hist = "".join(
        f"<tr><td>{escape(h['time'])}</td><td>{escape(h['symbol'])}</td><td>{h['fills']}</td>"
        f"<td class=\"{'pos' if h['profit'] >= 0 else 'neg'}\">{h['profit']:.3f} $</td><td>{escape(h['reason'])}</td></tr>"
        for h in reversed(s["history"][-20:])
    ) or '<tr><td colspan="5" class="muted">Закрытых сеток пока нет</td></tr>'

    err = f'<div class="err">{escape(s["error"])}</div>' if s["error"] else ""
    logs = escape("\n".join(reversed(s["logs"])))
    scan_t = time.strftime("%H:%M:%S UTC", time.gmtime(s["scan"]["time"])) if s["scan"]["time"] else "—"
    live = "РЕАЛЬНЫЙ" in str(s["mode"])
    paused = s["paused"]
    key_input = ('<input name="key" type="password" placeholder="ключ управления" autocomplete="current-password">'
                 if CONTROL_KEY else "")
    msg = f'<span class="cmsg">{escape(s["control_msg"])}</span>' if s["control_msg"] else ""
    controls = f"""
<form class="ctrl" method="post">{key_input}
  <button formaction="/api/start" class="start" {'disabled' if not paused else ''}>Старт</button>
  <button formaction="/api/stop" class="stop" {'disabled' if paused else ''}>Стоп</button>
  <span class="cstate {'off' if paused else 'on'}">{'На паузе' if paused else 'Торгует'}</span>{msg}
</form>
<p class="muted small">«Стоп» снимает ордера, закрывает позицию лимитками с догоном и ставит бота на паузу. «Старт» снова ищет монету и ставит сетку.</p>"""

    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vol Grid Bot</title>
<style>
:root{{--bg:#0e1412;--panel:#151d1a;--fg:#e2eae6;--muted:#8a9a93;--line:#24302b;--acc:#34c79c;--pos:#3fcf8e;--neg:#ef6b62;--warn:#e5a83b}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;padding:16px}}
.wrap{{max-width:1000px;margin:0 auto;display:grid;gap:14px}}
h1{{margin:0;font-size:20px}}h2{{margin:0 0 10px;font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}}
.top{{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;justify-content:space-between}}
.pill{{padding:3px 10px;border-radius:999px;font-size:12px;font-weight:700;background:{'#4a1d1a' if live else '#1d3a2f'};color:{'var(--neg)' if live else 'var(--acc)'}}}
.panel{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;min-width:0}}
.kv{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}}
.kv div{{display:grid;gap:2px}}.kv span{{font-size:12px;color:var(--muted)}}.kv b{{font:600 15px ui-monospace,Menlo,monospace}}
table{{width:100%;border-collapse:collapse;font:13px ui-monospace,Menlo,monospace}}
td,th{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
td:first-child,th:first-child{{text-align:left}}th{{font:600 11px system-ui;color:var(--muted);text-transform:uppercase}}
.scroll{{overflow-x:auto}}.muted{{color:var(--muted)}}.pos{{color:var(--pos)}}.neg{{color:var(--neg)}}
.err{{background:#3a1715;color:#ffb4ad;border-radius:8px;padding:10px 12px;font-family:ui-monospace,monospace;white-space:pre-wrap}}
.ctrl{{display:flex;flex-wrap:wrap;gap:10px;align-items:center}}
.ctrl input{{font:14px ui-monospace,monospace;padding:8px 10px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--fg);min-width:0;width:180px}}
.ctrl button{{font:700 14px system-ui;padding:9px 22px;border-radius:6px;border:0;cursor:pointer}}
.ctrl button:disabled{{opacity:.35;cursor:default}}
.start{{background:var(--acc);color:#06231a}}.stop{{background:var(--neg);color:#2a0705}}
.cstate{{font-weight:700}}.cstate.on{{color:var(--pos)}}.cstate.off{{color:var(--warn)}}
.cmsg{{color:var(--muted);font-size:13px}}
.topr{{display:flex;align-items:center;gap:8px}}
.live{{font:600 12px system-ui;padding:4px 10px;border-radius:999px;border:1px solid var(--line);background:transparent;
  color:var(--pos);cursor:pointer}}.live.off{{color:var(--warn)}}
.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:14px}}
.chart{{width:100%;height:auto;display:block}}.chart .grid{{stroke:var(--line);stroke-width:1}}
.chart .zero{{stroke:var(--muted);stroke-width:1;stroke-dasharray:4 3}}.chart .ax{{fill:var(--muted);font:11px ui-monospace,monospace}}
.legend{{display:flex;flex-wrap:wrap;gap:12px;margin-top:6px;font-size:12px;color:var(--muted)}}
.lg i{{display:inline-block;width:10px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}}
.nnpill{{font-size:11px;padding:2px 8px;border-radius:999px;margin-left:6px;letter-spacing:.04em}}
.nnpill.on{{background:#1d3a2f;color:var(--pos)}}.nnpill.off{{background:#3a2f17;color:var(--warn)}}
.thoughts{{max-height:560px;overflow-y:auto;display:grid;gap:10px}}
.th{{border-left:3px solid var(--line);padding:4px 0 4px 10px}}
.th-h{{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 10px}}.th-h b{{font-weight:600}}
.th-t{{color:var(--muted);font:11px ui-monospace,monospace;margin-left:auto}}
.th ul,.plan{{margin:6px 0 0;padding-left:18px;color:#c9d6d0;font-size:13px;line-height:1.5}}
.tag{{font:700 10px system-ui;letter-spacing:.06em;padding:2px 6px;border-radius:4px;background:#1f2a26;color:var(--muted)}}
.k-data{{background:#17283a;color:#7cb7f0}}.k-train{{background:#2a2140;color:#b9a2f5}}.k-decide{{background:#1d3a2f;color:var(--pos)}}
.k-result{{background:#3a3417;color:#f0d36a}}.k-plan{{background:#1f2a26;color:#a8c2b8}}.k-warn{{background:#3a2017;color:var(--warn)}}
.imp{{display:grid;grid-template-columns:minmax(0,190px) 1fr;gap:10px;align-items:center;font-size:12px;margin:5px 0;color:#c9d6d0}}
.cfgcell{{font-size:12px;white-space:normal;min-width:220px;text-align:left}}
.bar{{height:6px;background:var(--line);border-radius:3px;overflow:hidden}}.bar i{{display:block;height:100%;background:var(--acc)}}.small{{font-size:12px;margin:8px 0 0}}
pre{{margin:0;max-height:420px;overflow:auto;font:12px/1.5 ui-monospace,Menlo,monospace;color:#c9d6d0;white-space:pre-wrap}}
</style></head><body><div class="wrap">
<div class="top"><h1>Vol Grid Bot</h1><div class="topr"><button type="button" class="live" id="live-btn">Обновление: вкл</button><span class="pill">{escape(str(s['mode']))}</span></div></div>
{err}
<div class="panel">{controls}</div>
<div class="panel"><div class="kv">
  <div><span>Состояние</span><b>{escape(str(s['state']))}</b></div>
  <div><span>Баланс USDT</span><b>{_fmt(s['balance'])}</b></div>
  <div><span>Маржа / плечо</span><b>{_fmt(s['budget'])} $ × {_fmt(s['leverage'])}</b></div>
  <div><span>Работает</span><b>{s['uptime_min']} мин</b></div>
</div></div>
<div class="panel"><h2>Текущая сетка</h2>{grid_html}</div>
<div class="panel"><h2>Последний скан · {scan_t} · проверено {s['scan']['checked']}, подошло {s['scan']['passed']}</h2>
<div class="scroll"><table><tr><th>Монета</th><th>1m размах</th><th>ATR</th><th>ER</th><th>Score</th><th>Прогноз сети</th></tr>{rows}</table></div></div>
{render_nn(s.get("nn"))}
<div class="panel"><h2>Закрытые сетки</h2>
<div class="scroll"><table><tr><th>Время</th><th>Монета</th><th>Исполн.</th><th>Прибыль ≈</th><th>Причина</th></tr>{hist}</table></div></div>
<div class="panel"><h2>Лог (новое сверху)</h2><pre>{logs}</pre></div>
<p class="muted">Страница обновляется каждые 5 сек. Прибыль — оценка бота; точные цифры в истории Bybit.</p>
</div>
<script>
// Обновление без перезагрузки: раз в 5 сек подменяем содержимое на месте и оставляем на экране то,
// что сейчас читаешь (прокрутка страницы, лога и мыслей не прыгает). Пока выделен текст — не обновляем.
let paused = false;
const keyIn = () => document.querySelector('input[name=key]');
try {{ const k = keyIn(); if (k) k.value = localStorage.getItem('ctrlKey') || ''; }} catch (e) {{}}
document.addEventListener('submit', e => {{
  const k = keyIn();
  try {{ if (k) localStorage.setItem('ctrlKey', k.value); }} catch (er) {{}}
}});
function paintBtn() {{
  const b = document.getElementById('live-btn');
  if (b) {{ b.textContent = paused ? 'Обновление: пауза' : 'Обновление: вкл'; b.classList.toggle('off', paused); }}
}}
document.addEventListener('click', e => {{
  if (e.target && e.target.id === 'live-btn') {{ paused = !paused; paintBtn(); if (!paused) refresh(); }}
}});
const SCROLLS = '.thoughts, pre, .scroll';
// что сейчас вверху видимой области: строка лога или запись мыслей (+ сдвиг внутри неё)
function anchorOf(el) {{
  if (el.tagName === 'PRE') {{
    const lines = el.textContent.split('\\n'), lh = el.scrollHeight / Math.max(lines.length, 1);
    const i = Math.floor(el.scrollTop / lh);
    return {{ pre: true, text: lines[i], off: el.scrollTop - i * lh }};
  }}
  const r = el.getBoundingClientRect();
  const k = [...el.children].find(c => c.getBoundingClientRect().bottom > r.top + 1);
  return k ? {{ text: k.textContent.slice(0, 300), off: k.getBoundingClientRect().top - r.top }} : null;
}}
// новое добавляется сверху: возвращаем на экран ту же строку / ту же запись, что читал
function restore(el, o) {{
  el.scrollLeft = o.left;
  if (!(o.top > 0 || o.hover) || !o.a) return;
  if (o.a.pre) {{
    const lines = el.textContent.split('\\n'), lh = el.scrollHeight / Math.max(lines.length, 1);
    const i = lines.indexOf(o.a.text);
    el.scrollTop = i >= 0 ? i * lh + o.a.off : o.top;
    return;
  }}
  el.scrollTop = o.top;
  const k = [...el.children].find(c => c.textContent.slice(0, 300) === o.a.text);
  if (k) el.scrollTop += k.getBoundingClientRect().top - el.getBoundingClientRect().top - o.a.off;
}}
async function refresh() {{
  if (paused || document.hidden) return;
  const sel = window.getSelection && window.getSelection();
  if (sel && sel.toString()) return;
  const k = keyIn();
  if (k && document.activeElement === k) return;
  let html;
  try {{ html = await (await fetch(location.pathname, {{ cache: 'no-store' }})).text(); }} catch (e) {{ return; }}
  const nxt = new DOMParser().parseFromString(html, 'text/html').querySelector('.wrap');
  const cur = document.querySelector('.wrap');
  if (!nxt || !cur || paused) return;
  const old = [...cur.querySelectorAll(SCROLLS)].map(el => ({{ top: el.scrollTop, left: el.scrollLeft,
    hover: el.matches(':hover'), a: anchorOf(el) }}));
  const pkey = p => (p.querySelector('h2, h1') || {{}}).textContent || p.className;
  const panels = [...cur.children];
  const ap = panels.find(p => p.getBoundingClientRect().bottom > 0);
  const aKey = ap ? pkey(ap) : null, aTop = ap ? ap.getBoundingClientRect().top : 0, aIdx = panels.indexOf(ap);
  const keyVal = k ? k.value : '';
  cur.innerHTML = nxt.innerHTML;
  const k2 = keyIn();
  if (k2) k2.value = keyVal;
  [...cur.querySelectorAll(SCROLLS)].forEach((el, i) => {{ if (old[i]) restore(el, old[i]); }});
  if (ap) {{
    const np = [...cur.children].find(p => pkey(p) === aKey) || cur.children[aIdx];
    if (np) window.scrollBy(0, np.getBoundingClientRect().top - aTop);
  }}
  paintBtn();
}}
setInterval(refresh, 5000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path.startswith("/api/status"):
            body, ctype = json.dumps(snapshot(), ensure_ascii=False, default=str).encode(), "application/json"
        elif self.path in ("/", "/index.html"):
            body, ctype = render().encode(), "text/html; charset=utf-8"
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        form = parse_qs(self.rfile.read(min(n, 4096)).decode("utf-8", "ignore"))
        key = (form.get("key") or [""])[0]
        if CONTROL_KEY and not hmac.compare_digest(key, CONTROL_KEY):
            CONTROL["msg"] = "Неверный ключ управления"
        elif self.path.startswith("/api/stop"):
            if not CONTROL["paused"]:
                CONTROL["paused"] = True
                CONTROL["msg"] = "Остановка: снимаю ордера и закрываю позицию…"
                logging.getLogger("grid").info("СТОП со страницы")
        elif self.path.startswith("/api/start"):
            if CONTROL["paused"]:
                CONTROL["paused"] = False
                CONTROL["msg"] = "Запущен"
                logging.getLogger("grid").info("СТАРТ со страницы")
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()

    def log_message(self, *_):
        pass


def start():
    """Запустить страницу в фоне. Railway передаёт порт в переменной PORT."""
    logging.getLogger("grid").addHandler(RingHandler())
    port = int(os.getenv("PORT") or 8080)
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    logging.getLogger("grid").info("Страница состояния на порту %d", port)
