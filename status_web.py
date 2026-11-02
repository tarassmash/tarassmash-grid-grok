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
LOGS = collections.deque(maxlen=200)
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
    data["logs"] = list(LOGS)[-60:]
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
        return f"{v:.3g}"
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
    last = hist[-1] if hist else None
    pill = ('<span class="nnpill on">ПРИНИМАЕТ РЕШЕНИЯ</span>' if nn["active"]
            else '<span class="nnpill off">УЧИТСЯ</span>')
    if not nn["enabled"]:
        pill += ' <span class="nnpill off">решения выключены (--no-nn)</span>'
    kv = f"""<div class="kv">
      <div><span>Состояние</span><b>{escape(nn['text'])}</b></div>
      <div><span>Примеров</span><b>{nn['samples']:,}</b></div>
      <div><span>Из них реальных сеток</span><b>{nn['real']}</b></div>
      <div><span>Обучений</span><b>{len(hist)}</b></div>
      {f'<div><span>Последнее: отбор сети</span><b class="{"pos" if last["sel_avg"] > 0 else "neg"}">{last["sel_avg"]:+.3f}%</b></div>' if last else ''}
      {f'<div><span>Последнее: все сетки сканера</span><b class="{"pos" if last["base_avg"] > 0 else "neg"}">{last["base_avg"]:+.3f}%</b></div>' if last else ''}
      {f'<div><span>Связь прогноза с итогом</span><b>{last["corr"]:+.2f}</b></div>' if last else ''}
    </div>"""
    prog = f'<p class="muted small">Сейчас: {escape(nn["progress"])}</p>' if nn["progress"] else ""
    ch1 = svg_chart([("ошибка на обучении", "var(--acc)", [c["train"] for c in curve]),
                     ("ошибка на проверке", "var(--warn)", [c["val"] for c in curve])],
                    xlabels=[f"эп. {c['epoch']}" for c in curve])
    labels = [time.strftime("%d.%m %H:%M", time.gmtime(r["time"] / 1000)) for r in hist]
    ch2 = svg_chart([("сетки, отобранные сетью", "var(--pos)", [r["sel_avg"] for r in hist]),
                     ("все сетки сканера", "var(--muted)", [r["base_avg"] for r in hist])],
                    xlabels=labels, unit="%", zero=True)
    ch3 = svg_chart([("всего примеров", "var(--acc)", [r["samples"] for r in hist]),
                     ("реальных сеток ×100", "var(--warn)", [r["real"] * 100 for r in hist])], xlabels=labels)
    preds = "".join(
        f"<tr><td>{escape(x['symbol'])}</td><td>{'—' if x['pred'] is None else '%+.3f%%' % x['pred']}</td>"
        f"<td>{x['levels']}</td><td>{x['mult']:g}×</td><td class=\"{'pos' if x['take'] else 'neg'}\">{'беру' if x['take'] else 'пропуск'}</td></tr>"
        for x in nn["preds"]) or '<tr><td colspan="5" class="muted">Прогнозов ещё нет — дождись скана</td></tr>'
    return f"""
<div class="panel"><h2>Нейросеть {pill}</h2>{kv}{prog}
  <p class="muted small">Учится на учебных сетках по свежим свечам и на каждой закрытой реальной сетке (вес ×3).
  Решения принимает, только если на проверке (последние 20% данных, которых при обучении не было) сетки, которые она отбирает,
  в плюсе и лучше всех сеток сканера. Прогноз — доходность сетки в % от объёма позиции за {escape(str(nn.get('horizon', 60)))} мин.</p>
</div>
<div class="charts">
  <div class="panel"><h2>Последнее обучение: ошибка по эпохам</h2>{ch1}</div>
  <div class="panel"><h2>Качество на проверке по обучениям</h2>{ch2}</div>
  <div class="panel"><h2>Сколько данных накоплено</h2>{ch3}</div>
  <div class="panel"><h2>Прогнозы по текущим кандидатам</h2>
    <div class="scroll"><table><tr><th>Монета</th><th>Прогноз</th><th>Уровней</th><th>Шаг</th><th>Решение</th></tr>{preds}</table></div></div>
</div>"""


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
          <div><span>Объём на уровень</span><b>{escape(str(g['qty']))}</b></div>
          <div><span>Ордеров открыто</span><b>{g.get('open_orders', '—')}</b></div>
          <div><span>Исполнений</span><b>{g.get('fills', 0)}</b></div>
          <div><span>PnL сетки ≈</span><b class="{'pos' if g.get('profit', 0) >= 0 else 'neg'}">{g.get('profit', 0):.3f} $</b></div>
          <div><span>Позиция</span><b>{_fmt(g.get('net', 0))}</b></div>
          <div><span>Сдвигов за ценой</span><b>{g.get('shifts', 0)}</b></div>
          <div><span>Работает</span><b>{(time.time() - g['started']) / 60:.0f} мин</b></div>
          {f'<div><span>Закрываю лимитками</span><b class="neg">{escape(g["closing"])}</b></div>' if g.get('closing') else ''}
        </div>"""
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
.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:14px}}
.chart{{width:100%;height:auto;display:block}}.chart .grid{{stroke:var(--line);stroke-width:1}}
.chart .zero{{stroke:var(--muted);stroke-width:1;stroke-dasharray:4 3}}.chart .ax{{fill:var(--muted);font:11px ui-monospace,monospace}}
.legend{{display:flex;flex-wrap:wrap;gap:12px;margin-top:6px;font-size:12px;color:var(--muted)}}
.lg i{{display:inline-block;width:10px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}}
.nnpill{{font-size:11px;padding:2px 8px;border-radius:999px;margin-left:6px;letter-spacing:.04em}}
.nnpill.on{{background:#1d3a2f;color:var(--pos)}}.nnpill.off{{background:#3a2f17;color:var(--warn)}}.small{{font-size:12px;margin:8px 0 0}}
pre{{margin:0;max-height:420px;overflow:auto;font:12px/1.5 ui-monospace,Menlo,monospace;color:#c9d6d0;white-space:pre-wrap}}
</style></head><body><div class="wrap">
<div class="top"><h1>Vol Grid Bot</h1><span class="pill">{escape(str(s['mode']))}</span></div>
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
// автообновление каждые 5 сек; ключ управления запоминается в этом браузере
const k = document.querySelector('input[name=key]');
try {{ if (k) k.value = localStorage.getItem('ctrlKey') || ''; }} catch (e) {{}}
document.querySelector('.ctrl')?.addEventListener('submit', () => {{
  try {{ if (k) localStorage.setItem('ctrlKey', k.value); }} catch (e) {{}}
}});
setInterval(() => {{ if (document.activeElement !== k) location.reload(); }}, 5000);
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
