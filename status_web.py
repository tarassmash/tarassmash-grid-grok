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
    return data


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
          <div><span>Прибыль сетки ≈</span><b class="{'pos' if g.get('profit', 0) >= 0 else 'neg'}">{g.get('profit', 0):.3f} $</b></div>
          <div><span>Работает</span><b>{(time.time() - g['started']) / 60:.0f} мин</b></div>
        </div>"""
    else:
        grid_html = '<p class="muted">Сейчас сетки нет — бот ищет монету.</p>'

    rows = "".join(
        f"<tr><td>{escape(c['symbol'])}</td><td>{c['vol1m']:.2f}%</td><td>{c['atr']:.2f}%</td>"
        f"<td>{c['er']:.2f}</td><td>{c['score']:.2f}</td></tr>"
        for c in s["scan"]["top"]
    ) or '<tr><td colspan="5" class="muted">Скан ещё не прошёл</td></tr>'

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
<p class="muted small">«Стоп» снимает ордера, закрывает позицию по рынку и ставит бота на паузу. «Старт» снова ищет монету и ставит сетку.</p>"""

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
.cmsg{{color:var(--muted);font-size:13px}}.small{{font-size:12px;margin:8px 0 0}}
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
<div class="scroll"><table><tr><th>Монета</th><th>1m размах</th><th>ATR</th><th>ER</th><th>Score</th></tr>{rows}</table></div></div>
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
