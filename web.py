"""Сервер страницы бота: дашборд + API.

GET  /                 дашборд (dashboard.html)
GET  /api/state        состояние автопилота
GET  /api/candles      свечи Bybit: ?symbol=BTCUSDT&tf=1|5|15&limit=300
GET  /api/backtest     честный бэктест: ?symbol=BTCUSDT&days=30&balance=10000&risk=1
POST /api/control      {"action": "start|pause|close|close_all", "symbol": "...", "key": "..."}
"""

import hmac
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

CONTROL_KEY = os.getenv("CONTROL_KEY", "")
HERE = os.path.dirname(os.path.abspath(__file__))
BOT = None
_candle_cache = {}
_bt_lock = threading.Lock()


def _json(obj):
    return json.dumps(obj, ensure_ascii=False, default=str).encode()


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                with open(os.path.join(HERE, "dashboard.html"), "rb") as f:
                    return self.send(200, f.read(), "text/html; charset=utf-8")
            if u.path == "/api/state":
                return self.send(200, _json(BOT.snapshot()))
            if u.path == "/api/candles":
                sym = q.get("symbol", "BTCUSDT").upper()
                tf = q.get("tf", "1")
                if tf not in ("1", "5", "15", "60"):
                    tf = "1"
                limit = max(10, min(1000, int(q.get("limit", 300))))
                key = (sym, tf, limit)
                hit = _candle_cache.get(key)
                if not hit or time.time() - hit[0] > 2:
                    from bybit_util import klines
                    hit = (time.time(), klines(BOT.public, sym, tf, limit=limit))
                    _candle_cache[key] = hit
                return self.send(200, _json(hit[1]))
            if u.path == "/api/backtest":
                sym = q.get("symbol", "BTCUSDT").upper()
                if sym not in BOT.syms:
                    return self.send(400, _json({"error": "Монета не в списке бота"}))
                with _bt_lock:
                    res = BOT.backtest(sym, float(q.get("days", 30)), float(q.get("balance", 10000)),
                                       float(q.get("risk", BOT.p.risk_pct)))
                return self.send(200, _json(res))
            return self.send(404, _json({"error": "not found"}))
        except Exception as e:  # noqa: BLE001
            logging.getLogger("bot").warning("web %s: %s", u.path, e)
            return self.send(500, _json({"error": str(e)}))

    def do_POST(self):  # noqa: N802
        if urlparse(self.path).path != "/api/control":
            return self.send(404, _json({"error": "not found"}))
        n = int(self.headers.get("Content-Length") or 0)
        try:
            data = json.loads(self.rfile.read(min(n, 4096)) or b"{}")
        except ValueError:
            data = {}
        if CONTROL_KEY and not hmac.compare_digest(str(data.get("key") or ""), CONTROL_KEY):
            return self.send(403, _json({"error": "Невірний ключ керування"}))
        action = data.get("action")
        if action not in ("start", "pause", "close", "close_all", "optimize"):
            return self.send(400, _json({"error": "Невідома дія"}))
        BOT.command(action, data.get("symbol"))
        return self.send(200, _json({"ok": True}))

    def log_message(self, *_):
        pass


def start(bot):
    global BOT
    BOT = bot
    port = int(os.getenv("PORT") or 8080)
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    logging.getLogger("bot").info("Страница на порту %d", port)
