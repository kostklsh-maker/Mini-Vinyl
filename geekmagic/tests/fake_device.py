#!/usr/bin/env python3
"""Имитация стоковой прошивки SmallTV Ultra для разработки без часов.

    python tests/fake_device.py 8080
    python gmtv.py -d 127.0.0.1:8080 push photo.jpg
    открыть http://127.0.0.1:8080/ — там видно, что «на экране»

Повторяет то, что известно о прошивке Ultra-V9.0.x: /v.json, /app.json,
/album.json, /space.json, /brt.json, /set?..., POST /doUpload?dir=/image/
(поле file), /filelist, /delete. /set?img= отвечает FAIL, если файла нет.
"""

from __future__ import annotations

import email.parser
import email.policy
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

TOTAL = 3_121_152


class State:
    def __init__(self):
        self.theme = 1
        self.brightness = 50
        self.autoplay = 1
        self.interval = 10
        self.image: str | None = None
        self.files: dict[str, bytes] = {}
        self.log: list[str] = []
        self.lock = threading.Lock()


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype="text/plain", code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj):
            self._send(json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            state.log.append("GET " + self.path)
            if url.path == "/v.json":
                return self._json({"m": "SmallTV-Ultra", "v": "Ultra-V9.0.54"})
            if url.path == "/app.json":
                return self._json({"theme": state.theme})
            if url.path == "/brt.json":
                return self._json({"brt": str(state.brightness)})
            if url.path == "/album.json":
                return self._json({"autoplay": state.autoplay, "i_i": state.interval})
            if url.path == "/space.json":
                used = sum(map(len, state.files.values()))
                return self._json({"total": TOTAL, "free": TOTAL - 2_000_000 - used})
            if url.path == "/set":
                if "img" in q:
                    name = q["img"].rsplit("/", 1)[-1]
                    if name not in state.files:
                        return self._send(b"FAIL")
                    state.image = name
                if "theme" in q:
                    state.theme = int(q["theme"])
                if "brt" in q:
                    state.brightness = int(q["brt"])
                if "autoplay" in q:
                    state.autoplay = int(q["autoplay"])
                if "i_i" in q:
                    state.interval = int(q["i_i"])
                if q.get("clear") == "image":
                    state.files.clear()
                return self._send(b"OK")
            if url.path == "/filelist":
                rows = "".join(f"<tr><td><a href='/image/{n}'>{n}</a></td><td>{max(1, len(d) // 1024)}</td></tr>"
                               for n, d in state.files.items())
                return self._send(f"<table>{rows}</table>".encode(), "text/html")
            if url.path == "/delete":
                state.files.pop(q.get("file", "").rsplit("/", 1)[-1], None)
                return self._send(b"OK")
            if url.path.startswith("/image/"):
                data = state.files.get(url.path[7:])
                return self._send(data, "image/jpeg") if data else self._send(b"", code=404)
            if url.path == "/":
                showing = state.image if state.theme == 3 else None
                img = f"<img src='/image/{showing}' width=240 height=240>" if showing else f"тема {state.theme}"
                return self._send(f"<meta http-equiv=refresh content=2><body style='background:#000;color:#fff'>"
                                  f"{img}</body>".encode(), "text/html; charset=utf-8")
            self._send(b"", code=404)

        def do_POST(self):
            url = urlparse(self.path)
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            state.log.append("POST " + self.path)
            if url.path != "/doUpload":
                return self._send(b"", code=404)
            msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
                b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + body)
            for part in msg.iter_parts():
                if part.get_filename():
                    state.files[part.get_filename()] = part.get_payload(decode=True)
            self._send(b"OK")

    return Handler


def start(port: int = 0) -> tuple[ThreadingHTTPServer, State]:
    state = State()
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, state


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    srv, _ = start(port)
    print(f"фальшивые часы: http://127.0.0.1:{port}/  (Ctrl+C — выход)")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()
