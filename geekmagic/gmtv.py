#!/usr/bin/env python3
"""gmtv: картинки, GIF и текст на экран GeekMagic SmallTV Ultra по Wi-Fi.

Работает со стоковой прошивкой Ultra-V9.x (API взято из открытых проектов
geekmagic-hacs и TokenTV, отчёт по V9.0.40): прошивать часы не нужно. Один файл, нужен только
Pillow (`pip install pillow`). Подходит для ПК, Raspberry Pi и телефона
на Android через Termux.

Команды (подробно: python gmtv.py <команда> -h):
  find                      найти часы в локальной сети
  info                      модель, прошивка, место, яркость
  push  <файл|URL|->        показать картинку или GIF (любой размер и формат)
  text  "Привет"            показать текст
  brightness 0..100         яркость
  theme N                   встроенный экран: 1 часы-погода, 2 прогноз,
                            3 фотоальбом, 4-6 часы, 7 простые часы
  watch <папка>             следить за папкой и показывать каждую новую картинку
  serve                     мост: веб-страница + HTTP API для телефона

Адрес часов: --device 192.168.1.196 или переменная окружения GMTV_DEVICE.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import email.parser
import email.policy
import http.client
import io
import ipaddress
import json
import os
import re
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlparse
from urllib.request import ProxyHandler, Request, build_opener

try:
    from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps, ImageSequence
except ImportError:  # команды info/brightness/theme/find работают и без Pillow
    Image = None
try:  # фото с iPhone (HEIC), если установлен pillow-heif
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

__version__ = "1.0.0"

SIZE = 240                 # экран 240x240, ST7789
JPEG_LIMIT = 60_000        # стоковый веб-интерфейс шлёт ~20-60 КБ; больше не нужно
GIF_LIMIT = 400_000        # во флеше ~3 МБ, свободно обычно ~0,9 МБ
GIF_MAX_FRAMES = 150
PHOTO_THEME = 3            # «Фотоальбом»: единственный режим, где видна своя картинка
LIVE_JPG = "gmtv.jpg"      # одно и то же имя файла: перезапись не забивает флеш
LIVE_GIF = "gmtv.gif"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".heic", ".tif", ".tiff"}
MAX_BODY = 30 * 1024 * 1024

THEMES = {
    1: "Часы и погода",
    2: "Прогноз погоды",
    3: "Фотоальбом (свои картинки)",
    4: "Часы, стиль 1",
    5: "Часы, стиль 2",
    6: "Часы, стиль 3",
    7: "Простые часы с погодой",
}

FONT_CANDIDATES = [
    "/system/fonts/Roboto-Regular.ttf",                       # Android / Termux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",        # Debian, Raspberry Pi OS
    "/usr/share/fonts/TTF/DejaVuSans.ttf",                    # Arch
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",      # Fedora
    "/System/Library/Fonts/Supplemental/Arial.ttf",           # macOS
    "/Library/Fonts/Arial Unicode.ttf",
    "C:/Windows/Fonts/arial.ttf",                             # Windows
    "C:/Windows/Fonts/segoeui.ttf",
]


class DeviceError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Часы: стоковое HTTP API
# --------------------------------------------------------------------------

class Device:
    """Стоковая прошивка SmallTV Ultra. HTTP без пароля, всё через GET /set?..."""

    def __init__(self, host: str, timeout: float = 15):
        host = host.strip().rstrip("/")
        if not host.startswith(("http://", "https://")):
            host = "http://" + host
        self.base = host
        self.timeout = timeout
        # Часы в локальной сети: системный прокси (если есть) только мешает.
        self._opener = build_opener(ProxyHandler({}))
        self._lock = threading.Lock()
        self._album_checked = False

    # --- транспорт -------------------------------------------------------
    def request(self, path: str, data: bytes | None = None, headers: dict | None = None) -> bytes:
        req = Request(self.base + path, data=data, headers=headers or {},
                      method="POST" if data is not None else "GET")
        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                return resp.read()
        except HTTPError as err:
            raise DeviceError(f"{path}: HTTP {err.code}") from err
        except (URLError, socket.timeout, ConnectionError, http.client.HTTPException) as err:
            raise DeviceError(f"часы {self.base} не отвечают ({err!r})") from err

    def get_json(self, path: str) -> dict:
        raw = self.request(path)
        try:
            return json.loads(raw.decode("utf-8", "replace"))
        except ValueError as err:
            raise DeviceError(f"{path}: ответ не JSON: {raw[:80]!r}") from err

    def set(self, **params) -> None:
        query = "&".join(f"{k}={quote(str(v), safe='/')}" for k, v in params.items())
        body = self.request("/set?" + query).decode("utf-8", "replace").strip()
        if body.upper().startswith("FAIL"):
            raise DeviceError(f"часы отклонили /set?{query}")

    # --- чтение ----------------------------------------------------------
    def info(self) -> dict:
        out = {"device": self.base}
        for key, path in (("version", "/v.json"), ("space", "/space.json"),
                          ("brightness", "/brt.json"), ("app", "/app.json"),
                          ("album", "/album.json")):
            try:
                out[key] = self.get_json(path)
            except DeviceError as err:
                out[key] = {"error": str(err)}
        return out

    def files(self) -> list[dict]:
        html = self.request("/filelist?dir=/image/").decode("utf-8", "replace")
        rx = re.compile(r"<a href='(?P<path>[^']+)'>(?P<name>[^<]+)</a></td><td>(?P<kb>[^<]+)</td>")
        return [{"path": m["path"], "name": m["name"], "kb": m["kb"]} for m in rx.finditer(html)]

    # --- запись ----------------------------------------------------------
    def brightness(self, value: int) -> None:
        self.set(brt=max(-10, min(100, int(value))))

    def theme(self, n: int) -> None:
        self.set(theme=int(n))

    def delete(self, name: str) -> None:
        self.request("/delete?file=" + quote("/image/" + name, safe="/"))

    def upload(self, name: str, data: bytes, mime: str) -> None:
        boundary = "gmtv" + os.urandom(8).hex()
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                f"filename=\"{name}\"\r\nContent-Type: {mime}\r\n\r\n").encode() \
            + data + f"\r\n--{boundary}--\r\n".encode()
        try:
            self.request("/doUpload?dir=/image/", body,
                         {"Content-Type": "multipart/form-data; boundary=" + boundary})
        except DeviceError as err:
            # Прошивка отвечает на загрузку кривым HTTP (двойной Content-Length,
            # данные после Connection: close). Файл к этому моменту уже записан.
            if not isinstance(err.__cause__, (http.client.HTTPException, ConnectionResetError)):
                raise

    def _ensure_photo_mode(self) -> None:
        # Слайд-шоу альбома не должно перелистнуть на чужое фото.
        if not self._album_checked:
            try:
                album = self.get_json("/album.json")
                if int(album.get("autoplay", 0)):
                    self.set(i_i=int(album.get("i_i", 10) or 10), autoplay=0)
            except DeviceError:
                pass
            self._album_checked = True
        try:
            current = int(self.get_json("/app.json").get("theme", -1))
        except (DeviceError, ValueError, TypeError):
            current = -1    # у новых часов /app.json = 404, пока тему не меняли
        if current != PHOTO_THEME:
            self.theme(PHOTO_THEME)

    def show_file(self, name: str) -> None:
        self._ensure_photo_mode()
        self.set(img="/image/" + name)

    def push_prepared(self, data: bytes, is_gif: bool) -> str:
        """Загрузить уже готовый 240x240 JPEG/GIF и сразу показать."""
        name = LIVE_GIF if is_gif else LIVE_JPG
        with self._lock:
            self.upload(name, data, "image/gif" if is_gif else "image/jpeg")
            self.show_file(name)
        return name


# --------------------------------------------------------------------------
# Подготовка картинок
# --------------------------------------------------------------------------

def need_pillow() -> None:
    if Image is None:
        sys.exit("Нужен Pillow: pip install pillow  (в Termux: pkg install python-pillow)")


def _rgb(color: str) -> tuple[int, int, int]:
    return ImageColor.getrgb(color)[:3]


def _flatten(im, bg: str):
    """Любой режим -> RGB, прозрачность заливается фоном."""
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        base = Image.new("RGB", im.size, _rgb(bg))
        base.paste(im, mask=im.split()[-1])
        return base
    return im.convert("RGB")


def square(im, fit: str = "fill", bg: str = "#000000", rotate: int = 0):
    """240x240: fill — обрезать по центру, fit — вписать с полями."""
    im = _flatten(im, bg)
    if fit == "fit":
        im = ImageOps.contain(im, (SIZE, SIZE), Image.LANCZOS)
        canvas = Image.new("RGB", (SIZE, SIZE), _rgb(bg))
        canvas.paste(im, ((SIZE - im.width) // 2, (SIZE - im.height) // 2))
        im = canvas
    elif fit == "stretch":
        im = im.resize((SIZE, SIZE), Image.LANCZOS)
    else:
        im = ImageOps.fit(im, (SIZE, SIZE), Image.LANCZOS)
    if rotate:
        im = im.rotate(-rotate, expand=False)
    return im


def to_jpeg(im, limit: int = JPEG_LIMIT) -> bytes:
    # Только baseline JPEG: декодер часов не понимает progressive.
    data = b""
    for q in (90, 85, 80, 70, 60, 50, 40, 30, 20):
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=q, progressive=False)
        data = buf.getvalue()
        if len(data) <= limit:
            break
    return data


def to_gif(src, fit: str, bg: str, rotate: int, limit: int = GIF_LIMIT) -> bytes:
    frames, durations = [], []
    for frame in ImageSequence.Iterator(src):
        durations.append(int(frame.info.get("duration", src.info.get("duration", 100)) or 100))
        frames.append(square(frame.copy(), fit, bg, rotate))
    data = b""
    for step in range(1, len(frames) + 1):
        picked = frames[::step][:GIF_MAX_FRAMES]
        timing = [max(20, sum(durations[i:i + step])) for i in range(0, len(frames), step)][:len(picked)]
        colors = 256 if step == 1 else 128
        pal = [f.quantize(colors=colors, method=Image.Quantize.MEDIANCUT) for f in picked]
        buf = io.BytesIO()
        pal[0].save(buf, "GIF", save_all=True, append_images=pal[1:], duration=timing,
                    loop=0, optimize=True, disposal=1)
        data = buf.getvalue()
        if len(data) <= limit or len(picked) <= 2:
            break
    return data


def prepare(raw: bytes, fit: str = "fill", bg: str = "#000000", rotate: int = 0,
            keep_animation: bool = True) -> tuple[bytes, bool]:
    """Любые байты картинки -> (готовые байты, это GIF?)."""
    need_pillow()
    try:
        src = Image.open(io.BytesIO(raw))
        src.load()
    except Exception as err:  # noqa: BLE001 — Pillow бросает что угодно
        raise ValueError(f"не похоже на картинку ({err}); HEIC: pip install pillow-heif") from err
    if keep_animation and getattr(src, "n_frames", 1) > 1:
        return to_gif(src, fit, bg, rotate), True
    src = ImageOps.exif_transpose(src)
    return to_jpeg(square(src, fit, bg, rotate)), False


def find_font(path: str | None = None, size: int = 40):
    for candidate in ([path] if path else []) + FONT_CANDIDATES:
        if candidate and Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    try:
        return ImageFont.load_default(size)   # Pillow >= 10.1; кириллицы может не быть
    except TypeError:
        return ImageFont.load_default()


def _wrap(draw, text: str, font, width: int) -> list[str]:
    lines: list[str] = []
    for para in text.splitlines() or [""]:
        line = ""
        for word in para.split(" "):
            trial = (line + " " + word).strip()
            if draw.textlength(trial, font=font) <= width or not line:
                line = trial
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def render_text(text: str, fg: str = "#FFFFFF", bg: str = "#000000",
                font_path: str | None = None, rotate: int = 0):
    """Текст максимального размера, который влезает в экран."""
    need_pillow()
    # Цветных эмодзи в обычных шрифтах нет: вместо них были бы пустые квадраты.
    text = "".join(ch for ch in text if not (0x1F000 <= ord(ch) <= 0x1FAFF or 0x2600 <= ord(ch) <= 0x27BF
                                             or ord(ch) in (0xFE0F, 0x200D))).strip() or " "
    im = Image.new("RGB", (SIZE, SIZE), _rgb(bg))
    draw = ImageDraw.Draw(im)
    pad, box = 10, SIZE - 20
    for size in range(120, 9, -2):
        font = find_font(font_path, size)
        lines = _wrap(draw, text, font, box)
        body = "\n".join(lines)
        l, t, r, b = draw.multiline_textbbox((0, 0), body, font=font, spacing=size // 6, align="center")
        if r - l <= box and b - t <= box and all(draw.textlength(x, font=font) <= box for x in lines):
            break
    x = pad + (box - (r - l)) / 2 - l
    y = pad + (box - (b - t)) / 2 - t
    draw.multiline_text((x, y), body, font=font, fill=_rgb(fg), spacing=size // 6, align="center")
    if rotate:
        im = im.rotate(-rotate)
    return im


def load_source(src: str) -> bytes:
    if src == "-":
        return sys.stdin.buffer.read()
    if src.startswith(("http://", "https://")):
        req = Request(src, headers={"User-Agent": "gmtv/" + __version__})
        with build_opener().open(req, timeout=30) as resp:
            return resp.read(MAX_BODY)
    return Path(src).expanduser().read_bytes()


# --------------------------------------------------------------------------
# Поиск часов в сети
# --------------------------------------------------------------------------

def local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))   # пакет не уходит, просто выбор интерфейса
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def probe(ip: str, timeout: float = 0.8) -> dict | None:
    try:
        conn = http.client.HTTPConnection(ip, 80, timeout=timeout)
        conn.request("GET", "/v.json")
        resp = conn.getresponse()
        if resp.status != 200:
            return None
        data = json.loads(resp.read(512).decode("utf-8", "replace"))
        model = str(data.get("m", ""))
        if "SmallTV" in model or "GeekMagic" in model or model.startswith("Ultra"):
            return {"ip": ip, "model": model, "version": data.get("v")}
    except (OSError, ValueError, http.client.HTTPException):
        return None
    return None


def find_devices(subnet: str | None = None) -> list[dict]:
    net = ipaddress.ip_network(subnet or (local_ip() + "/24"), strict=False)
    hosts = [str(h) for h in net.hosts()]
    with concurrent.futures.ThreadPoolExecutor(64) as pool:
        return [r for r in pool.map(probe, hosts) if r]


# --------------------------------------------------------------------------
# Мост: веб-страница и HTTP API (для телефона, Tasker, iOS «Команд», скриптов)
# --------------------------------------------------------------------------

WEB_DIR = Path(__file__).resolve().parent / "web"


def parse_multipart(content_type: str, body: bytes) -> dict[str, list]:
    """{"поле": [str | (filename, bytes)]} без модуля cgi (его нет в Python 3.13)."""
    msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
        b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body)
    fields: dict[str, list] = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        payload = part.get_payload(decode=True) or b""
        filename = part.get_filename()
        value = (filename, payload) if filename is not None else payload.decode("utf-8", "replace")
        fields.setdefault(name, []).append(value)
    return fields


class Bridge:
    def __init__(self, device: Device, fit: str = "fill", bg: str = "#000000",
                 rotate: int = 0, token: str | None = None, font: str | None = None):
        self.device, self.fit, self.bg, self.rotate = device, fit, bg, rotate
        self.token, self.font = token, font
        self.last_preview: bytes | None = None
        self.last_is_gif = False
        self.history: list[dict] = []

    def _remember(self, data: bytes, is_gif: bool, what: str) -> dict:
        self.last_preview, self.last_is_gif = data, is_gif
        entry = {"time": time.strftime("%H:%M:%S"), "what": what, "bytes": len(data),
                 "type": "gif" if is_gif else "jpeg"}
        self.history = [entry] + self.history[:19]
        log(f"на экране: {what} ({len(data) // 1024} КБ {entry['type']})")
        return entry

    def push_image(self, raw: bytes, fit: str | None = None, what: str = "картинка",
                   animate: bool = True) -> dict:
        data, is_gif = prepare(raw, fit or self.fit, self.bg, self.rotate, animate)
        self.device.push_prepared(data, is_gif)
        return self._remember(data, is_gif, what)

    def push_text(self, text: str, fg: str = "#FFFFFF", bg: str | None = None) -> dict:
        im = render_text(text, fg, bg or self.bg, self.font, self.rotate)
        data = to_jpeg(im)
        self.device.push_prepared(data, False)
        return self._remember(data, False, "текст: " + text[:40])


def make_handler(bridge: Bridge):
    class Handler(BaseHTTPRequestHandler):
        server_version = "gmtv/" + __version__

        def log_message(self, fmt, *args):  # тише стандартного
            pass

        # --- ответы ---------------------------------------------------
        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def _error(self, code: int, msg: str):
            log(f"ошибка {code}: {msg}")
            self._json({"ok": False, "error": msg}, code)

        def _authorized(self, query: dict) -> bool:
            if not bridge.token:
                return True
            given = self.headers.get("X-Token") or (query.get("token") or [""])[0]
            return given == bridge.token

        def _body(self) -> bytes:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise ValueError("файл больше 30 МБ")
            return self.rfile.read(length) if length else b""

        # --- маршруты -------------------------------------------------
        def do_OPTIONS(self):
            self._send(204, b"", "text/plain", {
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, X-Token"})

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            static = {"/": ("index.html", "text/html; charset=utf-8"),
                      "/index.html": ("index.html", "text/html; charset=utf-8"),
                      "/manifest.webmanifest": ("manifest.webmanifest", "application/manifest+json"),
                      "/sw.js": ("sw.js", "text/javascript; charset=utf-8")}
            if url.path in static:
                name, ctype = static[url.path]
                return self._send(200, (WEB_DIR / name).read_bytes(), ctype)
            if url.path in ("/icon-192.png", "/icon-512.png"):
                return self._send(200, app_icon(192 if "192" in url.path else 512), "image/png")
            if not self._authorized(query):
                return self._error(401, "нужен токен")
            if url.path == "/api/info":
                try:
                    info = bridge.device.info()
                except DeviceError as err:
                    return self._error(502, str(err))
                info["history"] = bridge.history
                info["themes"] = THEMES
                return self._json(info)
            if url.path == "/api/last":
                if not bridge.last_preview:
                    return self._error(404, "ещё ничего не отправляли")
                return self._send(200, bridge.last_preview,
                                  "image/gif" if bridge.last_is_gif else "image/jpeg")
            if url.path == "/api/brightness" and "value" in query:
                return self._wrap(lambda: bridge.device.brightness(int(query["value"][0])))
            if url.path == "/api/theme" and "value" in query:
                return self._wrap(lambda: bridge.device.theme(int(query["value"][0])))
            if url.path == "/api/text" and "text" in query:
                return self._wrap(lambda: bridge.push_text(
                    query["text"][0], (query.get("fg") or ["#FFFFFF"])[0], (query.get("bg") or [None])[0]))
            return self._error(404, "нет такого адреса")

        def do_POST(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if not self._authorized(query):
                return self._error(401, "нужен токен")
            try:
                body = self._body()
            except ValueError as err:
                return self._error(413, str(err))
            ctype = self.headers.get("Content-Type", "")
            form = parse_multipart(ctype, body) if ctype.startswith("multipart/form-data") else {}
            fit = (query.get("fit") or [None])[0] or (form.get("fit") or [None])[0]
            animate = (query.get("animate") or ["1"])[0] != "0"

            if url.path in ("/api/push", "/share"):
                files = [v for vals in form.values() for v in vals if isinstance(v, tuple) and v[1]]
                if form and not files:
                    # share_target без файла: возможно, поделились ссылкой на картинку
                    link = next((v for k in ("url", "text") for v in form.get(k, [])
                                 if isinstance(v, str) and v.startswith("http")), None)
                    if not link:
                        return self._error(400, "в запросе нет картинки")
                    try:
                        raw, what = load_source(link.split()[0]), link[:60]
                    except Exception as err:  # noqa: BLE001
                        return self._error(400, f"не удалось скачать {link[:60]}: {err}")
                elif files:
                    what, raw = files[0]
                else:
                    raw, what = body, "картинка"
                if not raw:
                    return self._error(400, "пустое тело запроса")
                result = self._wrap(lambda: bridge.push_image(raw, fit, what or "картинка", animate),
                                    reply=url.path != "/share")
                if url.path == "/share":
                    status = "ok" if result else "error"
                    self.send_response(303)
                    self.send_header("Location", f"/?shared={status}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                return None

            if url.path == "/api/text":
                if ctype.startswith("application/json"):
                    data = json.loads(body or b"{}")
                elif form:
                    data = {k: v[0] for k, v in form.items() if isinstance(v[0], str)}
                else:
                    data = {"text": body.decode("utf-8", "replace")}
                text = str(data.get("text", "")).strip()
                if not text:
                    return self._error(400, "пустой текст")
                return self._wrap(lambda: bridge.push_text(text, data.get("fg") or "#FFFFFF", data.get("bg")))

            return self._error(404, "нет такого адреса")

        def _wrap(self, action, reply: bool = True):
            try:
                result = action()
            except ValueError as err:
                if reply:
                    self._error(400, str(err))
                return None
            except DeviceError as err:
                if reply:
                    self._error(502, str(err))
                return None
            if reply:
                self._json({"ok": True, **(result if isinstance(result, dict) else {})})
            return result or True

    return Handler


_ICONS: dict[int, bytes] = {}


def app_icon(size: int) -> bytes:
    """Иконка PWA рисуется на лету: в репозитории нет бинарников."""
    if size in _ICONS:
        return _ICONS[size]
    if Image is None:
        return b""
    im = Image.new("RGB", (size, size), (18, 18, 20))
    d = ImageDraw.Draw(im)
    m = size // 6
    d.rounded_rectangle((m, m, size - m, size - m), radius=size // 10, fill=(40, 40, 46))
    s = size // 4
    d.rounded_rectangle((s, s, size - s, size - s), radius=size // 24, fill=(255, 170, 40))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    _ICONS[size] = buf.getvalue()
    return _ICONS[size]


def serve(bridge: Bridge, host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(bridge))
    shown = "127.0.0.1" if host in ("0.0.0.0", "127.0.0.1") else host
    log(f"мост запущен: http://{shown}:{port}  → часы {bridge.device.base}")
    if host == "0.0.0.0":
        log(f"из локальной сети: http://{local_ip()}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("остановлен")


# --------------------------------------------------------------------------
# Слежение за папкой (например, скриншоты телефона)
# --------------------------------------------------------------------------

def watch(folder: Path, bridge: Bridge, interval: float = 2.0) -> None:
    def scan() -> dict[Path, tuple[float, int]]:
        out = {}
        for p in folder.iterdir():
            if p.suffix.lower() in IMAGE_EXT and not p.name.startswith("."):
                try:
                    st = p.stat()
                    out[p] = (st.st_mtime, st.st_size)
                except OSError:
                    pass
        return out

    seen = scan()
    pending: dict[Path, tuple[float, int]] = {}
    log(f"слежу за {folder} (уже там {len(seen)} файлов, их не показываю)")
    while True:
        time.sleep(interval)
        try:
            now = scan()
        except OSError as err:
            log(f"папка недоступна: {err}")
            continue
        for path, stamp in now.items():
            if seen.get(path) == stamp:
                continue
            if pending.get(path) == stamp:          # размер не меняется: файл дописан
                pending.pop(path)
                seen[path] = stamp
                try:
                    bridge.push_image(path.read_bytes(), what=path.name)
                except (ValueError, DeviceError, OSError) as err:
                    log(f"{path.name}: {err}")
            else:
                pending[path] = stamp


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="gmtv", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", "-d", default=os.environ.get("GMTV_DEVICE"),
                   help="IP часов (или GMTV_DEVICE). Пример: 192.168.1.196")
    p.add_argument("--fit", choices=("fill", "fit", "stretch"), default="fill",
                   help="fill: обрезать до квадрата (по умолчанию), fit: вписать с полями")
    p.add_argument("--bg", default="#000000", help="цвет полей и фона текста")
    p.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0,
                   help="повернуть картинку, если часы стоят на боку")
    p.add_argument("--font", help="путь к .ttf для текста")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("find", help="найти часы в сети")
    s.add_argument("subnet", nargs="?", help="например 192.168.1.0/24")
    sub.add_parser("info", help="состояние часов")
    s = sub.add_parser("push", help="показать картинку/GIF: файл, URL или - (stdin)")
    s.add_argument("source")
    s.add_argument("--still", action="store_true", help="GIF показать первым кадром")
    s.add_argument("--save", help="сохранить подготовленный файл (для проверки)")
    s = sub.add_parser("text", help="показать текст")
    s.add_argument("text")
    s.add_argument("--fg", default="#FFFFFF")
    s = sub.add_parser("brightness", help="яркость 0..100 (до -10 — совсем темно)")
    s.add_argument("value", type=int)
    s = sub.add_parser("theme", help="встроенный экран: " + ", ".join(f"{k} {v}" for k, v in THEMES.items()))
    s.add_argument("value", type=int)
    sub.add_parser("files", help="файлы в альбоме часов")
    s = sub.add_parser("delete", help="удалить файл из альбома")
    s.add_argument("name")
    s = sub.add_parser("watch", help="показывать каждую новую картинку из папки")
    s.add_argument("folder")
    s.add_argument("--interval", type=float, default=2.0)
    s = sub.add_parser("serve", help="мост: веб-страница и API для телефона")
    s.add_argument("--host", default="0.0.0.0", help="0.0.0.0 — вся сеть, 127.0.0.1 — только этот телефон/ПК")
    s.add_argument("--port", type=int, default=8240)
    s.add_argument("--token", default=os.environ.get("GMTV_TOKEN"), help="пароль для API (X-Token или ?token=)")

    a = p.parse_args(argv)

    if a.cmd == "find":
        found = find_devices(a.subnet)
        for d in found:
            print(f"{d['ip']}\t{d['model']}\t{d['version']}")
        if not found:
            print("не нашёл. Проверьте, что телефон/ПК в той же сети Wi-Fi, что и часы.", file=sys.stderr)
        return 0 if found else 1

    if a.cmd == "push" and a.save:          # можно проверить подготовку без часов
        data, is_gif = prepare(load_source(a.source), a.fit, a.bg, a.rotate, not a.still)
        Path(a.save).write_bytes(data)
        print(f"{a.save}: {len(data)} байт, {'GIF' if is_gif else 'JPEG'}")
        if not a.device:
            return 0

    if not a.device:
        p.error("укажите --device IP или GMTV_DEVICE (адрес видно на экране часов при включении; "
                "или запустите: gmtv.py find)")
    dev = Device(a.device)
    bridge = Bridge(dev, a.fit, a.bg, a.rotate, getattr(a, "token", None), a.font)
    try:
        if a.cmd == "info":
            print(json.dumps(dev.info(), ensure_ascii=False, indent=2))
        elif a.cmd == "push":
            bridge.push_image(load_source(a.source), what=a.source, animate=not a.still)
        elif a.cmd == "text":
            bridge.push_text(a.text, a.fg)
        elif a.cmd == "brightness":
            dev.brightness(a.value)
        elif a.cmd == "theme":
            dev.theme(a.value)
        elif a.cmd == "files":
            for f in dev.files():
                print(f"{f['name']}\t{f['kb']} КБ")
        elif a.cmd == "delete":
            dev.delete(a.name)
        elif a.cmd == "watch":
            watch(Path(a.folder).expanduser(), bridge, a.interval)
        elif a.cmd == "serve":
            serve(bridge, a.host, a.port)
    except (DeviceError, ValueError, OSError) as err:
        print(f"ошибка: {err}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
