"""Тесты gmtv против фальшивых часов. Запуск: python -m unittest discover -s tests"""

import io
import json
import sys
import threading
import unittest
from pathlib import Path
from urllib.request import Request, build_opener, ProxyHandler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import fake_device  # noqa: E402
import gmtv  # noqa: E402
from PIL import Image  # noqa: E402

OPENER = build_opener(ProxyHandler({}))


def photo(w=1600, h=900, fmt="PNG", mode="RGB") -> bytes:
    im = Image.new(mode, (w, h), (200, 30, 30) if mode == "RGB" else (200, 30, 30, 128))
    for x in range(0, w, 50):
        im.paste((20, 20, 220) if mode == "RGB" else (20, 20, 220, 255), (x, 0, x + 25, h))
    buf = io.BytesIO()
    im.save(buf, fmt)
    return buf.getvalue()


def gif(frames=40, size=320) -> bytes:
    ims = [Image.new("RGB", (size, size), (i * 6 % 255, 100, 255 - i * 6 % 255)) for i in range(frames)]
    buf = io.BytesIO()
    ims[0].save(buf, "GIF", save_all=True, append_images=ims[1:], duration=50, loop=0)
    return buf.getvalue()


class PrepareTests(unittest.TestCase):
    def test_any_photo_becomes_baseline_240_jpeg(self):
        data, is_gif = gmtv.prepare(photo(), "fill")
        self.assertFalse(is_gif)
        im = Image.open(io.BytesIO(data))
        self.assertEqual((im.format, im.size), ("JPEG", (240, 240)))
        self.assertFalse(im.info.get("progressive") or im.info.get("progression"))
        self.assertLessEqual(len(data), gmtv.JPEG_LIMIT)

    def test_transparent_png_and_fit_mode(self):
        data, _ = gmtv.prepare(photo(800, 200, mode="RGBA"), "fit", bg="#00ff00")
        im = Image.open(io.BytesIO(data)).convert("RGB")
        r, g, b = im.getpixel((120, 5))      # поле сверху залито фоном
        self.assertTrue(g > 200 and r < 60 and b < 60)

    def test_animated_gif_stays_animated_and_small(self):
        data, is_gif = gmtv.prepare(gif(), "fill")
        im = Image.open(io.BytesIO(data))
        self.assertTrue(is_gif)
        self.assertEqual(im.size, (240, 240))
        self.assertGreater(im.n_frames, 1)
        self.assertLessEqual(len(data), gmtv.GIF_LIMIT)

    def test_garbage_is_rejected(self):
        with self.assertRaises(ValueError):
            gmtv.prepare(b"not an image")

    def test_text_render_fits_cyrillic(self):
        im = gmtv.render_text("Не беспокоить, идёт запись подкаста 🎧")
        self.assertEqual(im.size, (240, 240))
        self.assertTrue(im.getbbox())


class DeviceTests(unittest.TestCase):
    def setUp(self):
        self.server, self.state = fake_device.start()
        self.dev = gmtv.Device(f"127.0.0.1:{self.server.server_port}")

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_push_uploads_switches_to_album_and_selects(self):
        bridge = gmtv.Bridge(self.dev)
        bridge.push_image(photo())
        self.assertEqual(self.state.theme, 3)
        self.assertEqual(self.state.image, gmtv.LIVE_JPG)
        self.assertEqual(self.state.autoplay, 0, "слайд-шоу должно быть выключено")
        self.assertIn(gmtv.LIVE_JPG, self.state.files)
        bridge.push_image(gif())
        self.assertEqual(self.state.image, gmtv.LIVE_GIF)
        self.assertEqual(sorted(self.state.files), [gmtv.LIVE_GIF, gmtv.LIVE_JPG], "файлы перезаписываются")

    def test_brightness_theme_info_files(self):
        self.dev.brightness(150)
        self.assertEqual(self.state.brightness, 100)
        self.dev.theme(5)
        info = self.dev.info()
        self.assertEqual(info["version"]["m"], "SmallTV-Ultra")
        self.assertEqual(info["app"]["theme"], 5)
        self.dev.push_prepared(b"\xff\xd8x", False)
        self.assertEqual([f["name"] for f in self.dev.files()], [gmtv.LIVE_JPG])
        self.dev.delete(gmtv.LIVE_JPG)
        self.assertEqual(self.dev.files(), [])

    def test_unreachable_device_gives_clear_error(self):
        dead = gmtv.Device("127.0.0.1:1", timeout=1)
        self.assertIn("error", dead.info()["version"])
        with self.assertRaises(gmtv.DeviceError):
            dead.brightness(1)

    def test_set_fail_is_reported(self):
        with self.assertRaises(gmtv.DeviceError):
            self.dev.show_file("missing.jpg")


class BridgeHttpTests(unittest.TestCase):
    def setUp(self):
        self.dev_server, self.state = fake_device.start()
        dev = gmtv.Device(f"127.0.0.1:{self.dev_server.server_port}")
        self.bridge = gmtv.Bridge(dev, token="s3cret")
        from http.server import ThreadingHTTPServer
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), gmtv.make_handler(self.bridge))
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.srv.server_port}"

    def tearDown(self):
        for srv in (self.srv, self.dev_server):
            srv.shutdown()
            srv.server_close()

    def call(self, path, data=None, headers=None, token=True):
        h = dict(headers or {})
        if token:
            h["X-Token"] = "s3cret"
        req = Request(self.base + path, data=data, headers=h, method="POST" if data is not None else "GET")
        try:
            with OPENER.open(req, timeout=20) as r:
                return r.status, r.read(), r.headers
        except Exception as err:  # HTTPError несёт код
            return err.code, err.read(), err.headers

    def test_raw_body_push(self):   # так шлют iOS «Команды», HTTP Shortcuts, curl --data-binary
        code, body, _ = self.call("/api/push?fit=fit", photo(), {"Content-Type": "image/png"})
        self.assertEqual(code, 200, body)
        self.assertTrue(json.loads(body)["ok"])
        self.assertEqual(self.state.image, gmtv.LIVE_JPG)

    def test_multipart_push_and_preview(self):
        boundary = "XyZ"
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"cat.png\"\r\n"
                f"Content-Type: image/png\r\n\r\n").encode() + photo() + f"\r\n--{boundary}--\r\n".encode()
        code, out, _ = self.call("/api/push", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        self.assertEqual(code, 200, out)
        code, img, headers = self.call("/api/last")
        self.assertEqual((code, headers["Content-Type"]), (200, "image/jpeg"))
        self.assertEqual(Image.open(io.BytesIO(img)).size, (240, 240))

    def test_share_target_redirects(self):
        boundary = "B"
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.gif\"\r\n"
                f"Content-Type: image/gif\r\n\r\n").encode() + gif(5) + f"\r\n--{boundary}--\r\n".encode()
        opener = build_opener(ProxyHandler({}))
        req = Request(self.base + "/share", data=body, method="POST",
                      headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "X-Token": "s3cret"})
        with opener.open(req, timeout=20) as r:   # urllib идёт по 303 на /
            self.assertIn("/?shared=ok", r.url)
        self.assertEqual(self.state.image, gmtv.LIVE_GIF)

    def test_text_brightness_theme_and_token(self):
        self.assertEqual(self.call("/api/text?text=hi", token=False)[0], 401)
        code, body, _ = self.call("/api/text", json.dumps({"text": "Привет"}).encode(),
                                  {"Content-Type": "application/json"})
        self.assertEqual(code, 200, body)
        self.assertEqual(self.call("/api/brightness?value=30")[0], 200)
        self.assertEqual(self.state.brightness, 30)
        self.assertEqual(self.call("/api/theme?value=1")[0], 200)
        code, body, _ = self.call("/api/info")
        self.assertEqual(json.loads(body)["app"]["theme"], 1)
        self.assertEqual(self.call("/")[0], 200)
        self.assertEqual(self.call("/manifest.webmanifest")[0], 200)
        self.assertEqual(self.call("/icon-192.png")[0], 200)

    def test_bad_input_and_dead_device(self):
        code, body, _ = self.call("/api/push", b"hello", {"Content-Type": "text/plain"})
        self.assertEqual(code, 400)
        self.dev_server.shutdown()
        self.dev_server.server_close()
        self.bridge.device.timeout = 1
        code, body, _ = self.call("/api/push", photo(), {"Content-Type": "image/png"})
        self.assertEqual(code, 502, body)


class WatchTests(unittest.TestCase):
    def test_new_file_in_folder_goes_to_screen(self):
        import tempfile
        server, state = fake_device.start()
        bridge = gmtv.Bridge(gmtv.Device(f"127.0.0.1:{server.server_port}"))
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "old.png").write_bytes(photo(50, 50))
            threading.Thread(target=gmtv.watch, args=(Path(d), bridge, 0.1), daemon=True).start()
            import time
            time.sleep(0.3)
            self.assertIsNone(state.image, "старые файлы не показываются")
            (Path(d) / "Screenshot_1.png").write_bytes(photo())
            for _ in range(50):
                if state.image:
                    break
                time.sleep(0.1)
        server.shutdown()
        server.server_close()
        self.assertEqual(state.image, gmtv.LIVE_JPG)


if __name__ == "__main__":
    unittest.main()
