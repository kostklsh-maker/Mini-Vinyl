#!/usr/bin/env python3
"""Собирает out/preview.html из preview_template.html и данных, которые пишет gramophone.py."""
import json, os
D = os.path.dirname(os.path.abspath(__file__))
import gramophone as g
summary = json.load(open(os.path.join(D, "out", "parts.json")))
plate_of = {}
for i, (name, keys) in enumerate(g.PLATES, 1):
    for k in keys:
        plate_of[k] = str(i)
for s in summary:
    s["plate"] = plate_of[s["key"]]
t = open(os.path.join(D, "preview_template.html"), encoding="utf-8").read()
t = t.replace("__DATA__", open(os.path.join(D, "out", "preview.json")).read())
t = t.replace("__SUMMARY__", json.dumps(summary, ensure_ascii=False))
t = t.replace("__NPARTS__", str(sum(s["count"] for s in summary))).replace("__DIMS__", "126 × 126 × %d" % round(g.build_height()))
open(os.path.join(D, "out", "preview.html"), "w", encoding="utf-8").write(t)
print("out/preview.html", os.path.getsize(os.path.join(D, "out", "preview.html")) // 1024, "КБ")
