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
ELEC = [  # № на модели, ключ макета, короткая подпись, название, маркировка на плате / в лоте, шт., где стоит, подключение
    dict(key="e_esp", short="ESP32", name="Контроллер ESP32", mark="ESP32-WROOM-32 DevKit V1, 30 pin, USB-C", qty=1,
         where="на ребре вдоль левой стенки, USB-C в заднюю стенку", pins="— (центр системы)"),
    dict(key="e_pn532", short="PN532", name="Считыватель NFC", mark="PN532 NFC RFID V3 (красная плата)", qty=1,
         where="под пластинкой, в световом стакане", pins="SDA 21 · SCL 22 · IRQ 32 · RST 33 · 3V3"),
    dict(key="e_ring", short="Кольцо", name="Светодиодное кольцо", mark="WS2812B ring, 24 LED", qty=1,
         where="под пластинкой, в стакане под рассеивателем", pins="DI 13 (через 330 Ом) · 5 В"),
    dict(key="e_enc", short="Ручка", name="Ручка громкости с подсветкой", mark="M5Stack Unit Encoder U135", qty=1,
         where="лицо, в белом держателе", pins="Grove: SDA 21 · SCL 22 · 5 В (адрес 0x40)"),
    dict(key="e_amp", short="Усилитель", name="Усилитель", mark="MAX98357A I2S 3W", qty=1,
         where="пол основания, слева спереди", pins="BCLK 26 · LRC 25 · DIN 27 · SD 4 · 5 В"),
    dict(key="e_exc", short="Вибродинамик", name="Вибродинамик (exciter)", mark="50 мм, 4 Ом, 25 Вт", qty=1,
         where="пол основания, приклеен пятаком", pins="к выходу +/− усилителя"),
    dict(key="e_dac", short="ЦАП / AUX", name="ЦАП с гнездом AUX", mark="GY-PCM5102 (PCM5102A)", qty=1,
         where="задняя стенка, нижняя полка «этажерки»", pins="BCK 26 · LCK 25 · DIN 27 · SCK→GND · 5 В"),
    dict(key="e_sd", short="microSD", name="Модуль карты памяти", mark="Micro SD SPI mini", qty=1,
         where="задняя стенка, над ЦАП", pins="CS 5 · SCK 18 · MOSI 23 · MISO 19"),
    dict(key=None, short="", name="Карта памяти", mark="microSD 16–32 ГБ, FAT32", qty=1,
         where="в модуле microSD, щель в задней стенке", pins="—"),
    dict(key=None, short="", name="Конденсатор", mark="1000 мкФ 16 В", qty=1,
         where="у разъёма кольца", pins="между 5 В и GND"),
    dict(key=None, short="", name="Резистор", mark="330 Ом", qty=1,
         where="в проводе данных кольца", pins="GPIO 13 → DI кольца"),
]
n = 0
for e in ELEC:
    if e["key"]:
        n += 1; e["n"] = str(n)
    else:
        e["n"] = "—"
t = t.replace("__ELEC__", json.dumps(ELEC, ensure_ascii=False))
t = t.replace("__NPARTS__", str(sum(s["count"] for s in summary))).replace("__DIMS__", "126 × 126 × %d" % round(g.build_height()))
open(os.path.join(D, "out", "preview.html"), "w", encoding="utf-8").write(t)
print("out/preview.html", os.path.getsize(os.path.join(D, "out", "preview.html")) // 1024, "КБ")
