#!/usr/bin/env python3
"""Mini-Vinyl — корпус-граммофон для NFC-проигрывателя на ESP32.

Генерирует все детали корпуса: STL по одной детали, столы 3MF для Bambu Studio (A1, сопло 0,4, слой 0,20)
и данные для 3D-превью (в сборе / в разборе).

    python3 gramophone.py            → out/stl/*.stl, out/3mf/*.3mf, out/preview.json

Нужны: Python 3, numpy, manifold3d.
Система координат сборки: X вправо, Y назад (лицо корпуса смотрит на −Y), Z вверх, мм.
Все размеры покупных плат — в словаре P; после замера реальных деталей поменяйте числа и запустите заново.
"""
import base64, json, math, os, struct, sys, zipfile
import numpy as np
from manifold3d import Manifold, CrossSection, Mesh, JoinType, OpType, set_circular_segments

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
set_circular_segments(72)
BED = 256.0

P = dict(
    fit=0.15,                 # посадочный зазор на сторону (как в Findypet)
    wall=2.4,                 # стенка корпуса
    # --- пластинка и NFC ---
    disc_d=50.0, disc_t=1.0,  # NFC-пластинка
    pn532=(43.0, 41.0, 4.5),  # плата PN532 V3: длина, ширина, высота с деталями
    ring_od=66.0, ring_id=52.0, ring_pcb=1.6,   # кольцо WS2812B на 24 диода
    # --- звук ---
    spk_d=40.0, spk_rim=3.0, spk_depth=19.0, spk_magnet=22.0,   # динамики Ø40 (2 боковых + 1 под трубой)
    # --- платы на основании ---
    esp=(28.3, 51.6),         # ESP32 DevKit 30 pin: ширина, длина
    amp=(18.0, 19.0),         # MAX98357A: ширина (вдоль гребёнки), длина
    dac=(17.5, 32.0),         # GY-PCM5102: ширина, длина (гнездо 3,5 мм на торце)
    sd=(24.0, 42.0),          # модуль microSD: ширина, длина (щель карты на торце)
    enc=(12.5, 13.5, 7.0),    # корпус энкодера EC11
)

# ------------------------------------------------------------------ геометрия корпуса
HB = 64.0                    # полуширина коробки стенок (128 × 128)
NOTCH = 12.0                 # угловая ниша под балясину
Z_FLOOR, Z_PL, Z_W1, Z_DECK, Z_TOP = 10.4, 18.0, 82.0, 89.0, 92.0   # пол, верх основания, низ карниза, низ и верх деки
C = np.array([-13.0, -11.0])  # центр диска (пластинка, свет, PN532)
H = np.array([33.0, 36.0])   # ось трубы граммофона и третьего динамика
SPK_Z = 50.0                 # высота центров боковых динамиков
BOSS = [(-58.6, -46.0), (58.6, -46.0), (-58.6, 46.0), (58.6, 46.0)]   # стойки под винты основания
EARS = [90.0, 210.0, 330.0]  # ушки светового стакана
EAR_R = 43.5
ESP_X, DAC_X, SD_X, ENC_X = -17.0, 10.0, 35.0, -42.0                 # разъёмы на задней стенке
USB_Z, AUX_Z, SD_Z, ENC_Z = 25.4, 30.0, 29.0, 44.0
PLAT_R, PLAT_T = 46.0, 4.0   # диск-платтер
RISER_L = 70.0               # высота стойки трубы над декой
BELL_L, BELL_R = 92.0, 62.0 # раструб: длина и радиус устья
TUBE_R, BORE_R, SPIG_R, SPIG_L = 15.0, 11.0, 13.0, 8.0
BEND_R, BEND_A = 40.0, 45.0
D_DIR = np.array([-1.0, -1.0, 0.0]) / math.sqrt(2)   # куда наклонён раструб
WIN = HB - P["wall"]         # внутренняя грань стенки
COLP = HB - NOTCH + 6.6      # центр балясины
GR_R, GR_T = 26.0, 2.2       # труба решётки: радиус и длина сквозь стенку
SCR_R = 23.15                # окружность винтов решётки
HORN_EARS = [135.0, 315.0]   # ушки прижима динамика трубы
AMP_Y = -52.0                # передний край отсека усилителей


# ------------------------------------------------------------------ помощники
def cyl(r, z0, z1, x=0.0, y=0.0, r2=None, seg=0):
    return Manifold.cylinder(z1 - z0, r, r if r2 is None else r2, seg).translate([x, y, z0])

def box(x0, y0, z0, x1, y1, z1):
    return Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])

def rrect(hx, hy, r):
    r = min(r, hx - 0.01, hy - 0.01)
    return CrossSection.square([2 * (hx - r), 2 * (hy - r)], True).offset(r, JoinType.Round)

def ext(cs, z0, z1):
    return cs.extrude(z1 - z0).translate([0, 0, z0])

def tiers(levels, rad):
    """Ступенчатый профиль карниза: levels = [(z, полуширина)], между уровнями — фаски."""
    parts = []
    for (z0, h0), (z1, h1) in zip(levels, levels[1:]):
        a = ext(rrect(h0, h0, rad + h0 - levels[0][1]), z0, z0 + 0.01)
        b = ext(rrect(h1, h1, rad + h1 - levels[0][1]), z1 - 0.01, z1)
        parts.append(Manifold.batch_hull([a, b]))
    return Manifold.batch_boolean(parts, OpType.Add)

def union(ms):
    return Manifold.batch_boolean(list(ms), OpType.Add)

def xf(m, cols, origin):
    """Аффинное преобразование: cols — образы осей X, Y, Z; origin — образ начала координат."""
    a = np.zeros((3, 4)); a[:, 0], a[:, 1], a[:, 2], a[:, 3] = cols[0], cols[1], cols[2], origin
    return m.transform(a)

def prism_x(cs, x0, x1):   # сечение (y, z) → тело вдоль X
    return xf(cs.extrude(x1 - x0), [(0, 1, 0), (0, 0, 1), (1, 0, 0)], (x0, 0, 0))

def prism_y(cs, y0, y1):   # сечение (−x, z) → тело вдоль Y
    return xf(cs.extrude(y1 - y0), [(-1, 0, 0), (0, 0, 1), (0, 1, 0)], (0, y0, 0))

def poly(pts):
    return CrossSection([np.asarray(pts, dtype=np.float64)])

def loft(rings):
    """Замкнутое тело по кольцам точек (одинаковое число точек, обход против часовой, ось вверх)."""
    rings = [np.asarray(r, dtype=np.float64) for r in rings]
    n, k = len(rings[0]), len(rings)
    v = np.vstack(rings + [rings[0].mean(0)[None], rings[-1].mean(0)[None]])
    t = []
    for i in range(k - 1):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            c, d = a + n, b + n
            t += [(a, b, d), (a, d, c)]
    cb, ct = k * n, k * n + 1
    for j in range(n):
        t.append((cb, (j + 1) % n, j))
        t.append((ct, (k - 1) * n + j, (k - 1) * n + (j + 1) % n))
    return Manifold(Mesh(vert_properties=v.astype(np.float32), tri_verts=np.asarray(t, dtype=np.uint32)))

def outline():
    """Контур коробки стенок: квадрат с угловыми нишами под балясины."""
    cs = CrossSection.square([2 * HB, 2 * HB], True)
    for sx in (-1, 1):
        for sy in (-1, 1):
            cs -= CrossSection.square([NOTCH + 1, NOTCH + 1], True).translate(
                [sx * (HB - NOTCH / 2 + 0.5), sy * (HB - NOTCH / 2 + 0.5)])
    return cs

def teardrop(r, flat_dir=-1):
    """Круг с «крышей» 45° (печать без поддержек). Плоская перемычка — в сторону flat_dir по оси v."""
    q, e = r / math.sqrt(2), r * (math.sqrt(2) - 1)
    roof = poly([(-q, q), (q, q), (e, r), (-e, r)] if flat_dir > 0 else [(-q, -q), (-e, -r), (e, -r), (q, -q)])
    return CrossSection.circle(r) + roof

def lens(w, h, ang):
    rl = (h * h / 4 + w * w / 4) / w
    cs = CrossSection.circle(rl).translate([rl - w / 2, 0]) ^ CrossSection.circle(rl).translate([-(rl - w / 2), 0])
    return cs.rotate(ang)


# ------------------------------------------------------------------ детали
def make_body():
    """Корпус: четыре стенки и верхняя дека одной деталью. Печатается декой вниз."""
    o = outline()
    inner = o.offset(-P["wall"], JoinType.Miter)
    cap = tiers([(Z_W1, HB + 2.5), (84.0, HB + 2.5), (85.5, HB + 5), (87.0, HB + 5), (88.5, HB + 8), (91.4, HB + 8), (Z_TOP, HB + 7.4)], 9.0)
    m = ext(o, Z_PL, Z_W1 + 0.5) + cap
    m -= ext(inner, Z_PL - 1, Z_DECK)
    # стойки под винты основания
    for x, y in BOSS:
        m += cyl(4.5, Z_PL, Z_DECK + 0.5, x, y) ^ ext(o, Z_PL, Z_TOP)
        m -= cyl(1.3, Z_PL - 1, Z_PL + 14, x, y)
    # воротник светового стакана и ушки
    cx, cy = C
    m += cyl(39.0, 79.0, Z_DECK + 0.5, cx, cy)
    for a in EARS:
        ex, ey = cx + EAR_R * math.cos(math.radians(a)), cy + EAR_R * math.sin(math.radians(a))
        m += cyl(5.5, 79.0, Z_DECK + 0.5, ex, ey)
        m -= cyl(1.3, 78.0, 88.0, ex, ey)
    m -= cyl(36.0, 78.0, Z_TOP + 1, cx, cy)
    # гнездо третьего динамика и горло трубы
    hx, hy = H
    rs = P["spk_d"] / 2 + 0.3
    m += cyl(rs + 2.5, 77.0, Z_DECK + 0.5, hx, hy) ^ ext(o, 70.0, Z_TOP)
    for a in HORN_EARS:
        ex, ey = hx + 26.5 * math.cos(math.radians(a)), hy + 26.5 * math.sin(math.radians(a))
        m += cyl(4.5, 77.0, Z_DECK + 0.5, ex, ey)
        m -= cyl(1.3, 76.0, 87.0, ex, ey)
    m -= cyl(rs, 76.0, 83.2, hx, hy)
    m -= cyl(rs - 1.3, 83.15, Z_DECK + 0.01, hx, hy, r2=SPIG_R + P["fit"])
    m -= cyl(SPIG_R + P["fit"], Z_DECK, Z_TOP + 1, hx, hy)
    # боковые динамики
    hole = teardrop(GR_R + 0.3).translate([0, SPK_Z])
    m -= prism_x(hole, HB - P["wall"] - 1, HB + 1)
    m -= prism_x(hole, -HB - 1, -HB + P["wall"] + 1)
    # лицо: декоративные прорези и углубление под шильдик
    vents = [lens(7, 15, -x * 1.2).translate([x, 66 - abs(x) * 0.12]) for x in (-24, -12, 0, 12, 24)]
    for v in vents:
        m -= prism_y(v, -HB - 1, -HB + P["wall"] + 1)
    m -= prism_y(rrect(30.2, 13.2, 3).translate([0, 37]), -HB - 1, -HB + 0.6)
    # задняя стенка: USB-C, AUX, microSD, ручка громкости (u = −x)
    m -= prism_y(rrect(7.0, 4.5, 2).translate([-ESP_X, USB_Z]), HB - P["wall"] - 1, HB + 1)
    m -= prism_y(CrossSection.circle(3.5).translate([-DAC_X, AUX_Z]), HB - P["wall"] - 1, HB + 1)
    m -= prism_y(rrect(8.0, 1.75, 1).translate([-SD_X, SD_Z]), HB - P["wall"] - 1, HB + 1)
    m -= prism_y(CrossSection.circle(3.6).translate([-ENC_X, ENC_Z]), HB - P["wall"] - 1, HB + 1)
    return m

def make_base():
    """Основание: поддон с карнизом, стойками под винты и ложементами всех плат."""
    o = outline()
    inner = o.offset(-P["wall"], JoinType.Miter)
    pocket = inner.offset(-2.3, JoinType.Miter)
    m = tiers([(8.0, HB + 8.4), (8.6, HB + 9), (11.0, HB + 9), (12.0, HB + 7), (14.5, HB + 7), (15.5, HB + 4.5), (Z_PL, HB + 4.5)], 11.0)
    m -= ext(pocket, Z_FLOOR, Z_PL + 1)
    lip = ext(inner.offset(-P["fit"], JoinType.Miter) - inner.offset(-P["fit"] - 2.0, JoinType.Miter), Z_PL - 0.5, Z_PL + 3)
    for x, y in BOSS:
        lip -= cyl(5.2, Z_PL - 1, Z_PL + 4, x, y)
    m += lip
    y1 = WIN - 0.5                                   # задний торец плат у стенки
    # --- ESP32: плата лежит деталями вниз, штырями вверх, USB — в заднюю стенку
    ew, el = P["esp"]
    y0, zl = y1 - el - 0.3, USB_Z + 1.6              # передний торец платы и опорная полка
    xa, xb = ESP_X - ew / 2 - 0.2, ESP_X + ew / 2 + 0.2
    fr = box(xa - 2.2, y0 - 8, Z_FLOOR - 0.1, xb + 2.2, y1, zl + 4.6)
    fr -= box(xa, y0, zl, xb, y1 + 1, zl + 6)                                  # карман платы
    fr -= box(xa + 1.5, y0 + 1.5, Z_FLOOR, xb - 1.5, y1 + 1, zl + 6)           # место под детали платы
    fr -= box(xa - 3, y0 - 9, zl + 1.6, xb + 3, y0 + 4.5, zl + 6)              # площадка под прижим
    fr -= box(ESP_X - 9.6, y0 - 0.1, zl - 4, ESP_X + 9.6, y0 + 2, zl + 1)            # антенна модуля не лежит на полке
    fr -= box(ESP_X - 7.5, y1 - 6, Z_PL + 0.5, ESP_X + 7.5, y1 + 1, zl + 6)    # окно USB
    for sx, xw in ((1, xa), (-1, xb)):                                         # зацепы у USB-конца
        w = poly([(-1, zl + 1.8), (0, zl + 1.8), (2.8, zl + 4.6), (-1, zl + 4.6)] if sx > 0 else [(1, zl + 1.8), (1, zl + 4.6), (-2.8, zl + 4.6), (0, zl + 1.8)])
        fr += xf(w.extrude(5.5), [(1, 0, 0), (0, 0, 1), (0, -1, 0)], (xw, y1 - 0.5, 0))
    for dx in (-10, 10):
        fr -= cyl(1.3, zl - 9, zl + 2, ESP_X + dx, y0 - 4)
    m += fr
    # --- ЦАП (AUX) и microSD: полка с бортиками, упор и площадка под прижим на дальнем торце
    for (bw, bl), bx, ztop in ((P["dac"], DAC_X, AUX_Z - 4.1), (P["sd"], SD_X, SD_Z - 2.6)):
        ya = y1 - bl
        sh = box(bx - bw / 2 - 1.4, ya - 6, Z_FLOOR - 0.1, bx + bw / 2 + 1.4, y1 + 0.4, ztop + 1.6)
        sh -= box(bx - bw / 2 - 0.2, ya, ztop, bx + bw / 2 + 0.2, y1 + 1, ztop + 3)
        for dx in (-bw / 2 + 4, bw / 2 - 4):
            sh -= cyl(1.3, ztop - 9, ztop + 2, bx + dx, ya - 3)
        m += sh
    # --- три усилителя MAX98357A: отсек с перегородками у передней стенки
    aw, al = P["amp"]
    wtot = 3 * (aw + 0.4) + 4 * 1.6
    bay = box(-wtot / 2 - 7, AMP_Y - 1.6, Z_FLOOR - 0.1, wtot / 2 + 7, AMP_Y + al + 2.0, Z_FLOOR + 2.0)
    for k in range(3):
        xk = -wtot / 2 + 1.6 + k * (aw + 2.0)
        bay -= box(xk, AMP_Y, Z_FLOOR, xk + aw + 0.4, AMP_Y + al + 0.4, Z_FLOOR + 3)
    m += bay
    for sx in (-1, 1):
        m -= cyl(1.3, 7.0, Z_FLOOR + 3, sx * (wtot / 2 + 3.5), AMP_Y + al / 2)
    # --- стойки и отверстия под винты M3 корпуса
    for x, y in BOSS:
        m += cyl(5.5, 8.0, Z_PL, x, y) ^ ext(inner, 8.0, Z_PL)
        m -= cyl(1.7, 7.0, Z_PL + 4, x, y)
        m -= cyl(3.3, 7.0, 13.0, x, y)
    return m

def end_clamp(bw):
    """Прижимная планка торца платы. Локально: лежит на z = 0."""
    m = box(-bw / 2 - 1.4, 0, 0, bw / 2 + 1.4, 9.0, 2.4)
    for dx in (-bw / 2 + 4, bw / 2 - 4):
        m -= cyl(1.7, -1, 3, dx, 3.0)
    return m

def make_esp_clamp():
    ew = P["esp"][0]
    m = box(-ew / 2 - 2.4, 0, 0, ew / 2 + 2.4, 12.0, 2.4)
    for dx in (-10, 10):
        m -= cyl(1.7, -1, 3, dx, 4.0)
    return m

def make_amp_clamp():
    """Рамка-прижим трёх усилителей: окна над платами, края рамки держат платы за боковые кромки."""
    aw, al = P["amp"]
    wtot = 3 * (aw + 0.4) + 4 * 1.6
    m = box(-wtot / 2 - 7, -1.6, 0, wtot / 2 + 7, al + 2.0, 2.0)
    for k in range(3):
        xk = -wtot / 2 + 1.6 + k * (aw + 2.0)
        m -= box(xk + 1.2, 0.0, -1, xk + aw + 0.4 - 1.2, al + 0.4, 3)
    for sx in (-1, 1):
        m -= cyl(1.7, -1, 3, sx * (wtot / 2 + 3.5), al / 2)
    return m

def make_spk_ring():
    """Кольцо-прижим бокового динамика: тремя винтами стягивает динамик, решётку и стенку. Локально: к стенке — z = 0."""
    m = cyl(28.5, 0.0, 2.4) - cyl(17.6, -1, 3)
    m -= cyl(P["spk_d"] / 2 + 0.4, -1, P["spk_rim"] - GR_T - P["wall"] + 2.4 + 0.2)
    for a in (90, 210, 330):
        m -= cyl(1.7, -1, 3, SCR_R * math.cos(math.radians(a)), SCR_R * math.sin(math.radians(a)))
    return m

def make_horn_ring():
    """Прижим динамика трубы снизу гнезда под декой."""
    rs = P["spk_d"] / 2
    m = cyl(rs + 2.5, 74.6, 77.0) + cyl(rs + 0.1, 77.0, 83.15 - P["spk_rim"] - 0.05)
    for a in HORN_EARS:
        ex, ey = 26.5 * math.cos(math.radians(a)), 26.5 * math.sin(math.radians(a))
        m += cyl(4.5, 74.6, 77.0, ex, ey)
    m -= cyl(rs - 2.4, 73.0, 84.0)
    m ^= ext(outline().offset(-P["wall"] - 0.3, JoinType.Miter).translate([-H[0], -H[1]]), 70.0, 90.0)
    for a in HORN_EARS:
        m -= cyl(1.7, 73.0, 78.0, 26.5 * math.cos(math.radians(a)), 26.5 * math.sin(math.radians(a)))
    return m

def baluster_profile():
    return [(0, 6.3), (3, 6.3), (4.3, 5.0), (6, 5.0), (7, 6.0), (9, 6.0), (10, 5.0), (12, 4.2), (20, 5.8),
            (30, 6.2), (40, 5.6), (50, 4.4), (52, 4.2), (53, 5.0), (55, 5.0), (56, 6.0), (57.5, 6.0),
            (58.5, 5.0), (60.2, 5.0), (61.5, 6.3), (63.7, 6.3)]

def make_column():
    pr = baluster_profile()
    pts = [(0.0, 0.0)] + [(r, z) for z, r in pr] + [(0.0, pr[-1][0])]
    return poly(pts).revolve(48)

def make_foot():
    return cyl(6.5, 0.0, 8.0, r2=9.0, seg=48)

def cup_holes(m, z0, z1):
    for a in EARS:
        m -= cyl(1.7, z0, z1, EAR_R * math.cos(math.radians(a)), EAR_R * math.sin(math.radians(a)))
    return m

def make_cup():
    """Световой стакан: держит PN532 под пластинкой и кольцо WS2812 под световым каналом."""
    pl, pw, ph = P["pn532"]
    ro = P["ring_od"] / 2
    m = cyl(36.0 - P["fit"], 79.0, Z_TOP) + cyl(39.0, 77.0, 79.0)
    for a in EARS:
        m += cyl(5.5, 77.0, 79.0, EAR_R * math.cos(math.radians(a)), EAR_R * math.sin(math.radians(a)))
    m = cup_holes(m, 76.0, 80.0)
    cav = poly([(ro - 7.5, 76.9), (ro + 0.6, 76.9), (ro + 0.6, 81.0), (ro - 0.6, 82.2), (ro - 0.6, 85.0),
                (33.5, 86.1), (33.5, 93.0), (30.5, 93.0), (30.5, 87.8), (ro - 7.5, 82.8)]).revolve(96)
    m -= cav
    for k in range(6):
        m += box(29.0, -0.6, 84.4, 34.5, 0.6, 89.6).rotate([0, 0, 15 + 60 * k])
    m -= box(-pl / 2 - 0.5, -pw / 2 - 0.5, Z_TOP - ph, pl / 2 + 0.5, pw / 2 + 0.5, Z_TOP + 1)
    m -= box(-12, -pw / 2 + 2, 76.0, 12, -pw / 2 + 7, Z_TOP - ph + 0.5)       # окно для проводов PN532
    return m

def make_retainer():
    """Прижим кольца WS2812 снизу стакана."""
    ro = P["ring_od"] / 2
    m = cyl(39.0, 74.6, 77.0)
    for a in EARS:
        m += cyl(5.5, 74.6, 77.0, EAR_R * math.cos(math.radians(a)), EAR_R * math.sin(math.radians(a)))
    m += cyl(ro + 0.3, 77.0, 79.4) - cyl(ro - 1.0, 76.0, 80.0)
    m = cup_holes(m, 73.0, 78.0)
    m -= cyl(15.0, 73.0, 78.0)
    m -= box(-12, -P["pn532"][1] / 2 + 2, 73.0, 12, -P["pn532"][1] / 2 + 7, 78.0)
    m -= box(-4, ro - 3, 73.0, 4, ro + 1, 80.0)                               # вывод проводов кольца
    return m

def make_hub():
    """Центр платтера с гнездом пластинки."""
    rd = P["disc_d"] / 2 + 0.3
    zt = Z_TOP + PLAT_T
    m = cyl(30.25, Z_TOP, zt)
    m -= cyl(rd, zt - P["disc_t"] - 0.2, zt + 1)
    m -= (cyl(8.0, zt - 2.6, zt + 1, 0, -23.0) ^ cyl(29.2, Z_TOP, zt + 1))    # вырез под палец
    return m

def make_diffuser():
    """Рассеиватель светового кольца (белый)."""
    zt = Z_TOP + PLAT_T
    return (cyl(33.9, Z_TOP, zt) - cyl(30.4, Z_TOP - 1, zt + 1)) - (cyl(33.3, Z_TOP - 1, zt - 0.8) - cyl(31.0, Z_TOP - 2, zt))

def make_platter():
    """Внешнее кольцо платтера с «канавками» винила."""
    zt = Z_TOP + PLAT_T
    m = cyl(PLAT_R, Z_TOP, zt - 1) + cyl(PLAT_R, zt - 1, zt, r2=PLAT_R - 1)
    m -= cyl(34.05, Z_TOP - 1, zt + 1)
    for r in np.arange(36.6, PLAT_R - 2.0, 1.6):
        m -= cyl(r + 0.35, zt - 0.3, zt + 1) - cyl(r, zt - 1, zt + 2)
    return m

def make_grille():
    """Решётка бокового динамика с гнездом. Локально: лицо на z = 0, труба вверх (сквозь стенку)."""
    rs = P["spk_d"] / 2
    m = cyl(29.6, 0.0, 0.8, r2=31.0) + cyl(31.0, 0.8, 2.4) + cyl(GR_R, 2.4, 2.4 + GR_T)
    m -= cyl(rs + 0.3, 2.4, 10.0)
    m -= cyl(rs - 1.5, -1.0, 3.0)
    blades = []
    for k in range(9):
        a0 = 2 * math.pi * k / 9
        pts_l, pts_r = [], []
        for r in np.linspace(3.0, rs - 1.0, 14):
            a = a0 + 0.055 * r
            n = np.array([-math.sin(a), math.cos(a)]); c = np.array([r * math.cos(a), r * math.sin(a)])
            pts_l.append(c + n * 1.1); pts_r.append(c - n * 1.1)
        blades.append(poly(pts_r + pts_l[::-1]))
    sw = CrossSection.batch_boolean(blades, OpType.Add) + CrossSection.circle(4.5)
    m += ext(sw, 0.0, 2.4)
    for a in (90, 210, 330):
        m -= cyl(1.3, 0.6, 6.0, SCR_R * math.cos(math.radians(a)), SCR_R * math.sin(math.radians(a)))
    return m

def make_plate():
    """Декоративный шильдик на лицо. Локально: лежит на z = 0, лицом вверх."""
    m = ext(rrect(30.0, 13.0, 3), 0.0, 1.6)
    m += ext(rrect(30.0, 13.0, 3) - rrect(28.4, 11.4, 2), 1.6, 2.4)
    m += ext(rrect(10.0, 4.2, 1.5).translate([0, 5.6]), 1.6, 2.6)
    m -= ext(rrect(8.6, 2.8, 1).translate([0, 5.6]), 2.0, 3.0)
    for sx in (-1, 1):
        for y in (5.8, -0.2, -6.2):
            m += ext(rrect(4.2, 1.9, 0.8).translate([sx * 20.5, y]), 1.6, 2.4)
    m -= ext(rrect(4.2, 1.5, 0.6).translate([0, -4.2]), 1.0, 3.0)
    for sx in (-1, 1):
        for sy in (-1, 1):
            m += cyl(1.2, 1.6, 2.6, sx * 26.6, sy * 9.6, seg=24)
    return m

def make_riser():
    """Стойка трубы: шип в деку, фланец, труба, шип под колено."""
    z0 = Z_TOP
    m = cyl(SPIG_R, Z_DECK + 0.2, z0 + 0.1) + cyl(18.0, z0, z0 + 2) + cyl(18.0, z0 + 2, z0 + 5, r2=TUBE_R)
    m += cyl(TUBE_R, z0 + 5, z0 + RISER_L - 2) + cyl(TUBE_R, z0 + RISER_L - 2, z0 + RISER_L, r2=SPIG_R)
    m += cyl(SPIG_R, z0 + RISER_L, z0 + RISER_L + SPIG_L)
    return (m - cyl(BORE_R, Z_DECK - 1, z0 + RISER_L + SPIG_L + 1)).translate([H[0], H[1], 0])

def horn_frames():
    """Опорные точки трубы: низ колена, конец изгиба, ось раструба."""
    p0 = np.array([H[0], H[1], Z_TOP + RISER_L + 10.0])
    a = math.radians(BEND_A)
    e = p0 + BEND_R * (1 - math.cos(a)) * D_DIR + BEND_R * math.sin(a) * np.array([0, 0, 1.0])
    ax = math.sin(a) * D_DIR + math.cos(a) * np.array([0, 0, 1.0])
    u = np.cross(ax, np.array([0, 0, 1.0])); u /= np.linalg.norm(u)
    return p0, e, ax, u, np.cross(ax, u)

def make_elbow():
    """Колено трубы: гнездо снизу, изгиб 45°, шип под раструб."""
    p0, e, ax, u, v = horn_frames()
    z0 = Z_TOP + RISER_L
    m = cyl(TUBE_R, z0, z0 + 10.0, H[0], H[1])
    ann = CrossSection.circle(TUBE_R).translate([BEND_R, 0])
    ex_, ey_ = -D_DIR, np.array([0, 0, 1.0])
    ez_ = np.cross(ex_, ey_)
    m += xf(ann.revolve(40, BEND_A), [ex_, ey_, ez_], p0 - BEND_R * ex_)
    bore = xf(CrossSection.circle(BORE_R).translate([BEND_R, 0]).revolve(40, BEND_A), [ex_, ey_, ez_], p0 - BEND_R * ex_)
    sp = xf(cyl(SPIG_R, -0.5, SPIG_L), [u, -v, ax], e)
    spb = xf(cyl(BORE_R, -1.0, SPIG_L + 1), [u, -v, ax], e)
    m = (m + sp) - bore - spb
    m -= cyl(BORE_R, z0 - 1, z0 + 10.5, H[0], H[1])
    m -= cyl(SPIG_R + P["fit"], z0 - 1, z0 + SPIG_L + 0.3, H[0], H[1])
    m -= cyl(SPIG_R + P["fit"], z0 + SPIG_L + 0.29, z0 + SPIG_L + 2.6, H[0], H[1], r2=BORE_R)
    return m

def bell_local():
    """Раструб в локальных координатах: горло на z = 0, устье на z = BELL_L. 8 лепестков."""
    n, kexp, neck = 192, 1.3, 12.0
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    petal = np.abs(np.sin(4 * th)) ** 0.7
    def ring(z, inner):
        t = max(0.0, (z - neck) / (BELL_L - neck))
        r = TUBE_R + (BELL_R - TUBE_R) * (math.exp(kexp * t) - 1) / (math.exp(kexp) - 1)
        r = r * (1 + 0.09 * t * t * petal)
        if inner:
            r = r - (1.6 + (TUBE_R - BORE_R - 1.6) * max(0.0, 1 - 5 * t))
        return np.column_stack([r * np.cos(th), r * np.sin(th), np.full(n, z)])
    zs = np.concatenate([[0.0, neck], np.linspace(neck, BELL_L, 60)[1:]])
    outer = loft([ring(z, False) for z in zs])
    zi = np.concatenate([[-1.0], zs[1:], [BELL_L + 1.0]])
    inner = loft([ring(min(max(z, 0.0), BELL_L), True) + np.array([0, 0, z - min(max(z, 0.0), BELL_L)]) for z in zi])
    return outer - inner - cyl(SPIG_R + P["fit"], -1.0, SPIG_L + 0.3)

def make_bell():
    p0, e, ax, u, v = horn_frames()
    return xf(bell_local(), [u, -v, ax], e)

def make_arm():
    """Тонарм: хомут на стойке трубы, короткая штанга и головка над краем диска."""
    z0, z1 = Z_TOP + 5.2, Z_TOP + 13.2
    d = (C - H) / np.linalg.norm(C - H)
    head = C - d * 41.0
    n = np.array([-d[1], d[0]])
    s0, s1 = H + d * 16.5, head
    beam = poly([s0 + n * 3.5, s0 - n * 3.5, s1 - n * 3.5, s1 + n * 3.5])
    m = ext(beam, z0, z1) + cyl(18.2, z0, z1, H[0], H[1]) + cyl(8.0, z0, z1, head[0], head[1])
    m += cyl(6.5, z1, z1 + 1.0, head[0], head[1]) + cyl(3.0, z1 + 1.0, z1 + 2.2, head[0], head[1])
    return m - cyl(TUBE_R + 0.2, z0 - 1, z1 + 1, H[0], H[1])

def make_knob():
    """Ручка громкости на вал энкодера EC11 (Ø6 с лыской). Локально: лицо на z = 0."""
    m = cyl(8.0, 0.0, 1.0, r2=9.0) + cyl(9.0, 1.0, 13.0)
    for k in range(18):
        a = 2 * math.pi * k / 18
        m -= cyl(0.9, 3.0, 14.0, 9.3 * math.cos(a), 9.3 * math.sin(a), seg=12)
    bore = cyl(3.1, 3.0, 14.0, seg=36) - box(1.6, -4, 2.0, 4, 4, 15.0)
    return m - bore


# ------------------------------------------------------------------ макеты покупных деталей (для проверки места)
def speaker_ghost():
    """Динамик: лицо на z = 0, корпус в +z."""
    rs = P["spk_d"] / 2
    return (cyl(rs, 0, P["spk_rim"]) + cyl(rs - 2.5, P["spk_rim"], P["spk_depth"] - 7, r2=P["spk_magnet"] / 2)
            + cyl(P["spk_magnet"] / 2, P["spk_depth"] - 7, P["spk_depth"]))

def ghosts():
    """→ [(ключ, название, тело в сборке, вектор разнесения)]"""
    g = []
    cx, cy = C; hx, hy = H; y1 = WIN - 0.5
    pl, pw, ph = P["pn532"]
    g.append(("e_pn532", "PN532", box(cx - pl / 2, cy - pw / 2, Z_TOP - ph + 0.2, cx + pl / 2, cy + pw / 2, Z_TOP - 0.1), (0, 0, 70)))
    ring = cyl(P["ring_od"] / 2, 79.4, 79.4 + P["ring_pcb"]) + cyl(P["ring_od"] / 2 - 1, 81.0, 82.6)
    g.append(("e_ring", "Кольцо WS2812", (ring - cyl(P["ring_id"] / 2, 78, 84)).translate([cx, cy, 0]), (0, 0, -32)))
    sp = speaker_ghost()
    g.append(("e_spk_r", "Динамик правый", xf(sp, [(0, -1, 0), (0, 0, 1), (-1, 0, 0)], (HB, 0, SPK_Z)), (28, 0, 0)))
    g.append(("e_spk_l", "Динамик левый", xf(sp, [(0, 1, 0), (0, 0, 1), (1, 0, 0)], (-HB, 0, SPK_Z)), (-28, 0, 0)))
    g.append(("e_spk_h", "Динамик трубы", xf(sp, [(1, 0, 0), (0, -1, 0), (0, 0, -1)], (hx, hy, 83.1)), (0, 0, -20)))
    ew, el = P["esp"]; zl = USB_Z + 1.6; y0 = y1 - el
    esp = box(ESP_X - ew / 2, y0, zl, ESP_X + ew / 2, y1, zl + 1.6)
    esp += box(ESP_X - 9, y0 + 1, zl - 3.4, ESP_X + 9, y0 + 27, zl)                    # модуль
    esp += box(ESP_X - 4.5, y1 - 7.5, zl - 3.3, ESP_X + 4.5, y1, zl)                   # USB-C
    for sx in (-1, 1):                                                               # штыри + разъёмы проводов
        esp += box(ESP_X + sx * (ew / 2 - 1.3) - 1.3, y0 + 7, zl + 1.6, ESP_X + sx * (ew / 2 - 1.3) + 1.3, y1 - 7, zl + 24)
    g.append(("e_esp", "ESP32", esp, (0, 0, -70)))
    aw, al = P["amp"]; wtot = 3 * (aw + 0.4) + 4 * 1.6
    for k in range(3):
        xk = -wtot / 2 + 1.6 + k * (aw + 2.0) + 0.2
        a = box(xk, AMP_Y + 0.2, Z_FLOOR + 0.3, xk + aw, AMP_Y + 0.2 + al, Z_FLOOR + 1.9)
        a += box(xk + 4, AMP_Y + 0.2, Z_FLOOR + 1.9, xk + aw - 4, AMP_Y + 8, Z_FLOOR + 11.5)   # клеммник
        g.append((f"e_amp{k+1}", "Усилитель MAX98357A", a, (0, 0, -70)))
    (dw, dl), (sw, sl) = P["dac"], P["sd"]
    zt = AUX_Z - 4.1
    dac = box(DAC_X - dw / 2, y1 - dl, zt, DAC_X + dw / 2, y1, zt + 1.6) + box(DAC_X - 3.2, y1 - 12, zt + 1.6, DAC_X + 3.2, y1, zt + 6.6) + xf(cyl(3.0, 0, 3.0), [(-1, 0, 0), (0, 0, 1), (0, 1, 0)], (DAC_X, y1, AUX_Z))
    g.append(("e_dac", "ЦАП PCM5102A", dac, (0, 0, -70)))
    zt = SD_Z - 2.6
    sd = box(SD_X - sw / 2, y1 - sl, zt, SD_X + sw / 2, y1, zt + 1.6) + box(SD_X - 7.5, y1 - 16, zt + 1.6, SD_X + 7.5, y1, zt + 3.6)
    g.append(("e_sd", "Модуль microSD", sd, (0, 0, -70)))
    bw, bh, bd = P["enc"]
    enc = box(ENC_X - bw / 2, WIN - bd, ENC_Z - bh / 2, ENC_X + bw / 2, WIN, ENC_Z + bh / 2)
    enc += xf(cyl(3.0, 0, 6.0), [(-1, 0, 0), (0, 0, 1), (0, 1, 0)], (ENC_X, WIN, ENC_Z))
    g.append(("e_enc", "Энкодер EC11", enc, (0, -28, 0)))
    return g


# ------------------------------------------------------------------ сборка
COL = dict(brown="#5c2019", gold="#c9a24a", black="#1c1c1e", white="#f2efe6", gray="#8d8f94", pcb="#2f7d4f")

def build():
    """→ список деталей: имя, название, цвет, тело в сборке, вектор разнесения, поворот для печати."""
    parts = []
    def add(key, title, color, m, explode, print_rot=(0, 0, 0), count_tag=None):
        parts.append(dict(key=key, title=title, color=color, m=m, explode=explode, rot=print_rot, src=count_tag or key))
    cx, cy = C
    add("body", "Корпус (стенки + дека)", "brown", make_body(), (0, 0, 0), (180, 0, 0))
    add("base", "Основание", "brown", make_base(), (0, 0, -70))
    col, foot = make_column(), make_foot()
    for i, (sx, sy) in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
        add(f"column{i+1}", "Балясина", "brown", col.translate([sx * COLP, sy * COLP, Z_PL + 0.15]),
            (sx * 45, sy * 45, 0), (0, 0, 0), "column")
        add(f"foot{i+1}", "Ножка", "brown", foot.translate([sx * (HB - 6), sy * (HB - 6), 0]), (0, 0, -100), (180, 0, 0), "foot")
    add("cup", "Световой стакан", "white", make_cup().translate([cx, cy, 0]), (0, 0, -22))
    add("retainer", "Прижим кольца", "white", make_retainer().translate([cx, cy, 0]), (0, 0, -42))
    add("hub", "Гнездо пластинки", "black", make_hub().translate([cx, cy, 0]), (0, 0, 55))
    add("diffuser", "Рассеиватель", "white", make_diffuser().translate([cx, cy, 0]), (0, 0, 40), (180, 0, 0))
    add("platter", "Диск-платтер", "black", make_platter().translate([cx, cy, 0]), (0, 0, 25))
    g = make_grille()
    add("grille_r", "Решётка динамика", "gold", xf(g, [(0, -1, 0), (0, 0, 1), (-1, 0, 0)], (HB + 2.4, 0, SPK_Z)),
        (55, 0, 0), None, "grille")
    add("grille_l", "Решётка динамика", "gold", xf(g, [(0, 1, 0), (0, 0, 1), (1, 0, 0)], (-HB - 2.4, 0, SPK_Z)),
        (-55, 0, 0), None, "grille")
    add("plate", "Шильдик", "gold", xf(make_plate(), [(1, 0, 0), (0, 0, 1), (0, -1, 0)], (0, -HB + 0.6, 37)),
        (0, -45, 0), None, "plate")
    add("knob", "Ручка громкости", "gold", xf(make_knob(), [(-1, 0, 0), (0, 0, 1), (0, -1, 0)], (ENC_X, HB + 17.0, ENC_Z)),
        (0, 40, 0), None, "knob")
    # крепёж электроники
    sr = make_spk_ring()
    add("spk_ring_r", "Прижим бокового динамика", "gray", xf(sr, [(0, -1, 0), (0, 0, 1), (-1, 0, 0)], (WIN, 0, SPK_Z)), (-30, 0, 0), None, "spk_ring")
    add("spk_ring_l", "Прижим бокового динамика", "gray", xf(sr, [(0, 1, 0), (0, 0, 1), (1, 0, 0)], (-WIN, 0, SPK_Z)), (30, 0, 0), None, "spk_ring")
    add("horn_ring", "Прижим динамика трубы", "gray", make_horn_ring().translate([H[0], H[1], 0]), (0, 0, -34))
    y1 = WIN - 0.5; zl = USB_Z + 1.6
    add("esp_clamp", "Прижим ESP32", "gray", make_esp_clamp().translate([ESP_X, y1 - P["esp"][1] - 0.3 - 8, zl + 1.6]), (0, 0, -50), None, "esp_clamp")
    aw, al = P["amp"]
    add("amp_clamp", "Рамка усилителей", "gray", make_amp_clamp().translate([0, AMP_Y, Z_FLOOR + 2.0]), (0, 0, -50), None, "amp_clamp")
    add("dac_clamp", "Прижим ЦАП", "gray", end_clamp(P["dac"][0]).translate([DAC_X, y1 - P["dac"][1] - 6, AUX_Z - 4.1 + 1.6]), (0, 0, -50), None, "dac_clamp")
    add("sd_clamp", "Прижим microSD", "gray", end_clamp(P["sd"][0]).translate([SD_X, y1 - P["sd"][1] - 6, SD_Z - 2.6 + 1.6]), (0, 0, -50), None, "sd_clamp")
    add("riser", "Стойка трубы", "gold", make_riser(), (0, 0, 45), (180, 0, 0))
    add("arm", "Тонарм", "gold", make_arm(), (0, 0, 78))
    add("elbow", "Колено трубы", "gold", make_elbow(), (0, 0, 105))
    add("bell", "Раструб", "gold", make_bell(), (-45, -45, 175), None, "bell")
    return parts

def build_height():
    return max(p["m"].bounding_box()[5] for p in build())

LOCAL = dict(grille=make_grille, plate=make_plate, knob=make_knob, column=make_column, foot=make_foot,
             spk_ring=lambda: make_spk_ring().rotate([180, 0, 0]), esp_clamp=make_esp_clamp, amp_clamp=make_amp_clamp,
             dac_clamp=lambda: end_clamp(P["dac"][0]), sd_clamp=lambda: end_clamp(P["sd"][0]))

def print_mesh(part):
    """Тело в положении печати, опущенное на стол, центр по XY в нуле."""
    if part["src"] == "bell":
        m = bell_local().rotate([180, 0, 0])
    elif part["src"] in LOCAL and part["rot"] in (None, (0, 0, 0)):
        m = LOCAL[part["src"]]()
    elif part["src"] == "foot":
        m = make_foot().rotate([180, 0, 0])
    else:
        m = part["m"].rotate(list(part["rot"]))
    x0, y0, z0, x1, y1, z1 = m.bounding_box()
    return m.translate([-(x0 + x1) / 2, -(y0 + y1) / 2, -z0])


# ------------------------------------------------------------------ вывод
def tris_of(m):
    me = m.to_mesh()
    v = np.asarray(me.vert_properties, dtype=np.float64)[:, :3]
    f = np.asarray(me.tri_verts, dtype=np.int64)
    return v, f

def write_stl(path, m):
    v, f = tris_of(m)
    t = v[f].astype(np.float32)
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]); ln = np.linalg.norm(n, axis=1, keepdims=True); ln[ln == 0] = 1
    rec = np.zeros(len(t), dtype=[("n", "<f4", 3), ("v", "<f4", 9), ("a", "<u2")])
    rec["n"] = n / ln; rec["v"] = t.reshape(-1, 9)
    with open(path, "wb") as fh:
        fh.write(b"Mini-Vinyl gramophone".ljust(80, b"\0")); fh.write(struct.pack("<I", len(t))); fh.write(rec.tobytes())

def write_3mf(path, objects):
    """objects: [(name, manifold, (tx, ty))] — каждое тело отдельным объектом на столе."""
    res, build, ids = [], [], {}
    for name, m, (tx, ty) in objects:
        if id(m) not in ids:
            i = ids[id(m)] = len(ids) + 1
            v, f = tris_of(m)
            vs = "".join(f'<vertex x="{a:.4f}" y="{b:.4f}" z="{c:.4f}"/>' for a, b, c in v)
            fs = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in f)
            res.append(f'<object id="{i}" name="{name}" type="model"><mesh><vertices>{vs}</vertices>'
                       f'<triangles>{fs}</triangles></mesh></object>')
        build.append(f'<item objectid="{ids[id(m)]}" transform="1 0 0 0 1 0 0 0 1 {tx:.3f} {ty:.3f} 0"/>')
    model = ('<?xml version="1.0" encoding="UTF-8"?><model unit="millimeter" xml:lang="en-US" '
             'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
             f'<resources>{"".join(res)}</resources><build>{"".join(build)}</build></model>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
        z.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
        z.writestr("3D/3dmodel.model", model)

def pack(items, gap=5.0, margin=6.0):
    """Раскладка «полками»: items = [(name, manifold)] → [(name, manifold, (x, y))] в пределах стола."""
    out, x, y, row = [], margin, margin, 0.0
    for name, m in sorted(items, key=lambda it: -(it[1].bounding_box()[4] - it[1].bounding_box()[1])):
        bb = m.bounding_box(); w, d = bb[3] - bb[0], bb[4] - bb[1]
        if x + w > BED - margin:
            x, y, row = margin, y + row + gap, 0.0
        if y + d > BED - margin:
            raise SystemExit(f"не помещается на стол: {name}")
        out.append((name, m, (x + w / 2, y + d / 2)))
        x += w + gap; row = max(row, d)
    return out

PLATES = [
    ("1_korpus_brown", ["body"]),
    ("2_osnovanie_brown", ["base", "column", "column", "column", "column", "foot", "foot", "foot", "foot"]),
    ("3_rastrub_gold", ["bell", "riser", "knob"]),
    ("4_dekor_gold", ["elbow", "arm", "grille", "grille", "plate"]),
    ("5_disk_black", ["platter", "hub", "spk_ring", "spk_ring", "horn_ring", "esp_clamp", "amp_clamp", "dac_clamp", "sd_clamp"]),
    ("6_svet_white", ["cup", "retainer", "diffuser"]),
]

def main():
    os.makedirs(os.path.join(OUT, "stl"), exist_ok=True); os.makedirs(os.path.join(OUT, "3mf"), exist_ok=True)
    parts = build()
    for p in parts:
        if p["m"].is_empty() or p["m"].volume() <= 0:
            raise SystemExit(f"пустая деталь: {p['key']}")
        if len(p["m"].decompose()) != 1:
            raise SystemExit(f"деталь распадается на куски: {p['key']} ({len(p['m'].decompose())})")
    # проверка пересечений в сборке
    bad = []
    for i, a in enumerate(parts):
        for b in parts[i + 1:]:
            vol = (a["m"] ^ b["m"]).volume()
            if vol > 0.5:
                bad.append((a["key"], b["key"], round(vol, 1)))
    print("пересечения деталей (мм³):", bad or "нет")
    gh = ghosts(); bad = []
    for i, (k, t, m, e) in enumerate(gh):
        for p in parts:
            vol = (m ^ p["m"]).volume()
            if vol > 0.5:
                bad.append((k, p["key"], round(vol, 1)))
        for k2, t2, m2, e2 in gh[i + 1:]:
            vol = (m ^ m2).volume()
            if vol > 0.5:
                bad.append((k, k2, round(vol, 1)))
    print("электроника упирается (мм³):", bad or "нет")
    # печать: STL и столы
    pm, summary = {}, []
    for p in parts:
        if p["src"] not in pm:
            pm[p["src"]] = print_mesh(p)
            write_stl(os.path.join(OUT, "stl", p["src"] + ".stl"), pm[p["src"]])
            bb = pm[p["src"]].bounding_box()
            summary.append((p["src"], p["title"], p["color"], sum(q["src"] == p["src"] for q in parts),
                            round(bb[3] - bb[0], 1), round(bb[4] - bb[1], 1), round(bb[5], 1), round(pm[p["src"]].volume() / 1000, 1)))
    for name, keys in PLATES:
        write_3mf(os.path.join(OUT, "3mf", f"plate_{name}.3mf"), pack([(k, pm[k]) for k in keys]))
    for s in summary:
        print("%-9s %-24s %-6s ×%d  %6.1f × %6.1f × %5.1f мм  %6.1f см³" % s)
    # превью
    prev = dict(parts=[], colors=COL)
    for p in parts:
        v, f = tris_of(p["m"])
        prev["parts"].append(dict(key=p["key"], title=p["title"], color=p["color"], explode=list(p["explode"]),
                                  v=base64.b64encode(v.astype(np.float32).tobytes()).decode(),
                                  f=base64.b64encode(f.astype(np.uint32).tobytes()).decode()))
    for k, t, m, e in gh:
        v, f = tris_of(m)
        prev["parts"].append(dict(key=k, title=t, color="pcb", explode=list(e), ghost=True,
                                  v=base64.b64encode(v.astype(np.float32).tobytes()).decode(),
                                  f=base64.b64encode(f.astype(np.uint32).tobytes()).decode()))
    json.dump(prev, open(os.path.join(OUT, "preview.json"), "w"))
    json.dump([dict(zip(("key", "title", "color", "count", "x", "y", "z", "cm3"), s)) for s in summary],
              open(os.path.join(OUT, "parts.json"), "w"), ensure_ascii=False, indent=1)
    print("треугольников в превью:", sum(len(tris_of(p["m"])[1]) for p in parts))

if __name__ == "__main__":
    main()
