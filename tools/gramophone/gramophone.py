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
    spk_d=40.0,               # динамики Ø40 (2 боковых + 1 под трубой)
    # --- платы на основании ---
    esp=(28.3, 51.6),         # ESP32 DevKit 30 pin: ширина, длина
)

# ------------------------------------------------------------------ геометрия корпуса
HB = 75.0                    # полуширина коробки стенок (150 × 150)
NOTCH = 13.0                 # угловая ниша под балясину
Z_FLOOR, Z_PL, Z_W1, Z_DECK, Z_TOP = 10.4, 18.0, 82.0, 89.0, 92.0   # пол, верх основания, низ карниза, низ и верх деки
C = np.array([-14.0, -8.0])  # центр диска (пластинка, свет, PN532)
H = np.array([44.0, 48.0])   # ось трубы граммофона и третьего динамика
SPK_Z = 50.0                 # высота центров боковых динамиков
BOSS = [(-40.0, -69.6), (40.0, -69.6), (-38.0, 69.6), (8.0, 69.6)]   # стойки под винты основания
EARS = [90.0, 210.0, 330.0]  # ушки светового стакана
EAR_R = 43.5
ESP_X, DAC_X, SD_X, ENC_X = -14.0, 30.0, 50.0, -50.0                 # разъёмы на задней стенке
USB_Z, AUX_Z, SD_Z, ENC_Z = 25.4, 30.0, 29.0, 42.0
PLAT_R, PLAT_T = 55.0, 4.0   # диск-платтер
RISER_L = 80.0               # высота стойки трубы над декой
BELL_L, BELL_R = 105.0, 75.0 # раструб: длина и радиус устья
TUBE_R, BORE_R, SPIG_R, SPIG_L = 15.0, 11.0, 13.0, 8.0
BEND_R, BEND_A = 40.0, 45.0
D_DIR = np.array([-1.0, -1.0, 0.0]) / math.sqrt(2)   # куда наклонён раструб


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
    cap = tiers([(Z_W1, 77.5), (84.0, 77.5), (85.5, 80.0), (87.0, 80.0), (88.5, 83.0), (91.4, 83.0), (Z_TOP, 82.4)], 10.5)
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
    m += cyl(rs + 2.7, 77.0, Z_DECK + 0.5, hx, hy)
    m -= cyl(rs, 76.0, 83.2, hx, hy)
    m -= cyl(rs - 1.3, 83.15, Z_DECK + 0.01, hx, hy, r2=SPIG_R + P["fit"])
    m -= cyl(SPIG_R + P["fit"], Z_DECK, Z_TOP + 1, hx, hy)
    # боковые динамики
    hole = teardrop(P["spk_d"] / 2 + 3.3).translate([0, SPK_Z])
    m -= prism_x(hole, HB - P["wall"] - 1, HB + 1)
    m -= prism_x(hole, -HB - 1, -HB + P["wall"] + 1)
    # лицо: декоративные прорези и углубление под шильдик
    vents = [lens(8, 17, -x * 1.1).translate([x, 67 - abs(x) * 0.12]) for x in (-30, -15, 0, 15, 30)]
    for v in vents:
        m -= prism_y(v, -HB - 1, -HB + P["wall"] + 1)
    m -= prism_y(rrect(35.2, 15.2, 3).translate([0, 38]), -HB - 1, -HB + 0.6)
    # задняя стенка: USB-C, AUX, microSD, ручка громкости (u = −x)
    m -= prism_y(rrect(7.0, 4.5, 2).translate([-ESP_X, USB_Z]), HB - P["wall"] - 1, HB + 1)
    m -= prism_y(CrossSection.circle(3.5).translate([-DAC_X, AUX_Z]), HB - P["wall"] - 1, HB + 1)
    m -= prism_y(rrect(8.0, 1.75, 1).translate([-SD_X, SD_Z]), HB - P["wall"] - 1, HB + 1)
    m -= prism_y(CrossSection.circle(3.6).translate([-ENC_X, ENC_Z]), HB - P["wall"] - 1, HB + 1)
    return m

def make_base():
    """Основание: поддон с карнизом, стойками под винты и ложементами плат."""
    o = outline()
    inner = o.offset(-P["wall"], JoinType.Miter)
    pocket = inner.offset(-2.3, JoinType.Miter)
    m = tiers([(8.0, 83.4), (8.6, 84.0), (11.0, 84.0), (12.0, 82.0), (14.5, 82.0), (15.5, 79.0), (Z_PL, 79.0)], 12.0)
    m -= ext(pocket, Z_FLOOR, Z_PL + 1)
    lip = ext(inner.offset(-P["fit"], JoinType.Miter) - inner.offset(-P["fit"] - 2.0, JoinType.Miter), Z_PL - 0.5, Z_PL + 3)
    for x, y in BOSS:
        lip -= cyl(5.2, Z_PL - 1, Z_PL + 4, x, y)
    m += lip
    # ложемент ESP32: плата лежит деталями вниз, штырями вверх, USB — в заднюю стенку
    ew, el = P["esp"]
    y1 = HB - P["wall"] - 0.5                       # задний торец платы
    y0 = y1 - el - 0.3
    zl = USB_Z + 1.6                                # опорная полка платы
    fr = box(ESP_X - ew / 2 - 2.2, y0 - 2, Z_FLOOR - 0.1, ESP_X + ew / 2 + 2.2, y1, zl + 3.0)
    fr -= box(ESP_X - ew / 2 - 0.2, y0, zl, ESP_X + ew / 2 + 0.2, y1 + 1, zl + 4)
    fr -= box(ESP_X - ew / 2 + 1.5, y0 + 1.5, Z_FLOOR, ESP_X + ew / 2 - 1.5, y1 + 1, zl + 4)
    fr -= box(ESP_X - 7.5, y1 - 6, Z_PL + 0.5, ESP_X + 7.5, y1 + 1, zl + 4)
    m += fr
    # полки под плату ЦАП (AUX) и модуль microSD
    m += box(DAC_X - 9, y1 - 34, Z_FLOOR - 0.1, DAC_X + 9, y1 - 2.4, AUX_Z - 4.1)
    m += box(SD_X - 9, y1 - 26, Z_FLOOR - 0.1, SD_X + 9, y1 - 2.4, SD_Z - 2.0)
    # стойки и отверстия под винты M3
    for x, y in BOSS:
        m += cyl(5.5, 8.0, Z_PL, x, y) ^ ext(inner, 8.0, Z_PL)
        m -= cyl(1.7, 7.0, Z_PL + 4, x, y)
        m -= cyl(3.3, 7.0, 13.0, x, y)
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
    for r in np.arange(37.0, 53.0, 1.6):
        m -= cyl(r + 0.35, zt - 0.3, zt + 1) - cyl(r, zt - 1, zt + 2)
    return m

def make_grille():
    """Решётка бокового динамика с гнездом. Локально: лицо на z = 0, труба вверх."""
    rs = P["spk_d"] / 2
    m = cyl(27.6, 0.0, 0.8, r2=29.0) + cyl(29.0, 0.8, 2.4) + cyl(rs + 3.0, 2.4, 9.0)
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
    m += cyl(2.2, 0.0, 2.4)
    return m

def make_plate():
    """Декоративный шильдик на лицо. Локально: лежит на z = 0, лицом вверх."""
    m = ext(rrect(35.0, 15.0, 3), 0.0, 1.6)
    m += ext(rrect(35.0, 15.0, 3) - rrect(33.4, 13.4, 2), 1.6, 2.4)
    m += ext(rrect(11.5, 4.8, 1.5).translate([0, 6.5]), 1.6, 2.6)
    m -= ext(rrect(10.0, 3.3, 1).translate([0, 6.5]), 2.0, 3.0)
    for sx in (-1, 1):
        for y in (6.5, -0.5, -7.5):
            m += ext(rrect(4.6, 2.2, 0.8).translate([sx * 24.5, y]), 1.6, 2.4)
    m -= ext(rrect(4.5, 1.6, 0.6).translate([0, -4.5]), 1.0, 3.0)
    for sx in (-1, 1):
        for sy in (-1, 1):
            m += cyl(1.3, 1.6, 2.6, sx * 31.2, sy * 11.2, seg=24)
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
    head = C - d * 46.0
    n = np.array([-d[1], d[0]])
    s0, s1 = H + d * 16.5, head
    beam = poly([s0 + n * 3.5, s0 - n * 3.5, s1 - n * 3.5, s1 + n * 3.5])
    m = ext(beam, z0, z1) + cyl(18.2, z0, z1, H[0], H[1]) + cyl(11.0, z0, z1, head[0], head[1])
    m += cyl(9.0, z1, z1 + 1.0, head[0], head[1]) + cyl(4.0, z1 + 1.0, z1 + 2.2, head[0], head[1])
    return m - cyl(TUBE_R + 0.2, z0 - 1, z1 + 1, H[0], H[1])

def make_knob():
    """Ручка громкости на вал энкодера EC11 (Ø6 с лыской). Локально: лицо на z = 0."""
    m = cyl(8.0, 0.0, 1.0, r2=9.0) + cyl(9.0, 1.0, 13.0)
    for k in range(18):
        a = 2 * math.pi * k / 18
        m -= cyl(0.9, 3.0, 14.0, 9.3 * math.cos(a), 9.3 * math.sin(a), seg=12)
    bore = cyl(3.1, 3.0, 14.0, seg=36) - box(1.6, -4, 2.0, 4, 4, 15.0)
    return m - bore


# ------------------------------------------------------------------ сборка
COL = dict(brown="#5c2019", gold="#c9a24a", black="#1c1c1e", white="#f2efe6")

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
        add(f"column{i+1}", "Балясина", "brown", col.translate([sx * 68.6, sy * 68.6, Z_PL + 0.15]),
            (sx * 45, sy * 45, 0), (0, 0, 0), "column")
        add(f"foot{i+1}", "Ножка", "brown", foot.translate([sx * 66, sy * 66, 0]), (0, 0, -100), (180, 0, 0), "foot")
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
    add("plate", "Шильдик", "gold", xf(make_plate(), [(1, 0, 0), (0, 0, 1), (0, -1, 0)], (0, -HB + 0.6, 38)),
        (0, -45, 0), None, "plate")
    add("knob", "Ручка громкости", "gold", xf(make_knob(), [(-1, 0, 0), (0, 0, 1), (0, -1, 0)], (ENC_X, HB + 17.0, ENC_Z)),
        (0, 40, 0), None, "knob")
    add("riser", "Стойка трубы", "gold", make_riser(), (0, 0, 45), (180, 0, 0))
    add("arm", "Тонарм", "gold", make_arm(), (0, 0, 78))
    add("elbow", "Колено трубы", "gold", make_elbow(), (0, 0, 105))
    add("bell", "Раструб", "gold", make_bell(), (-45, -45, 175), None, "bell")
    return parts

LOCAL = dict(grille=make_grille, plate=make_plate, knob=make_knob, column=make_column, foot=make_foot)

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
    ("5_disk_black", ["platter", "hub"]),
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
    json.dump(prev, open(os.path.join(OUT, "preview.json"), "w"))
    json.dump([dict(zip(("key", "title", "color", "count", "x", "y", "z", "cm3"), s)) for s in summary],
              open(os.path.join(OUT, "parts.json"), "w"), ensure_ascii=False, indent=1)
    print("треугольников в превью:", sum(len(tris_of(p["m"])[1]) for p in parts))

if __name__ == "__main__":
    main()
