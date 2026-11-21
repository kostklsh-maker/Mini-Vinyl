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
    exc=(50.0, 30.0, 34.0),   # вибродинамик AIYIMA 50 мм 4 Ом: диаметр корпуса, полная высота, Ø пятака
    # --- платы ---
    esp=(28.3, 51.6),         # ESP32 DevKit 30 pin: ширина, длина
    amp=(18.0, 19.0),         # MAX98357A: ширина, длина
    dac=(17.5, 32.0),         # GY-PCM5102: ширина, длина (гнездо 3,5 мм на торце)
    sd=(24.0, 42.0),          # модуль microSD: ширина, длина (щель карты на торце)
    m5enc=(32.0, 24.0, 11.0, 16.0),   # модуль M5Stack Unit Encoder: ширина, высота, толщина корпуса без ручки, Ø ручки
)

# ------------------------------------------------------------------ геометрия корпуса
HB = 54.0                    # полуширина коробки стенок (108 × 108)
NOTCH = 12.0                 # угловая ниша под балясину
Z_GND = -3.0                 # стол (низ ножек-«булок»)
Z_FLOOR, Z_PL, Z_W1, Z_DECK, Z_TOP = 10.4, 18.0, 61.0, 68.0, 71.0   # пол, верх основания, низ карниза, низ и верх деки
DZ = Z_DECK - 89.0           # сдвиг светового стакана
WIN = HB - P["wall"]         # внутренняя грань стенки
COLP = HB - NOTCH + 6.6      # центр балясины
C = np.array([-12.0, -12.0])  # центр диска (пластинка, свет, PN532)
H = np.array([29.9, 29.9])   # ось трубы граммофона: канал из корпуса в раструб
BOSS = [(-32.0, -(WIN - 3), (0, -1)), (WIN - 3, -30.0, (1, 0)), (WIN - 3, 30.0, (1, 0)), (-10.0, WIN - 3, (0, 1))]   # стойки винтов основания: x, y, сторона стенки
EARS = [45.0, 135.0, 315.0]  # ушки светового стакана
EAR_R = 43.5
ESP_X = -33.0                # ESP32 стоит на ребре вдоль левой стенки: плоскость платы; штыри смотрят внутрь корпуса (+x)
ESP_ZB = 22.0                # нижняя кромка платы ESP32
USB_X, USB_Z = ESP_X - 2.4, ESP_ZB + P["esp"][0] / 2
TOW_X, AUX_Z, SD_Z = 25.0, 30.0, 44.0      # «этажерка» у задней стенки: ЦАП внизу, microSD над ним
ENC_Z = 38.5                 # центр ручки громкости на лице
ENC_SCR = 21.0               # винты держателя ручки: ±x от центра
EXC = np.array([2.0, -10.0])   # центр вибродинамика на дне
EXC_POSTS = [90.0, 225.0, 330.0]
EXC_PR = 28.5
AMP_C = (-38.3, -20.0)       # центр усилителя (длинной стороной вдоль Y)
PLAT_R, PLAT_T = 40.0, 4.0   # диск-платтер (снаружи — латунный обод до Ø86,8)
RISER_L = 55.0               # высота стойки трубы над декой
BELL_L, BELL_R = 72.0, 46.0  # раструб: длина и радиус устья
TUBE_R, BORE_R, SPIG_R, SPIG_L = 12.0, 7.5, 9.5, 8.0
BEND_R, BEND_A = 32.0, 45.0
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
    p = np.asarray(pts, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:      # всегда против часовой
        p = p[::-1]
    return CrossSection([p])

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
def boss_shape(x, y, side):
    """Стойка под винт основания на стенке: 18 мм, выше сходит на нет под 45° (печать декой вниз без поддержек)."""
    c = cyl(4.5, Z_PL, Z_PL + 18, x, y)
    wx, wy = x + side[0] * 3.0, y + side[1] * 3.0
    tip = box(wx - 0.6, wy - 0.6, Z_PL + 26.5, wx + 0.6, wy + 0.6, Z_PL + 27)
    return Manifold.batch_hull([c, tip])

def make_body():
    """Корпус: четыре стенки и верхняя дека одной деталью. Печатается декой вниз."""
    o = outline()
    inner = o.offset(-P["wall"], JoinType.Miter)
    cap = tiers([(Z_W1, HB + 2.5), (Z_W1 + 2, HB + 2.5), (Z_W1 + 3.5, HB + 5), (Z_W1 + 5, HB + 5), (Z_W1 + 6.5, HB + 8),
                 (Z_TOP - 0.6, HB + 8), (Z_TOP, HB + 7.4)], 9.0)
    m = ext(o, Z_PL, Z_W1 + 0.5) + cap
    m -= ext(inner, Z_PL - 1, Z_DECK)
    for x, y, side in BOSS:
        m += boss_shape(x, y, side) ^ ext(o, Z_PL, Z_TOP)
        m -= cyl(1.3, Z_PL - 1, Z_PL + 14, x, y)
    # воротник светового стакана и ушки
    cx, cy = C
    m += cyl(39.0, Z_DECK - 10, Z_DECK + 0.5, cx, cy) ^ ext(o, Z_PL, Z_TOP)
    for a in EARS:
        ex, ey = cx + EAR_R * math.cos(math.radians(a)), cy + EAR_R * math.sin(math.radians(a))
        m += cyl(5.5, Z_DECK - 10, Z_DECK + 0.5, ex, ey)
        m -= cyl(1.3, Z_DECK - 11, Z_DECK - 1, ex, ey)
    m -= cyl(36.0, Z_DECK - 11, Z_TOP + 1, cx, cy)
    # горло трубы: канал из корпуса в раструб
    m -= cyl(SPIG_R + P["fit"], Z_DECK - 1, Z_TOP + 1, H[0], H[1])
    # лицо: окно ручки громкости, два винта держателя, декоративные «прорези» (глухие)
    m -= prism_y(CrossSection.circle(13.7).translate([0, ENC_Z]), -HB - 1, -HB + P["wall"] + 1)
    for sx in (-1, 1):
        m -= prism_y(CrossSection.circle(1.7, 24).translate([sx * ENC_SCR, ENC_Z]), -HB - 1, -HB + P["wall"] + 1)
    # задняя стенка: USB-C (плата стоит на ребре — разъём вертикально), AUX, microSD  (u = −x)
    m -= prism_y(rrect(3.9, 6.6, 2).translate([-USB_X, USB_Z]), WIN - 1, HB + 1)
    m -= prism_y(CrossSection.circle(3.5).translate([-TOW_X, AUX_Z]), WIN - 1, HB + 1)
    m -= prism_y(rrect(8.0, 1.75, 1).translate([-TOW_X, SD_Z]), WIN - 1, HB + 1)
    return m

def strip_y(pts, y0, y1):
    """Сечение (x, z) → тело вдоль Y от y0 до y1."""
    return xf(poly(pts).extrude(y1 - y0), [(1, 0, 0), (0, 0, 1), (0, -1, 0)], (0, y1, 0))

def make_base():
    """Основание: поддон с карнизом, стойками под винты и ложементами всей электроники."""
    o = outline()
    inner = o.offset(-P["wall"], JoinType.Miter)
    pocket = inner.offset(-2.3, JoinType.Miter)
    m = tiers([(8.0, HB + 8.4), (8.6, HB + 9), (11.0, HB + 9), (12.0, HB + 7), (14.5, HB + 7), (15.5, HB + 4.5), (Z_PL, HB + 4.5)], 11.0)
    m -= ext(pocket, Z_FLOOR, Z_PL + 1)
    lip = ext(inner.offset(-P["fit"], JoinType.Miter) - inner.offset(-P["fit"] - 2.0, JoinType.Miter), Z_PL - 0.5, Z_PL + 3)
    for x, y, side in BOSS:
        lip -= cyl(5.2, Z_PL - 1, Z_PL + 4, x, y)
    m += lip
    y1 = WIN - 0.5                                   # задний торец плат у стенки
    F = Z_FLOOR - 0.1
    # --- ESP32 на ребре: колонны со стороны стенки, низкие щёчки со стороны деталей, упор спереди
    ew, el = P["esp"]
    y0 = y1 - el - 0.3
    xl, xr = ESP_X - 0.95, ESP_X + 0.95              # щель под плату
    for ya, yb, stop in ((y0, y0 + 6, True), (y1 - 6, y1, False)):
        col = box(xr, ya - (2 if stop else 0), F, xr + 7, yb, ESP_ZB + ew)
        col += box(xl - 2.5, ya - (2 if stop else 0), F, xr + 7, yb, ESP_ZB + 4)
        col -= box(xl, ya, ESP_ZB, xr, yb + 1, ESP_ZB + 5)
        col -= cyl(1.3, ESP_ZB + ew - 12, ESP_ZB + ew + 1, xr + 3.5, (ya + yb) / 2)
        m += col
    # --- «этажерка»: ЦАП на полке внизу, microSD на уступах над ним, оба торцом в заднюю стенку
    (dw, dl), (sw, sl) = P["dac"], P["sd"]
    zd, zs = AUX_Z - 4.1, SD_Z - 2.6
    sh = box(TOW_X - dw / 2 - 1.4, y1 - dl - 6, F, TOW_X + dw / 2 + 1.4, y1 + 0.4, zd + 1.6)
    sh -= box(TOW_X - dw / 2 - 0.2, y1 - dl, zd, TOW_X + dw / 2 + 0.2, y1 + 1, zd + 3)
    for dx in (-dw / 2 + 4, dw / 2 - 4):
        sh -= cyl(1.3, zd - 9, zd + 2, TOW_X + dx, y1 - dl - 3)
    m += sh
    hi = sw / 2 + 0.2
    for sx in (-1, 1):
        xi = TOW_X + sx * hi
        m += box(min(xi, xi + sx * 1.6), y1 - sl + 5, F, max(xi, xi + sx * 1.6), y1 + 0.4, zs + 3.1)
        m += strip_y([(xi + sx * 0.5, zs - 2.0), (xi + sx * 0.5, zs), (xi - sx * 1.5, zs), (xi, zs - 1.5)], y1 - sl + 5, y1 + 0.4)
        pil = box(min(xi, xi + sx * 4.2), y1 - sl + 5, F, max(xi, xi + sx * 4.2), y1 - sl + 11, zs + 3.6)
        m += pil - cyl(1.3, zs - 7, zs + 4, xi + sx * 2.3, y1 - sl + 8)
    # --- усилитель MAX98357A: карман на полу, длинной стороной вдоль Y
    aw, al = P["amp"]
    ax, ay = AMP_C
    bay = box(ax - aw / 2 - 1.8, ay - al / 2 - 1.8 - 5, F, ax + aw / 2 + 1.8, ay + al / 2 + 1.8 + 5, Z_FLOOR + 2.0)
    bay -= box(ax - aw / 2 - 0.2, ay - al / 2 - 0.2, Z_FLOOR, ax + aw / 2 + 0.2, ay + al / 2 + 0.2, Z_FLOOR + 3)
    m += bay
    for sy in (-1, 1):
        m -= cyl(1.3, 7.0, Z_FLOOR + 3, ax, ay + sy * (al / 2 + 4.3))
    # --- вибродинамик: три стойки ограничителя вокруг (сам клеится пятаком к полу)
    ed, eh, epd = P["exc"]
    for a in EXC_POSTS:
        px, py = EXC[0] + EXC_PR * math.cos(math.radians(a)), EXC[1] + EXC_PR * math.sin(math.radians(a))
        m += cyl(3.5, F, Z_FLOOR + eh + 2.0, px, py) - cyl(1.3, Z_FLOOR + eh - 10, Z_FLOOR + eh + 3, px, py)
    # --- стойки и отверстия под винты M3 корпуса
    for x, y, side in BOSS:
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

def make_esp_cap():
    """Колпачок на колонну ESP32: прижимает верхнюю кромку платы. Локально: щель платы на x = 0, низ пластины z = 0."""
    m = box(-3.45, 0, 0, 7.95, 6.0, 2.4) + box(-3.45, 0, -3.0, -0.95, 6.0, 0.1)
    return m - cyl(1.7, -1, 3, 4.45, 3.0)

def make_sd_clamp():
    """Перемычка над модулем microSD с двумя лапками на кромки платы. Локально: низ перемычки z = 0."""
    hi = P["sd"][0] / 2 + 0.2
    m = box(-hi - 4.2, 0, 0, hi + 4.2, 6.0, 2.4)
    for sx in (-1, 1):
        m += box(min(sx * (hi - 0.4), sx * (hi - 2.6)), 0, -2.0, max(sx * (hi - 0.4), sx * (hi - 2.6)), 6.0, 0.1)
        m -= cyl(1.7, -1, 3, sx * (hi + 2.3), 3.0)
    return m

def make_amp_clamp():
    """Рамка-прижим усилителя: окно над платой, края держат плату за две кромки. Локально: центр платы в нуле."""
    aw, al = P["amp"]
    m = box(-aw / 2 - 1.8, -al / 2 - 6.8, 0, aw / 2 + 1.8, al / 2 + 6.8, 2.0)
    m -= box(-aw / 2 - 0.2, -al / 2 + 1.2, -1, aw / 2 + 0.2, al / 2 - 1.2, 3)
    for sy in (-1, 1):
        m -= cyl(1.7, -1, 3, 0, sy * (al / 2 + 4.3))
    return m

def make_exc_cage():
    """Ограничитель вибродинамика: три луча на стойки, держит корпус динамика при переноске. Локально: центр динамика."""
    m = cyl(15.0, 0, 2.4)
    for a in EXC_POSTS:
        c, sn = math.cos(math.radians(a)), math.sin(math.radians(a))
        n = np.array([-sn, c]) * 3.5; e = np.array([c, sn]) * EXC_PR
        m += ext(poly([n, -n, e - n, e + n]), 0, 2.4) + cyl(3.5, 0, 2.4, e[0], e[1])
    m -= cyl(7.0, -1, 3)
    for a in EXC_POSTS:
        m -= cyl(1.7, -1, 3, EXC_PR * math.cos(math.radians(a)), EXC_PR * math.sin(math.radians(a)))
    return m

def make_enc_holder():
    """Держатель модуля M5Stack Unit Encoder (белый). Локально: z = 0 — наружная грань стенки, +z внутрь корпуса.
    Модуль лежит в кармане ручкой наружу; держатель и крышку стягивают с лицом корпуса два винта M3 с гайками."""
    w, h, d, kd = P["m5enc"]
    t, f = P["wall"], 1.6
    rk = kd / 2 + 1.5                                                    # зазор вокруг ручки: в нём светятся диоды модуля
    m = cyl(13.5, 0, t + 0.1) + ext(rrect(ENC_SCR + 3.5, h / 2 + 2.2, 3), t, t + f)
    m += ext(rrect(w / 2 + 2.2, h / 2 + 2.2, 2), t + f, t + f + d)
    for sx in (-1, 1):
        m += cyl(3.5, t + f, t + f + d, sx * ENC_SCR, 0)
        m -= cyl(1.7, -1, t + f + d + 1, sx * ENC_SCR, 0)
    m -= cyl(rk, -1, t + f + 0.5)
    m -= ext(rrect(w / 2 + P["fit"], h / 2 + P["fit"], 0.3), t + f, t + f + d + 1)     # карман модуля
    m -= box(w / 2 - 1, -6.0, t + f, w / 2 + 3, 6.0, t + f + d + 1)                    # выход кабеля Grove
    return m

def make_enc_cap():
    """Крышка держателя: прижимает модуль сзади, гайки M3 в шестигранных гнёздах. Локально: z = 0 к модулю."""
    w, h, d, kd = P["m5enc"]
    m = ext(rrect(ENC_SCR + 3.5, h / 2 + 2.2, 3), 0, 2.4)
    for sx in (-1, 1):
        m -= cyl(1.7, -1, 3, sx * ENC_SCR, 0)
        m -= cyl(3.2, 1.0, 3, sx * ENC_SCR, 0, seg=6)                    # гайка M3 (под ключ 5,5)
    m -= box(w / 2 - 1, -6.0, -1, w / 2 + 3, 6.0, 3)                       # кабель
    m -= ext(rrect(w / 2 - 6, h / 2 - 6, 2), -1, 3)                        # окно: облегчение и проверка модуля
    return m

# ------------------------------------------------------------------ стимпанк-декор
def sphere(r, x=0.0, y=0.0, z=0.0, seg=16):
    return Manifold.sphere(r, seg).translate([x, y, z])

def rod(a, b, r, seg=20):
    """Цилиндр между точками a и b (3D)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a; L = float(np.linalg.norm(d)); w = d / L
    u = np.cross(w, [0, 0, 1.0]) if abs(w[2]) < 0.9 else np.cross(w, [1.0, 0, 0])
    u /= np.linalg.norm(u); v = np.cross(w, u)
    return xf(cyl(r, 0, L, seg=seg), [u, v, w], a)

def flat_back(m):
    """Срезать всё, что ниже z = 0: накладка ложится плоской спиной на стенку и на стол принтера."""
    return m ^ box(-500, -500, 0, 500, 500, 500)

DIAL_R = 15.0                # циферблат вокруг ручки: наружный радиус
PANEL = (41.0, 18.0)         # латунная панель на лице: полуширина, полувысота

def make_panel():
    """Латунная панель лица: рамка, обод циферблата, тумблеры, ручки-обманки, лампочки, винтики.
    Локально: z = 0 — к стенке, +z наружу; (0, 0) — ось ручки громкости."""
    pw, ph = PANEL
    m = ext(rrect(pw, ph, 3), 0, 2.4)
    m += Manifold.batch_hull([ext(rrect(pw, ph, 3), 2.39, 2.4), ext(rrect(pw - 0.8, ph - 0.8, 2.4), 2.99, 3.0)]) - \
        ext(rrect(pw - 2.4, ph - 2.4, 1.8), 2.3, 4)                                 # рамка с фаской
    m += cyl(DIAL_R + 2.6, 0, 3.4) + cyl(DIAL_R + 2.6, 3.4, 4.2, r2=DIAL_R + 1.8)          # обод циферблата
    m -= cyl(DIAL_R + P["fit"], -1, 5)
    for sx in (-1, 1):
        m -= cyl(3.4, -1, 1.6, sx * ENC_SCR, 0)                                     # головки винтов держателя
        for sy in (-1, 1):
            m += sphere(1.2, sx * (pw - 3.4), sy * (ph - 3.4), 2.4, 12)             # винтики по углам
        for x in (sx * 26.0, sx * 34.0):                                            # тумблеры
            m += cyl(2.6, 2.4, 3.8, x, 8.0, seg=6) + cyl(1.7, 3.8, 5.0, x, 8.0, seg=16)
            tip = np.array([x, 8.0 - 2.6, 5.0 + 5.6])
            m += rod([x, 8.0, 4.8], tip, 0.85, 12) + sphere(1.2, *tip, seg=12)
        m += sphere(2.0, sx * 26.0, -1.5, 2.4, 14) + sphere(2.0, sx * 34.0, -1.5, 2.4, 14)   # лампочки-обманки
        kx = sx * 30.0                                                              # ручка-обманка
        k = cyl(4.2, 2.4, 6.4, kx, -10.0, seg=36) + cyl(3.2, 6.4, 7.2, kx, -10.0, seg=36)
        for t in range(16):
            a = 2 * math.pi * t / 16
            k -= cyl(0.55, 3.2, 6.5, kx + 4.4 * math.cos(a), -10.0 + 4.4 * math.sin(a), seg=8)
        m += k
    return m

def dial_ticks(r0, r1, n, z, h, long_every=5, arc=270.0, w=0.7):
    t = []
    for i in range(n):
        a = math.radians(-arc / 2 + arc * i / (n - 1) - 90)
        rr0 = r0 - (0.9 if i % long_every == 0 else 0)
        c, sn = math.cos(a), math.sin(a)
        nrm = np.array([-sn, c]) * w / 2
        p0, p1 = np.array([c, sn]) * rr0, np.array([c, sn]) * r1
        t.append(ext(poly([p0 + nrm, p0 - nrm, p1 - nrm, p1 + nrm]), z, z + h))
    return union(t)

def make_dial_face():
    """Белый циферблат вокруг ручки: подсвечивается диодами модуля изнутри. Шкала 0–10 рисками.
    Печать: спиной на стол, на высоте 3,2 мм — пауза и смена на чёрный (риски)."""
    rk = P["m5enc"][3] / 2 + 1.5
    m = cyl(DIAL_R, 0, 3.2) - cyl(rk, -1, 4)
    return m + dial_ticks(DIAL_R - 3.6, DIAL_R - 1.0, 21, 3.2, 0.6)

def make_column():
    """Латунная колонна с каннелюрами (вместо балясины)."""
    hc = Z_W1 - Z_PL - 0.3
    prof = [(0, 6.3), (2.2, 6.3), (2.6, 5.7), (3.6, 5.7), (4.0, 5.2), (hc - 5.0, 5.2), (hc - 4.4, 5.8), (hc - 3.0, 5.8),
            (hc - 2.4, 6.3), (hc, 6.3)]
    m = poly([(0.0, 0.0)] + [(r, z) for z, r in prof] + [(0.0, hc)]).revolve(64)
    fl = []
    for k in range(12):
        a = 2 * math.pi * k / 12
        x, y = 5.55 * math.cos(a), 5.55 * math.sin(a)
        fl.append(cyl(0.75, 5.5, hc - 6.0, x, y, seg=10) + sphere(0.75, x, y, 5.5, 8) + sphere(0.75, x, y, hc - 6.0, 8))
    return m - union(fl)

def make_foot():
    """Латунная ножка-«булка». Локально: z = 0 — стол, верх (фланец) — под основанием."""
    hf = 8.0 - Z_GND
    pr = [(0, 7.0), (1.5, 8.6), (4.0, 9.4), (6.5, 8.6), (hf - 2.8, 6.6), (hf - 2.0, 6.3), (hf - 1.2, 7.6), (hf, 7.6)]
    return poly([(0.0, 0.0)] + [(r, z) for z, r in pr] + [(0.0, hf)]).revolve(64)

def make_spiral_grille(R=17.0):
    """Решётка-обманка со спиралью, как у динамика на картинке. Локально: спина на z = 0."""
    m = cyl(R, 0, 1.2)
    holes = []
    for gx in np.arange(-R, R + 0.1, 2.2):
        for gy in np.arange(-R, R + 0.1, 2.2):
            if gx * gx + gy * gy < (R - 3.2) ** 2:
                holes.append(cyl(0.65, -1, 2, gx, gy, seg=8))
    m -= union(holes)
    m += cyl(R, 0, 3.0) - cyl(R - 2.4, -1, 4)
    m += cyl(R - 0.3, 3.0, 3.6, r2=R - 1.0) - cyl(R - 2.4, 2, 5)
    arms = []
    for k in range(8):
        a0 = 2 * math.pi * k / 8
        pl, prr = [], []
        for r in np.linspace(2.5, R - 2.0, 16):
            a = a0 + 0.075 * r
            n = np.array([-math.sin(a), math.cos(a)]); c = np.array([r * math.cos(a), r * math.sin(a)])
            pl.append(c + n * 0.75); prr.append(c - n * 0.75)
        arms.append(ext(poly(prr + pl[::-1]), 0, 2.4))
    m += union(arms) + cyl(3.0, 0, 3.0) + sphere(1.6, 0, 0, 3.0, 14)
    return m

def make_gear(R, teeth, t=2.4):
    """Шестерёнка-накладка со спицами. Локально: спина на z = 0."""
    pts = []
    for i in range(teeth):
        a = 2 * math.pi * i / teeth; s = 2 * math.pi / teeth
        for f, rr in ((0.0, R - 1.6), (0.18, R), (0.5, R), (0.68, R - 1.6)):
            pts.append((rr * math.cos(a + f * s), rr * math.sin(a + f * s)))
    m = ext(poly(pts), 0, t)
    ri, ro = 3.4, R - 3.2
    if ro - ri > 2.5:
        for k in range(5):
            a0, a1 = 2 * math.pi * k / 5 + 0.22, 2 * math.pi * (k + 1) / 5 - 0.22
            win = [(ri * math.cos(a), ri * math.sin(a)) for a in np.linspace(a0, a1, 6)] + \
                  [(ro * math.cos(a), ro * math.sin(a)) for a in np.linspace(a1, a0, 10)]
            m -= ext(poly(win), -1, t + 1)
    m += cyl(3.0, 0, t + 1.4) + sphere(1.4, 0, 0, t + 1.4, 12)
    return m

def make_tank():
    """Латунный бак (половинка капсулы) с обручами и патрубками. Локально: ось вдоль v, u = 0."""
    r, v0, v1 = 6.0, 25.0, 45.0
    zc = r * 0.35
    m = rod([0, v0 + r, zc], [0, v1 - r, zc], r, 40) + sphere(r, 0, v0 + r, zc, 32) + sphere(r, 0, v1 - r, zc, 32)
    for v in (v0 + 5.5, (v0 + v1) / 2, v1 - 5.5):
        m += rod([0, v - 0.8, zc], [0, v + 0.8, zc], r + 0.6, 40)
    m += rod([0, v1 - 1, zc], [0, v1 + 2.6, zc], 1.6, 16)                    # патрубок вверх к манометру
    m += rod([0, v0 + 1, zc], [0, v0 - 2.0, zc], 1.8, 16)                    # патрубок вниз к трубе
    return flat_back(m)

def make_gauge():
    """Корпус манометра: латунь, гнездо под белый циферблат. Локально: спина на z = 0."""
    m = cyl(6.0, 0, 3.6) + cyl(6.0, 3.6, 4.4, r2=5.4)
    m -= cyl(4.6, 3.0, 5)
    return m

def make_gauge_face():
    m = cyl(4.45, 0, 1.2)
    return m + dial_ticks(2.6, 4.0, 11, 1.2, 0.5, long_every=5, w=0.5) + ext(poly([(-0.35, 0), (0.35, 0), (1.8, 3.0), (1.5, 3.2)]), 1.2, 1.7)

PIPE_R = 2.2
def make_pipes():
    """Медные трубы-накладки правого бока (плоская спина). Локально: u — вдоль бока (мир Y), v — высота (мир Z)."""
    zc = PIPE_R * 0.55
    path = [(-39.0, 57.5), (-39.0, 20.8), (39.5, 20.8), (39.5, 40.0), (39.0, 40.0)]
    pts = [np.array([u, v, zc]) for u, v in path]
    parts = [rod(a, b, PIPE_R) for a, b in zip(pts, pts[1:])] + [sphere(PIPE_R, *p, seg=20) for p in pts]
    top = [np.array([-39.0, 57.5, zc]), np.array([-2.5, 57.5, zc])]
    parts += [rod(*top, PIPE_R), sphere(PIPE_R, *top[1], seg=20)]
    parts.append(rod([30.0, 20.8, zc], [30.0, 22.6, zc], PIPE_R))              # к баку снизу
    parts.append(rod([39.0, 40.0, zc], [36.9, 40.0, zc], 1.6))                 # к баку сбоку
    for u, v, horiz in ((-25, 20.8, True), (5, 20.8, True), (-39, 40, False), (-31, 57.5, True), (39.5, 30, False)):
        a = np.array([u - 1.2, v, zc]) if horiz else np.array([u, v - 1.2, zc])
        b = np.array([u + 1.2, v, zc]) if horiz else np.array([u, v + 1.2, zc])
        parts.append(rod(a, b, PIPE_R + 0.7))                                     # муфты
    parts.append(cyl(1.0, zc, zc + PIPE_R + 2.2, -39.0, 31.0) + cyl(3.0, zc + PIPE_R + 2.2, zc + PIPE_R + 3.0, -39.0, 31.0))  # вентиль
    return flat_back(union(parts))

def make_bulb_socket():
    return cyl(3.6, 0, 1.6, seg=32) + cyl(2.9, 1.6, 6.0, seg=32) + cyl(3.3, 3.2, 4.2, seg=32) - cyl(1.4, 2.0, 7, seg=16)

def make_bulb():
    """Лампа Эдисона (прозрачный пластик). Локально: цоколь вниз, штырь в патрон."""
    pr = [(0, 2.6), (2.5, 2.6), (3.2, 2.4), (5.5, 3.6), (9.0, 4.6), (12.5, 4.4), (15.0, 3.2), (16.6, 1.6), (17.2, 0.0)]
    m = poly([(0.0, 0.0)] + [(r, z) for z, r in pr]).revolve(48)
    return m + cyl(1.25, -2.0, 0.2, seg=16)

def make_soundbox():
    """Мембранная коробка на конце тонарма: диск с розеткой. Локально: спина на z = 0."""
    m = cyl(6.0, 0, 2.4, seg=48) + (cyl(6.0, 2.4, 3.2, seg=48) - cyl(5.0, 2, 4, seg=48))
    for k in range(6):
        a = 2 * math.pi * k / 6
        m -= ext(CrossSection.circle(1.25, 16).translate([3.0 * math.cos(a), 3.0 * math.sin(a)]), 1.6, 3)
    m += cyl(1.5, 2.4, 3.2, seg=16)
    return m

def make_rim():
    """Латунный обод вокруг диска."""
    zt = Z_TOP + PLAT_T
    m = cyl(43.4, Z_TOP, zt + 0.2, seg=128) + cyl(43.4, zt + 0.2, zt + 0.8, r2=42.8, seg=128)
    return m - cyl(PLAT_R + P["fit"], Z_TOP - 1, zt + 2, seg=128)

def cup_holes(m, z0, z1):
    for a in EARS:
        m -= cyl(1.7, z0, z1, EAR_R * math.cos(math.radians(a)), EAR_R * math.sin(math.radians(a)))
    return m

ZT = 92.0   # стакан построен в своих координатах (верх на 92), в сборке сдвигается на DZ

def make_cup():
    """Световой стакан: держит PN532 под пластинкой и кольцо WS2812 под световым каналом."""
    pl, pw, ph = P["pn532"]
    ro = P["ring_od"] / 2
    m = cyl(36.0 - P["fit"], 79.0, ZT) + cyl(39.0, 77.0, 79.0)
    for a in EARS:
        m += cyl(5.5, 77.0, 79.0, EAR_R * math.cos(math.radians(a)), EAR_R * math.sin(math.radians(a)))
    m = cup_holes(m, 76.0, 80.0)
    cav = poly([(ro - 7.5, 76.9), (ro + 0.6, 76.9), (ro + 0.6, 81.0), (ro - 0.6, 82.2), (ro - 0.6, 85.0),
                (33.5, 86.1), (33.5, 93.0), (30.5, 93.0), (30.5, 87.8), (ro - 7.5, 82.8)]).revolve(96)
    m -= cav
    for k in range(6):
        m += box(29.0, -0.6, 84.4, 34.5, 0.6, 89.6).rotate([0, 0, 15 + 60 * k])
    m -= box(-pl / 2 - 0.5, -pw / 2 - 0.5, ZT - ph, pl / 2 + 0.5, pw / 2 + 0.5, ZT + 1)
    m -= box(-12, -pw / 2 + 2, 76.0, 12, -pw / 2 + 7, ZT - ph + 0.5)       # окно для проводов PN532
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
    for r in np.arange(36.4, PLAT_R - 1.6, 1.4):
        m -= cyl(r + 0.35, zt - 0.3, zt + 1) - cyl(r, zt - 1, zt + 2)
    return m

def make_riser():
    """Стойка трубы: шип в деку, фланец, труба, шип под колено."""
    z0 = Z_TOP
    m = cyl(SPIG_R, Z_DECK + 0.2, z0 + 0.1) + cyl(TUBE_R + 3.5, z0, z0 + 2) + cyl(TUBE_R + 3.5, z0 + 2, z0 + 5.5, r2=TUBE_R)
    m += cyl(TUBE_R, z0 + 5.5, z0 + RISER_L - 2) + cyl(TUBE_R, z0 + RISER_L - 2, z0 + RISER_L, r2=SPIG_R)
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
    n, kexp, neck = 192, 1.3, 10.0
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
    m = outer - inner - cyl(SPIG_R + P["fit"], -1.0, SPIG_L + 0.3)
    def rad(z, a):
        t = max(0.0, (z - neck) / (BELL_L - neck))
        r = TUBE_R + (BELL_R - TUBE_R) * (math.exp(kexp * t) - 1) / (math.exp(kexp) - 1)
        return r * (1 + 0.09 * t * t * abs(math.sin(4 * a)) ** 0.7)
    deco = []
    for k in range(8):                                            # швы по впадинам лепестков + заклёпки
        a = math.pi / 4 * k
        c, sn = math.cos(a), math.sin(a)
        zs_ = np.linspace(neck + 2, BELL_L - 0.8, 22)
        pts = [(rad(z, a) - 0.25) * np.array([c, sn, 0]) + np.array([0, 0, z]) for z in zs_]
        for p, q in zip(pts, pts[1:]):
            deco.append(Manifold.batch_hull([sphere(0.55, *p, seg=8), sphere(0.55, *q, seg=8)]))
        for z in np.arange(neck + 6, BELL_L - 2, 6.5):
            for da in (-0.11, 0.11):
                aa = a + da * (TUBE_R + 6) / max(rad(z, a), 1)
                deco.append(sphere(0.7, *((rad(z, aa) - 0.15) * np.array([math.cos(aa), math.sin(aa), 0]) + np.array([0, 0, z])), seg=10))
    for z, nn in ((BELL_L - 3.0, 48), (neck + 1.5, 16)):          # пояса заклёпок у устья и у горла
        for i in range(nn):
            aa = 2 * math.pi * (i + 0.5) / nn
            deco.append(sphere(0.7, *((rad(z, aa) - 0.15) * np.array([math.cos(aa), math.sin(aa), 0]) + np.array([0, 0, z])), seg=10))
    return m + union(deco)

def make_bell():
    p0, e, ax, u, v = horn_frames()
    return xf(bell_local(), [u, -v, ax], e)

def make_arm():
    """Тонарм: хомут на стойке трубы, короткая штанга и головка над краем диска."""
    z0, z1 = Z_TOP + 5.7, Z_TOP + 12.7
    d = (C - H) / np.linalg.norm(C - H)
    head = C - d * 37.0
    n = np.array([-d[1], d[0]])
    s0, s1 = H + d * (TUBE_R + 1.5), head - d * 1.6
    beam = poly([s0 + n * 2.6, s0 - n * 2.6, s1 - n * 2.6, s1 + n * 2.6])
    m = ext(beam, z0, z1 - 1.5) + cyl(TUBE_R + 3.0, z0, z1, H[0], H[1])
    m += cyl(TUBE_R + 3.6, z1 - 1.2, z1, H[0], H[1])
    return m - cyl(TUBE_R + 0.2, z0 - 1, z1 + 1, H[0], H[1])

# ------------------------------------------------------------------ макеты покупных деталей (для проверки места)
def ghosts():
    """→ [(ключ, название, тело в сборке, вектор разнесения)]"""
    g = []
    cx, cy = C; y1 = WIN - 0.5
    pl, pw, ph = P["pn532"]
    g.append(("e_pn532", "PN532", box(cx - pl / 2, cy - pw / 2, Z_TOP - ph + 0.2, cx + pl / 2, cy + pw / 2, Z_TOP - 0.1), (0, 0, 62)))
    ring = cyl(P["ring_od"] / 2, 79.4, 79.4 + P["ring_pcb"]) + cyl(P["ring_od"] / 2 - 1, 81.0, 82.6)
    g.append(("e_ring", "Кольцо WS2812", (ring - cyl(P["ring_id"] / 2, 78, 84)).translate([cx, cy, DZ]), (0, 0, -30)))
    ed, eh, epd = P["exc"]
    exc = cyl(epd / 2, Z_FLOOR, Z_FLOOR + eh - 20.0) + cyl(ed / 2, Z_FLOOR + eh - 20.0, Z_FLOOR + eh)
    g.append(("e_exc", "Вибродинамик", exc.translate([EXC[0], EXC[1], 0]), (0, 0, -52)))
    ew, el = P["esp"]; y0 = y1 - el; zb = ESP_ZB
    esp = box(ESP_X - 0.8, y0, zb, ESP_X + 0.8, y1, zb + ew)
    esp += box(ESP_X - 4.2, y0 + 1, zb + ew / 2 - 9, ESP_X - 0.8, y0 + 27, zb + ew / 2 + 9)        # модуль
    esp += box(ESP_X - 4.0, y1 - 7.5, zb + ew / 2 - 4.5, ESP_X - 0.8, y1, zb + ew / 2 + 4.5)       # USB-C
    for zc in (zb + 1.3, zb + ew - 1.3):                                                        # штыри + разъёмы проводов
        esp += box(ESP_X + 0.8, y0 + 7, zc - 1.3, ESP_X + 16.8, y1 - 7, zc + 1.3)   # Г-образные разъёмы или пайка
    g.append(("e_esp", "ESP32", esp, (0, 0, -70)))
    aw, al = P["amp"]; ax, ay = AMP_C
    a = box(ax - aw / 2, ay - al / 2, Z_FLOOR + 0.3, ax + aw / 2, ay + al / 2, Z_FLOOR + 1.9)
    a += box(ax - aw / 2 + 4, ay - al / 2 + 1.5, Z_FLOOR + 1.9, ax + aw / 2 - 4, ay - al / 2 + 9, Z_FLOOR + 11.5)   # клеммник
    g.append(("e_amp", "Усилитель MAX98357A", a, (0, 0, -70)))
    (dw, dl), (sw, sl) = P["dac"], P["sd"]
    zd, zs = AUX_Z - 4.1, SD_Z - 2.6
    dac = box(TOW_X - dw / 2, y1 - dl, zd, TOW_X + dw / 2, y1, zd + 1.6) + box(TOW_X - 3.2, y1 - 12, zd + 1.6, TOW_X + 3.2, y1, zd + 6.6)
    dac += xf(cyl(3.0, 0, 3.0), [(-1, 0, 0), (0, 0, 1), (0, 1, 0)], (TOW_X, y1, AUX_Z))
    g.append(("e_dac", "ЦАП PCM5102A", dac, (0, 0, -70)))
    sd = box(TOW_X - sw / 2, y1 - sl, zs, TOW_X + sw / 2, y1, zs + 1.6) + box(TOW_X - 7.5, y1 - 16, zs + 1.6, TOW_X + 7.5, y1, zs + 3.4)
    g.append(("e_sd", "Модуль microSD", sd, (0, 0, -70)))
    w, h, d, kd = P["m5enc"]
    y0 = -HB + P["wall"] + 1.6
    enc = box(-w / 2, y0, ENC_Z - h / 2, w / 2, y0 + d, ENC_Z + h / 2)
    enc += xf(cyl(kd / 2, 0, 25.4 - d), [(1, 0, 0), (0, 0, 1), (0, -1, 0)], (0, y0, ENC_Z))
    enc += box(-w / 2 - 2.5, y0 + 2, ENC_Z - 4, -w / 2, y0 + 9, ENC_Z + 4)                    # разъём Grove
    g.append(("e_enc", "Модуль M5Stack Unit Encoder", enc, (0, 26, 0)))
    return g


# ------------------------------------------------------------------ сборка
COL = dict(brown="#4a1a14", gold="#c9a24a", black="#1c1c1e", white="#f2efe6", gray="#8d8f94", pcb="#2f7d4f", copper="#b8693e", clear="#f6e7c1")

def build():
    """→ список деталей: имя, название, цвет, тело в сборке, вектор разнесения, поворот для печати."""
    parts = []
    def add(key, title, color, m, explode, print_rot=(0, 0, 0), src=None):
        parts.append(dict(key=key, title=title, color=color, m=m, explode=explode, rot=print_rot, src=src or key))
    cx, cy = C; y1 = WIN - 0.5
    add("body", "Корпус (стенки + дека)", "brown", make_body(), (0, 0, 0), (180, 0, 0))
    add("base", "Основание", "brown", make_base(), (0, 0, -70))
    col, foot = make_column(), make_foot()
    for i, (sx, sy) in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
        add(f"column{i+1}", "Колонна с каннелюрами", "gold", col.translate([sx * COLP, sy * COLP, Z_PL + 0.15]), (sx * 35, sy * 35, 0), (0, 0, 0), "column")
        add(f"foot{i+1}", "Ножка-«булка»", "gold", foot.translate([sx * (HB - 6), sy * (HB - 6), Z_GND]), (0, 0, -92), None, "foot")
    add("cup", "Световой стакан", "white", make_cup().translate([cx, cy, DZ]), (0, 0, -18))
    add("retainer", "Прижим кольца", "white", make_retainer().translate([cx, cy, DZ]), (0, 0, -38))
    add("hub", "Гнездо пластинки", "black", make_hub().translate([cx, cy, 0]), (0, 0, 50))
    add("diffuser", "Рассеиватель", "white", make_diffuser().translate([cx, cy, 0]), (0, 0, 36), (180, 0, 0))
    add("platter", "Диск-платтер", "black", make_platter().translate([cx, cy, 0]), (0, 0, 22))
    add("rim", "Латунный обод диска", "gold", make_rim().translate([cx, cy, 0]), (0, 0, 14))
    # ручка громкости с подсветкой на лице
    add("enc_holder", "Держатель ручки", "white", xf(make_enc_holder(), [(-1, 0, 0), (0, 0, 1), (0, 1, 0)], (0, -HB, ENC_Z)), (0, 30, 0), None, "enc_holder")
    FRONT = [(1, 0, 0), (0, 0, 1), (0, -1, 0)]
    add("panel", "Латунная панель лица", "gold", xf(make_panel(), FRONT, (0, -HB, ENC_Z)), (0, -34, 0), None, "panel")
    add("dial", "Циферблат ручки", "white", xf(make_dial_face(), FRONT, (0, -HB, ENC_Z)), (0, -50, 0), None, "dial")
    # правый бок: решётка, шестерни, бак, манометр, трубы
    RIGHT = [(0, 1, 0), (0, 0, 1), (1, 0, 0)]
    LEFT = [(0, -1, 0), (0, 0, 1), (-1, 0, 0)]
    g = make_spiral_grille(15.5)
    add("grille_r", "Спиральная решётка", "gold", xf(g.translate([-19.5, 39.5, 0]), RIGHT, (HB, 0, 0)), (28, 0, 0), None, "grille")
    add("grille_l", "Спиральная решётка", "gold", xf(g.translate([0.0, 39.5, 0]), LEFT, (-HB, 0, 0)), (-28, 0, 0), None, "grille")
    add("gear1", "Шестерня большая", "gold", xf(make_gear(10.0, 16).translate([11.0, 48.5, 0]), RIGHT, (HB, 0, 0)), (34, 0, 0), None, "gear1")
    add("gear2", "Шестерня малая", "gold", xf(make_gear(7.5, 12).translate([14.0, 31.0, 0]), RIGHT, (HB, 0, 0)), (34, 0, 0), None, "gear2")
    add("tank", "Бак", "gold", xf(make_tank().translate([30.0, 0, 0]), RIGHT, (HB, 0, 0)), (30, 0, 0), None, "tank")
    add("gauge", "Манометр", "gold", xf(make_gauge().translate([30.0, 54.0, 0]), RIGHT, (HB, 0, 0)), (40, 0, 0), None, "gauge")
    add("gauge_face", "Циферблат манометра", "white", xf(make_gauge_face().translate([30.0, 54.0, 3.0]), RIGHT, (HB, 0, 0)), (48, 0, 0), None, "gauge_face")
    add("pipes", "Медные трубы", "copper", xf(make_pipes(), RIGHT, (HB, 0, 0)), (22, 0, 0), None, "pipes")
    # лампы Эдисона на деке справа
    for i, by in enumerate((-32.0, -20.0, -8.0)):
        add(f"socket{i+1}", "Патрон лампы", "gold", make_bulb_socket().translate([48.0, by, Z_TOP]), (0, 0, 26), None, "socket")
        add(f"bulb{i+1}", "Лампа Эдисона", "clear", make_bulb().translate([48.0, by, Z_TOP + 6.0]), (0, 0, 44), None, "bulb")
    w, h, d, kd = P["m5enc"]
    add("enc_cap", "Крышка держателя ручки", "white", xf(make_enc_cap(), [(-1, 0, 0), (0, 0, 1), (0, 1, 0)], (0, -HB + P["wall"] + 1.6 + d, ENC_Z)), (0, 46, 0), None, "enc_cap")
    # крепёж электроники
    ew, el = P["esp"]; y0 = y1 - el - 0.3
    cap = make_esp_cap()
    for i, ya in enumerate((y0, y1 - 6)):
        add(f"esp_cap{i+1}", "Колпачок ESP32", "gray", cap.translate([ESP_X, ya, ESP_ZB + ew + 0.2]), (0, 0, -42), None, "esp_cap")
    add("amp_clamp", "Рамка усилителя", "gray", make_amp_clamp().translate([AMP_C[0], AMP_C[1], Z_FLOOR + 2.0]), (0, 0, -55), None, "amp_clamp")
    add("dac_clamp", "Прижим ЦАП", "gray", end_clamp(P["dac"][0]).translate([TOW_X, y1 - P["dac"][1] - 6, AUX_Z - 4.1 + 1.6]), (0, 0, -55), None, "dac_clamp")
    add("sd_clamp", "Прижим microSD", "gray", make_sd_clamp().translate([TOW_X, y1 - P["sd"][1] + 5, SD_Z - 2.6 + 3.6]), (0, 0, -42), None, "sd_clamp")
    add("exc_cage", "Ограничитель вибродинамика", "gray", make_exc_cage().translate([EXC[0], EXC[1], Z_FLOOR + P["exc"][1] + 2.0]), (0, 0, -40), None, "exc_cage")
    add("riser", "Стойка трубы", "gold", make_riser(), (0, 0, 38), (180, 0, 0))
    add("arm", "Тонарм", "gold", make_arm(), (0, 0, 66))
    dd = (C - H) / np.linalg.norm(C - H); head = C - dd * 37.0
    sb = make_soundbox().translate([0, 0, -1.6])
    nrm = np.array([dd[0], dd[1], 0.0]); side = np.array([-dd[1], dd[0], 0.0])
    add("soundbox", "Мембранная коробка", "gold", xf(sb, [side, (0, 0, 1), nrm], (head[0], head[1], Z_TOP + 11.6)), (0, 0, 76), None, "soundbox")
    add("elbow", "Колено трубы", "gold", make_elbow(), (0, 0, 88))
    add("bell", "Раструб", "gold", make_bell(), (-38, -38, 140), None, "bell")
    return parts

def build_height():
    return max(p["m"].bounding_box()[5] for p in build()) - Z_GND

LOCAL = dict(column=make_column, enc_cap=make_enc_cap,
             foot=lambda: make_foot().rotate([180, 0, 0]),
             panel=make_panel, dial=make_dial_face, grille=lambda: make_spiral_grille(15.5),
             gear1=lambda: make_gear(10.0, 16), gear2=lambda: make_gear(7.5, 12),
             tank=make_tank, gauge=make_gauge, gauge_face=make_gauge_face, pipes=make_pipes,
             socket=make_bulb_socket, bulb=lambda: make_bulb().translate([0, 0, 2.0]), soundbox=make_soundbox,
             rim=lambda: make_rim(),
             enc_holder=lambda: make_enc_holder().rotate([180, 0, 0]),
             esp_cap=lambda: make_esp_cap().rotate([180, 0, 0]),
             sd_clamp=lambda: make_sd_clamp().rotate([180, 0, 0]),
             amp_clamp=make_amp_clamp, exc_cage=make_exc_cage, dac_clamp=lambda: end_clamp(P["dac"][0]))

def print_mesh(part):
    """Тело в положении печати, опущенное на стол, центр по XY в нуле."""
    if part["src"] == "bell":
        m = bell_local().rotate([180, 0, 0])
    elif part["src"] in LOCAL:
        m = LOCAL[part["src"]]()
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
    ("2_osnovanie_brown", ["base"]),
    ("3_truba_gold", ["bell", "riser", "elbow", "arm", "soundbox"]),
    ("4_dekor_gold", ["rim", "panel", "grille", "grille", "gear1", "gear2", "tank", "gauge", "socket", "socket", "socket",
                      "column", "column", "column", "column", "foot", "foot", "foot", "foot"]),
    ("5_disk_black", ["platter", "hub", "exc_cage", "esp_cap", "esp_cap", "amp_clamp", "dac_clamp", "sd_clamp"]),
    ("6_svet_white", ["cup", "retainer", "diffuser", "enc_holder", "enc_cap", "dial", "gauge_face"]),
    ("7_truby_copper", ["pipes"]),
    ("8_lampy_clear", ["bulb", "bulb", "bulb"]),
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
