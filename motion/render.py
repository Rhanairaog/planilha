import math, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops
from multiprocessing import Pool

BASE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(BASE, '..', '..', 'images')
OUT = os.path.join(BASE, 'frames')
os.makedirs(OUT, exist_ok=True)

W, H = 1080, 1920
FPS, DUR = 30, 10.0
N = int(FPS * DUR)

salon = Image.open(os.path.join(IMG, '2.png')).convert('RGB')
flyer = Image.open(os.path.join(IMG, '1.jpg')).convert('RGB')
FW, FH = flyer.size  # 1353 x 2000

# ---------- objeto: panfleto dentro de um display de acrilico ----------
M = 70           # margem do acrilico
LIP = 150        # aba frontal da base
OW, OH = FW + 2 * M, FH + 2 * M + LIP

def build_object():
    obj = Image.new('RGBA', (OW, OH), (0, 0, 0, 0))
    # acrilico (vidro levemente esverdeado e translucido)
    acr = Image.new('RGBA', (OW, OH), (0, 0, 0, 0))
    d = ImageDraw.Draw(acr)
    d.rounded_rectangle([0, 0, OW - 1, OH - LIP + 10], radius=26, fill=(225, 238, 236, 46))
    d.rounded_rectangle([-40, OH - LIP, OW + 39, OH - 1], radius=18, fill=(214, 232, 230, 90))
    obj = Image.alpha_composite(obj, acr)
    # papel com luz quente vinda do alto a direita
    f = np.asarray(flyer).astype(np.float32) / 255
    yy, xx = np.mgrid[0:FH, 0:FW].astype(np.float32)
    g = 1.04 - 0.16 * np.clip(np.hypot((xx - FW * 0.85) / FW, (yy + FH * 0.1) / FH) - 0.25, 0, 1)
    f = f * g[..., None] * np.array([1.03, 1.0, 0.95])
    paper = Image.fromarray((np.clip(f, 0, 1) * 255).astype(np.uint8)).convert('RGBA')
    obj.alpha_composite(paper, (M, M))
    # bordas polidas do acrilico
    hl = Image.new('RGBA', (OW, OH), (0, 0, 0, 0))
    d = ImageDraw.Draw(hl)
    d.rounded_rectangle([2, 2, OW - 3, OH - LIP + 8], radius=26, outline=(255, 255, 255, 150), width=7)
    d.line([(10, OH - LIP + 4), (OW - 10, OH - LIP + 4)], fill=(255, 255, 255, 190), width=6)
    d.line([(10, OH - 6), (OW - 10, OH - 6)], fill=(170, 190, 188, 160), width=6)
    hl = hl.filter(ImageFilter.GaussianBlur(2))
    obj = Image.alpha_composite(obj, hl)
    # reflexo diagonal fixo no acrilico
    gl = np.zeros((OH, OW), np.float32)
    yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
    band = (xx * 0.55 + yy) / (OW * 0.55 + OH)
    gl += 0.10 * np.exp(-((band - 0.18) / 0.035) ** 2)
    gl += 0.05 * np.exp(-((band - 0.24) / 0.012) ** 2)
    a = np.asarray(obj).astype(np.float32)
    a[..., :3] = a[..., :3] + (255 - a[..., :3]) * gl[..., None] * (a[..., 3:4] / 255)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

OBJ = build_object()
OBJ_NP = np.asarray(OBJ).astype(np.float32)
_yy, _xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
SWEEP_COORD = (_xx * 0.7 - _yy * 0.0 + _yy * 0.45) / (OW * 0.7 + OH * 0.45)

# ---------- mundo (coordenadas da foto do salao) ----------
ANCHOR = np.array([413.0, 167.2])   # base do display, na borda do balcao
OBJ_HW = 26.5                        # altura do objeto em px da foto
S_OBJ = OBJ_HW / OH                  # px foto por px do objeto

def ease_io(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * x * (x * (x * 6 - 15) + 10)

def seg(t, a, b):
    return ease_io((t - a) / (b - a))

def lerp(a, b, k):
    return a + (b - a) * k

def object_corners(t):
    """Projeta os 4 cantos do objeto (camera orbitando: yaw diminui)."""
    k = seg(t, 0.4, 7.4)
    yaw = math.radians(lerp(30, 5, k) + 1.2 * math.sin(t * 0.7))
    lean = math.radians(7)
    D = 3.2 * OH
    pts = []
    for (u, v) in [(0, 0), (OW, 0), (OW, OH), (0, OH)]:
        x = u - OW / 2
        y = OH - v                    # altura a partir da base
        # inclinacao para tras
        yz = y * math.sin(lean)
        y2 = y * math.cos(lean)
        X = x * math.cos(yaw)
        Z = x * math.sin(yaw) + yz
        p = D / (D + Z)
        pts.append((X * p, -y2 * p))
    return [(float(ANCHOR[0] + px * S_OBJ), float(ANCHOR[1] + py * S_OBJ)) for px, py in pts]

def world_point_on_object(t, u, v):
    c = object_corners(t)
    Hm = homography([(0, 0), (OW, 0), (OW, OH), (0, OH)], c)
    q = Hm @ np.array([u, v, 1.0])
    return q[:2] / q[2]

def homography(src, dst):
    A, b = [], []
    for (x, y), (X, Y) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -X * x, -X * y]); b.append(X)
        A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y]); b.append(Y)
    h = np.linalg.solve(np.array(A, float), np.array(b, float))
    return np.append(h, 1).reshape(3, 3)

# centro do cupom (no papel) e alvo final
COUPON_UV = (M + 676, M + 1590)

def camera(t):
    """Retorna centro (cx, cy) e altura visivel h em px da foto."""
    flyer_c = world_point_on_object(7.4, OW / 2, OH * 0.47)
    coupon = world_point_on_object(9.6, *COUPON_UV)
    # pan pela recepcao e, depois, zoom ancorado no display (o alvo nunca sai do quadro)
    h0, hf = 305.0, 34.0
    pan_x = lerp(112, 372, seg(t, 0.0, 3.8))
    pan_y = lerp(198, 192, seg(t, 0.0, 3.6))
    lh = lerp(math.log(h0), math.log(hf), seg(t, 2.4, 7.3))
    f = (math.exp(lh) - hf) / (h0 - hf)
    cx = flyer_c[0] + (pan_x - flyer_c[0]) * f
    cy = flyer_c[1] + (pan_y - flyer_c[1]) * f
    # fechamento: aproxima no cupom
    kf = seg(t, 7.7, 9.7)
    cx = lerp(cx, coupon[0], kf)
    cy = lerp(cy, coupon[1] + 0.6, kf)
    lh = lerp(lh, math.log(21), kf)
    # respiracao de camera na mao
    cx += 0.35 * math.sin(t * 1.3) * math.exp(lh) / 300
    cy += 0.25 * math.sin(t * 0.9 + 1) * math.exp(lh) / 300
    return cx, cy, math.exp(lh)

def world_to_screen(cam, p):
    cx, cy, h = cam
    s = H / h
    return (float((p[0] - cx) * s + W / 2), float((p[1] - cy) * s + H / 2))

# ---------- camadas ----------
def render_bg(cam, t):
    cx, cy, h = cam
    w = h * W / H
    s = H / h
    # profundidade de campo: desfoque cresce com o zoom
    dof = (s - 6.3) / (100 - 6.3)
    dof = min(max(dof, 0), 1) ** 0.8
    blur_out = 1.2 + 0.45 * s + 70 * dof         # raio em px da saida
    q = 4 if blur_out > 12 else 2
    sw, sh = W // q, H // q
    box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    bg = salon.transform((sw, sh), Image.EXTENT, box, Image.BICUBIC)
    bg = bg.filter(ImageFilter.GaussianBlur(blur_out / q))
    bg = bg.resize((W, H), Image.BICUBIC)
    return bg, dof

def light_sweep(t):
    """Brilho que atravessa o acrilico e as joias."""
    a = OBJ_NP.copy()
    k = (t - 6.9) / 1.3
    if 0 <= k <= 1:
        pos = lerp(-0.15, 1.15, ease_io(k))
        gl = 0.55 * np.exp(-((SWEEP_COORD - pos) / 0.05) ** 2)
        gl += 0.25 * np.exp(-((SWEEP_COORD - pos + 0.09) / 0.015) ** 2)
        # dourado nas pecas
        a[..., :3] += gl[..., None] * np.array([255, 225, 160]) * (a[..., 3:4] / 255) * 0.6
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

def sparkle(draw_img, center, size, alpha):
    if alpha <= 0.01:
        return
    lay = Image.new('L', (int(size * 4), int(size * 4)), 0)
    d = ImageDraw.Draw(lay)
    c = size * 2
    d.polygon([(c, c - size * 2), (c + size * 0.16, c - size * 0.16), (c + size * 2, c),
               (c + size * 0.16, c + size * 0.16), (c, c + size * 2), (c - size * 0.16, c + size * 0.16),
               (c - size * 2, c), (c - size * 0.16, c - size * 0.16)], fill=255)
    d.ellipse([c - size * 0.35, c - size * 0.35, c + size * 0.35, c + size * 0.35], fill=255)
    lay = lay.filter(ImageFilter.GaussianBlur(max(1, size * 0.06)))
    glow = lay.filter(ImageFilter.GaussianBlur(size * 0.5))
    lay = ImageChops.add(lay, glow)
    lay = lay.point(lambda v: int(v * alpha))
    col = Image.new('RGB', lay.size, (255, 236, 190))
    x, y = int(center[0] - c), int(center[1] - c)
    region = draw_img.crop((x, y, x + lay.size[0], y + lay.size[1]))
    region = ImageChops.add(region, ImageChops.multiply(col, Image.merge('RGB', [lay] * 3)))
    draw_img.paste(region, (x, y))

RNG_ORBS = np.random.default_rng(7)
ORBS = [(RNG_ORBS.uniform(0, 1), RNG_ORBS.uniform(0, 0.65), RNG_ORBS.uniform(60, 190),
         RNG_ORBS.uniform(0.05, 0.16), RNG_ORBS.uniform(-0.012, 0.012)) for _ in range(14)]

def orbs_layer(t, strength):
    lay = Image.new('RGB', (W // 4, H // 4), (0, 0, 0))
    d = ImageDraw.Draw(lay)
    for (x, y, r, a, vx) in ORBS:
        xx = ((x + vx * t) % 1.1 - 0.05) * W / 4
        yy = (y + 0.01 * math.sin(t + x * 9)) * H / 4
        rr = r / 4
        v = int(255 * a * strength)
        d.ellipse([xx - rr, yy - rr, xx + rr, yy + rr], fill=(v, int(v * 0.82), int(v * 0.55)))
    lay = lay.filter(ImageFilter.GaussianBlur(3)).resize((W, H), Image.BICUBIC)
    return lay

YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
VIG = 1 - 0.42 * np.clip(np.hypot((XX - W / 2) / (W * 0.62), (YY - H * 0.48) / (H * 0.6)) - 0.35, 0, 1) ** 1.6
VIG = VIG[..., None]

def render(i):
    t = i / FPS
    cam = camera(t)
    bg, dof = render_bg(cam, t)
    s = H / cam[2]

    # sombra de contato e reflexo no balcao
    corners = [world_to_screen(cam, p) for p in object_corners(t)]
    base_l, base_r = corners[3], corners[2]
    shadow = Image.new('L', (W, H), 0)
    sd = ImageDraw.Draw(shadow)
    bw = abs(base_r[0] - base_l[0])
    bx = (base_l[0] + base_r[0]) / 2
    by = (base_l[1] + base_r[1]) / 2
    sd.ellipse([bx - bw * 0.62, by - bw * 0.035, bx + bw * 0.62, by + bw * 0.06], fill=120)
    shadow = shadow.filter(ImageFilter.GaussianBlur(max(2, bw * 0.03)))
    bg = Image.composite(Image.new('RGB', (W, H), (70, 60, 52)), bg, shadow)

    obj = light_sweep(t)
    Hm = homography(corners, [(0, 0), (OW, 0), (OW, OH), (0, OH)])
    coeffs = (Hm / Hm[2, 2]).flatten()[:8]
    warped = obj.transform((W, H), Image.PERSPECTIVE, tuple(coeffs), Image.BICUBIC)

    # reflexo suave no tampo do balcao
    refl_corners = [corners[3], corners[2],
                    (corners[2][0] + (corners[2][0] - corners[1][0]) * 0.07, corners[2][1] + (corners[2][1] - corners[1][1]) * 0.07),
                    (corners[3][0] + (corners[3][0] - corners[0][0]) * 0.07, corners[3][1] + (corners[3][1] - corners[0][1]) * 0.07)]
    Hr = homography(refl_corners, [(0, OH), (OW, OH), (OW, OH * 0.75), (0, OH * 0.75)])
    refl = OBJ.transform((W, H), Image.PERSPECTIVE, tuple((Hr / Hr[2, 2]).flatten()[:8]), Image.BICUBIC)
    ra = refl.split()[3].point(lambda v: int(v * 0.06))
    refl = refl.filter(ImageFilter.GaussianBlur(max(2, s * 0.14)))
    frame = ImageChops.add(bg, orbs_layer(t, dof))
    frame.paste(refl.convert('RGB'), (0, 0), ra)

    # foco: no inicio o display ainda esta levemente fora de foco
    focus_k = seg(t, 1.2, 4.6)
    if focus_k < 1:
        warped = warped.filter(ImageFilter.GaussianBlur(lerp(2.2, 0, focus_k)))
    frame.paste(warped.convert('RGB'), (0, 0), warped.split()[3])

    # brilhos nas joias (coordenadas no papel)
    for (u, v, t0, size) in [(M + 905, M + 650, 7.35, 1.0), (M + 820, M + 860, 7.55, 0.7),
                             (M + 600, M + 1215, 7.75, 0.8), (M + 790, M + 1265, 8.0, 0.65),
                             (M + 1040, M + 1318, 9.3, 0.8)]:
        a = math.sin(math.pi * min(max((t - t0) / 0.75, 0), 1))
        if a > 0:
            p = world_to_screen(cam, world_point_on_object(t, u, v))
            sparkle(frame, p, size * s * 0.9, a)

    # pulso de luz no cupom no fechamento
    kc = seg(t, 8.6, 9.3)
    if kc > 0:
        c1 = world_to_screen(cam, world_point_on_object(t, M + 300, M + 1310))
        c2 = world_to_screen(cam, world_point_on_object(t, M + 1050, M + 1700))
        glow = Image.new('L', (W, H), 0)
        ImageDraw.Draw(glow).rounded_rectangle([c1[0], c1[1], c2[0], c2[1]], radius=int(s * 1.2), outline=255, width=max(3, int(s * 0.25)))
        glow = glow.filter(ImageFilter.GaussianBlur(s * 0.5))
        pulse = 0.55 + 0.25 * math.sin((t - 8.6) * 4.0)
        glow = glow.point(lambda v: int(v * kc * pulse))
        frame = ImageChops.add(frame, ImageChops.multiply(Image.new('RGB', (W, H), (255, 214, 140)), Image.merge('RGB', [glow] * 3)))


    # gradacao, vinheta, grao
    f = np.asarray(frame).astype(np.float32) / 255
    f = f * np.array([1.035, 1.0, 0.94])
    f = 0.5 + (f - 0.5) * 1.06
    f = f * VIG
    rng = np.random.default_rng(i)
    f += rng.normal(0, 0.012, (H // 2, W // 2, 1)).repeat(2, 0).repeat(2, 1)
    fade = min(1, t / 0.6) * min(1, (DUR - t) / 0.35 + 0.0) if t > DUR - 0.35 else min(1, t / 0.6)
    f *= ease_io(fade) if fade < 1 else 1
    out = Image.fromarray((np.clip(f, 0, 1) * 255).astype(np.uint8))
    out.save(os.path.join(OUT, f'{i:04d}.png'), compress_level=1)
    return i

if __name__ == '__main__':
    only = [int(a) for a in sys.argv[1:]]
    frames = only if only else list(range(N))
    with Pool(os.cpu_count()) as p:
        for _ in p.imap_unordered(render, frames):
            pass
