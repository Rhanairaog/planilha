"""Take 1 — "Alô, Espírito Santo!" para painel de LED, a partir das camadas separadas.
Uso: python3 render2.py fundo.webp titulo.webp faixa.webp saida.mp4 [duracao_s] [largura altura]"""
import numpy as np, subprocess, sys, math
from PIL import Image
from scipy import ndimage as ndi

BG_F, TIT_F, BAN_F, OUT = sys.argv[1:5]
DUR = float(sys.argv[5]) if len(sys.argv) > 5 else 6.0
W, H = (int(sys.argv[6]), int(sys.argv[7])) if len(sys.argv) > 7 else (1920, 1080)
FPS = 30
K = W / 1920                                  # layout desenhado em 1920 de largura

# ---------- fundo (cobre o quadro inteiro) ----------
b = Image.open(BG_F).convert('RGB')
s = max(W / b.width, H / b.height)
b = b.resize((math.ceil(b.width * s), math.ceil(b.height * s)), Image.LANCZOS)
ox, oy = (b.width - W) // 2, (b.height - H) // 2
bg = np.asarray(b.crop((ox, oy, ox + W, oy + H))).astype(np.float32) / 255
yy, xx = np.mgrid[0:H, 0:W]
lum = bg.mean(-1)

# centro do giroflex = blob mais claro do terço esquerdo
gl = ndi.gaussian_filter(lum * (xx < W * .3), 25 * K)
SY, SX = np.unravel_index(gl.argmax(), gl.shape)

# ---------- camadas de texto ----------
def load(f, width, cx, cy):
    im = Image.open(f).convert('RGBA'); im = im.crop(im.getbbox())
    sc = width * K / im.width
    im = im.resize((round(im.width * sc), round(im.height * sc)), Image.LANCZOS)
    a = np.asarray(im).astype(np.float32) / 255
    return a, int(cx * K - im.width / 2), int(cy * K - im.height / 2)

def split(rgba, x0, y0, labels):
    """Divide uma camada em partes: cada pixel vai para a 'face' rotulada mais próxima."""
    lab = labels.copy()
    idx = ndi.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
    near = lab[idx[0], idx[1]]
    out = []
    for k in range(1, lab.max() + 1):
        a = rgba.copy(); a[..., 3] *= (near == k)
        ys, xs = np.where(a[..., 3] > .01)
        if len(ys) == 0: continue
        y_0, y_1, x_0, x_1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        face = (lab[y_0:y_1, x_0:x_1] == k).astype(np.float32)
        out.append(dict(rgba=a[y_0:y_1, x_0:x_1], face=face, x=x0 + x_0, y=y0 + y_0,
                        w=x_1 - x_0, h=y_1 - y_0))
    return out

def sat_of(c): mx, mn = c.max(-1), c.min(-1); return (mx - mn) / (mx + 1e-3)

# título: ALÔ, | ESPÍRITO | SANTO!
T, tx, ty = load(TIT_F, 1400, 1150, 380)
c, a = T[..., :3], T[..., 3]
solid = a > .9
white_face = solid & (sat_of(c) < .18) & (c.mean(-1) > .55)
yel_face = solid & (sat_of(c) > .5) & (c[..., 0] > .7)
lab = np.zeros(a.shape, np.int32)
lab[white_face] = 1
cut = T.shape[1] * (980 - 66) / (1853 - 66)   # vão entre "ESPÍRITO" e "SANTO!"
lab[yel_face & (np.arange(T.shape[1])[None] < cut)] = 2
lab[yel_face & (np.arange(T.shape[1])[None] >= cut)] = 3
ALO, ESP, SAN = split(T, tx, ty, lab)

# faixa: pincelada | sublinhado | palavras pretas
Bn, bx, by = load(BAN_F, 1180, 1110, 770)
c, a = Bn[..., :3], Bn[..., 3]
dark = (a > .5) & (c.max(-1) < .3)
dark = ndi.binary_opening(dark, iterations=1)
comp, n = ndi.label(a > .9)
sizes = ndi.sum(np.ones_like(a), comp, range(1, n + 1))
order = np.argsort(sizes)[::-1] + 1
brush_l, under_l = order[0], order[1]
if ndi.center_of_mass(comp == under_l)[0] < ndi.center_of_mass(comp == brush_l)[0]: brush_l, under_l = under_l, brush_l
# texto preto -> palavras (agrupando colunas)
cols = ndi.binary_dilation(dark.any(0), iterations=int(14 * K))
wl, nw = ndi.label(cols)
words_lab = np.where(dark, wl[None, :], 0).astype(np.int32)
WORDS = split(np.dstack([c, a * dark]), bx, by, words_lab)
# pincelada sem o texto: preenche o lugar das letras com o amarelo vizinho
hole = ndi.binary_dilation(dark, iterations=2)
known = (~hole & (a > .9)).astype(np.float32)
num = np.stack([ndi.gaussian_filter(c[..., i] * known, 6) for i in range(3)], -1)
den = ndi.gaussian_filter(known, 6)[..., None] + 1e-6
cf = np.where(hole[..., None], num / den, c)
blab = np.zeros(a.shape, np.int32); blab[comp == brush_l] = 1; blab[comp == under_l] = 2
blab[hole & (comp == 0)] = 0
BRUSH, UNDER = split(np.dstack([cf, a * (1 - (dark * 1.0))]) if False else np.dstack([cf, a]), bx, by, blab)
for w in WORDS: w['face'] = (w['rgba'][..., 3] > .5).astype(np.float32)

# ---------- utilidades ----------
clamp = lambda v, lo=0., hi=1.: max(lo, min(hi, v))
prog = lambda t, a, b: clamp((t - a) / (b - a))
ease_out = lambda p: 1 - (1 - p) ** 3
def back(p, s=2.0): p -= 1; return p * p * ((s + 1) * p + s) + 1

def paste(canvas, lay, scale=1., alpha=1., dx=0, dy=0, rot=0., xmask=None, extra=None):
    if alpha <= .001: return
    rgba = lay['rgba']
    if extra is not None: rgba = rgba.copy(); rgba[..., :3] += extra
    if scale != 1. or rot:
        pim = Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), 'RGBA')
        pim = pim.resize((max(1, int(lay['w'] * scale)), max(1, int(lay['h'] * scale))), Image.BILINEAR)
        if rot: pim = pim.rotate(rot, Image.BILINEAR, expand=True)
        rgba = np.asarray(pim).astype(np.float32) / 255
    h, w = rgba.shape[:2]
    x0 = int(round(lay['x'] + lay['w'] / 2 + dx - w / 2)); y0 = int(round(lay['y'] + lay['h'] / 2 + dy - h / 2))
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if X1 <= X0 or Y1 <= Y0: return
    src = rgba[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    al = np.clip(src[..., 3:4], 0, 1) * alpha
    if xmask is not None: al = al * xmask[Y0:Y1, X0:X1, None]
    canvas[Y0:Y1, X0:X1] = canvas[Y0:Y1, X0:X1] * (1 - al) + np.clip(src[..., :3], 0, 1) * al

def glint(lay, p, width=50):
    """faixa de brilho diagonal sobre a face da camada (coordenadas locais)."""
    ly, lx = np.mgrid[0:lay['h'], 0:lay['w']]
    pos = -200 * K + p * (lay['w'] + 400 * K)
    band = np.exp(-((lx - pos + ly * .45) / (width * K)) ** 2) * lay['face']
    return (band * .6)[..., None]

orange = np.array([1., .55, .08], np.float32)
ang = np.arctan2(yy - SY, xx - SX); dist = np.hypot(xx - SX, yy - SY)
siren_core = np.exp(-(dist / (70 * K)) ** 2 / 2); siren_wide = np.exp(-(dist / (380 * K)) ** 2 / 2)
# mar: abaixo do horizonte, à esquerda dos pilares da ponte
horizon = int(H * .41)
water = ndi.gaussian_filter(((yy > horizon + 20 * K) & (xx > W * .34) & (xx < W * .72)).astype(np.float32), 30 * K)[..., None]
# luzes da cidade/ponte cintilando
lights = ((lum > .62) & (yy > H * .28) & (yy < H * .5) & (xx > W * .3)).astype(np.float32)
lights = ndi.gaussian_filter(lights, 1.2)[..., None]
lphase = np.random.default_rng(3).uniform(0, 6.28, (H, W)).astype(np.float32)
lphase = ndi.gaussian_filter(lphase, 2)[..., None]
# traços amarelos de alerta já presentes no fundo pulsam junto com o giroflex
rays = ((bg[..., 0] > .8) & (bg[..., 1] > .65) & (bg[..., 2] < .3) & (xx < W * .3) & (dist > 160 * K)).astype(np.float32)
rays = ndi.gaussian_filter(rays, 3 * K)[..., None]
vign = (1 - .22 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)).astype(np.float32)[..., None]

rng = np.random.default_rng(7); NP = 70
P = dict(x=rng.uniform(0, W, NP), y=rng.uniform(0, H, NP), v=rng.uniform(40, 140, NP) * K,
         r=rng.uniform(1.5, 4.5, NP) * K, ph=rng.uniform(0, 6.28, NP))
pg = int(24 * K) | 1

# ---------- linha do tempo ----------
T_ALO, T_ESP, T_SAN = .55, .95, 1.15
T_BRUSH, T_UNDER, T_WORDS = 1.5, 1.85, 1.68
IMPACTS = ((T_ALO + .22, 22, .4), (T_ESP + .2, 18, .3), (T_SAN + .2, 30, .55))

def shake(t):
    s = sum(amp * math.exp(-(t - t0) * 9) for t0, amp, _ in IMPACTS if t >= t0) * K
    return s * math.sin(t * 95), s * math.cos(t * 77)

ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
    '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '14',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', OUT], stdin=subprocess.PIPE)

for f in range(int(FPS * DUR)):
    t = f / FPS
    # fundo: mar ondulando + luzes cintilando
    shift = np.round(np.sin(yy[:, :1] * .09 / K + t * 5.5) * 3 * K).astype(np.int32)
    fr = bg * (1 - water) + bg[yy, np.clip(xx + shift, 0, W - 1)] * water
    fr = fr + lights * fr * .5 * np.sin(lphase * 3 + t * 7)
    fr = fr * (.04 + .96 * ease_out(prog(t, .2, 1.0)))

    # giroflex: duas piscadas de ignição, depois pulso + feixe giratório
    if t < .45: blink = 1. if (.05 < t < .12 or .22 < t < .3 or t > .38) else .1
    else: blink = .35 + .65 * abs(math.sin(math.pi * (t - .45) * 2.4 + 1.57)) ** 4
    d = np.angle(np.exp(1j * (ang - t * 5.2)))
    beam = np.exp(-(d / .28) ** 2) * np.exp(-dist / (900 * K)) * (dist > 40 * K) * prog(t, .4, .8)
    glow = siren_core * 1.2 * blink + siren_wide * .4 * blink + beam * .5
    fr = fr + glow[..., None] * orange + rays * .5 * blink * orange
    fr = fr * (1 + .06 * blink * np.array([1, .7, .4], np.float32))

    # ALÔ,
    p = prog(t, T_ALO, T_ALO + .25)
    if p > 0: paste(fr, ALO, scale=1 + 2.2 * (1 - ease_out(p)), alpha=min(1, p * 3), rot=10 * (1 - p),
                    extra=glint(ALO, prog(t, 4.0, 4.5)) if 4.0 < t < 4.5 else None)
    # ESPÍRITO (vem da esquerda) e SANTO! (vem da direita)
    breathe = 1 + .012 * math.sin((t - 2) * 3.2) * prog(t, 2.0, 2.6)
    for lay, t0, dirx in ((ESP, T_ESP, -1), (SAN, T_SAN, 1)):
        p = prog(t, t0, t0 + .22)
        if p > 0:
            g = None
            for gt in (2.4, 4.6):
                if gt < t < gt + .6: g = glint(lay, prog(t, gt + (.12 if lay is SAN else 0), gt + .48 + (.12 if lay is SAN else 0)))
            paste(fr, lay, scale=(1 + .9 * (1 - ease_out(p))) * breathe, alpha=min(1, p * 4),
                  dx=dirx * 700 * K * (1 - ease_out(p)), extra=g)

    # pincelada amarela (wipe em diagonal), sublinhado e palavras
    p = prog(t, T_BRUSH, T_BRUSH + .35)
    if p > 0:
        edge = BRUSH['x'] - 120 * K + ease_out(p) * (BRUSH['w'] + 260 * K)
        wipe = np.clip((edge - xx - (yy - BRUSH['y']) * .25) / (60 * K), 0, 1).astype(np.float32)
        paste(fr, BRUSH, xmask=wipe if p < 1 else None)
    p = prog(t, T_UNDER, T_UNDER + .3)
    if p > 0:
        edge = UNDER['x'] + ease_out(p) * (UNDER['w'] + 60 * K)
        paste(fr, UNDER, xmask=np.clip((edge - xx) / (40 * K), 0, 1).astype(np.float32) if p < 1 else None)
    for i, wd in enumerate(WORDS):
        p = prog(t, T_WORDS + i * .08, T_WORDS + i * .08 + .22)
        if p > 0:
            g = glint(wd, prog(t, 3.0 + i * .07, 3.35 + i * .07)) if 3.0 < t < 3.8 else None
            paste(fr, wd, scale=max(.01, back(p, 2.6)), alpha=min(1, p * 3), extra=g)

    # faíscas subindo
    if t > .4:
        for i in range(NP):
            px = (P['x'][i] + 25 * K * math.sin(t * 1.3 + P['ph'][i])) % W
            py = (P['y'][i] - P['v'][i] * t) % H
            x0, y0 = int(px) - pg // 2, int(py) - pg // 2
            if 0 <= x0 < W - pg and 0 <= y0 < H - pg:
                g = np.exp(-((xx[:pg, :pg] - pg // 2) ** 2 + (yy[:pg, :pg] - pg // 2) ** 2) / (2 * P['r'][i] ** 2))
                fl = .5 + .5 * math.sin(t * 9 + P['ph'][i])
                fr[y0:y0 + pg, x0:x0 + pg] += (g * .8 * fl * prog(t, .4, 1))[..., None] * orange

    for t0, _, a in IMPACTS:
        if t >= t0: fr = fr + a * math.exp(-(t - t0) * 14)
    fr = np.clip(fr * vign, 0, 1)

    # câmera: aproximação lenta + tremor dos impactos
    z = 1.08 - .06 * ease_out(prog(t, 0, DUR)); sx, sy = shake(t)
    cx, cy = W / 2 + sx, H / 2 + sy
    pim = Image.fromarray((fr * 255).astype(np.uint8)).transform(
        (W, H), Image.AFFINE, (1 / z, 0, cx - W / (2 * z), 0, 1 / z, cy - H / (2 * z)), Image.BILINEAR)
    ff.stdin.write(pim.tobytes())
ff.stdin.close(); ff.wait()
