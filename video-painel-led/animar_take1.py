import numpy as np, subprocess, sys, math
from PIL import Image, ImageFilter
from scipy import ndimage as ndi

SRC = sys.argv[1]; OUT = sys.argv[2]
FPS, DUR = 30, float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
img = Image.open(SRC).convert('RGB')
W, H = img.size
im = np.asarray(img).astype(np.float32) / 255
R, G, B = im[..., 0], im[..., 1], im[..., 2]
mx, mn = im.max(-1), im.min(-1); sat = (mx - mn) / (mx + 1e-3)
yy, xx = np.mgrid[0:H, 0:W]
box = lambda x0, y0, x1, y1: (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
dil = lambda m, r: ndi.binary_dilation(m, iterations=r)

# ---------- recorte das camadas ----------
yellow = (R > .75) & (G > .58) & (B < .43) & (R - B > .43)
white = (mn > .59) & (sat < .25)
alo = dil(box(560, 100, 1010, 425) & white, 9) & box(540, 90, 1030, 440)
alo = ndi.binary_fill_holes(ndi.binary_closing(alo, iterations=6))
tit = dil(box(470, 398, 1820, 650) & yellow, 11) & box(460, 385, 1830, 662)
tit = ndi.binary_fill_holes(ndi.binary_closing(tit, iterations=8))
ban = dil(box(530, 630, 1700, 925) & yellow, 4)
ban = ndi.binary_fill_holes(ndi.binary_closing(ban, iterations=14)) & box(520, 620, 1710, 935)
lines = dil((box(415, 300, 530, 470) | box(0, 300, 160, 560)) & yellow, 4)
masks = dict(alo=alo, tit=tit, ban=ban, lines=lines)

# fundo limpo (preenchimento por convolução normalizada)
hole = dil(alo | tit | ban | lines, 6)
known = (~hole).astype(np.float32)
def nfill(sig, step):
    k = known[::step, ::step]; c = im[::step, ::step]
    num = np.stack([ndi.gaussian_filter(c[..., i] * k, sig / step, truncate=6) for i in range(3)], -1)
    den = ndi.gaussian_filter(k, sig / step, truncate=6)[..., None]
    out = np.asarray(Image.fromarray((np.clip(num / (den + 1e-6), 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)) / 255.
    d = np.asarray(Image.fromarray((np.clip(den[..., 0], 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)) / 255.
    return out.astype(np.float32), d.astype(np.float32)[..., None]
fill, _ = nfill(160, 8)
for sig, step in ((60, 4), (20, 2), (6, 1)):
    f2, d = nfill(sig, step)
    w = np.clip(d * 4, 0, 1)
    fill = fill * (1 - w) + f2 * w
bg = np.where(hole[..., None], fill, im)

def layer(m):
    a = ndi.gaussian_filter(m.astype(np.float32), 1.0)
    ys, xs = np.where(m); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    rgba = np.dstack([im[y0:y1, x0:x1], a[y0:y1, x0:x1]])
    return dict(rgba=rgba, x=x0, y=y0, w=x1 - x0, h=y1 - y0)
L = {k: layer(v) for k, v in masks.items()}

# ---------- utilidades ----------
clamp = lambda v, a=0., b=1.: max(a, min(b, v))
def prog(t, a, b): return clamp((t - a) / (b - a))
ease_out = lambda p: 1 - (1 - p) ** 3
def back(p, s=2.2): p -= 1; return p * p * ((s + 1) * p + s) + 1

def paste(canvas, lay, scale=1., alpha=1., dx=0, dy=0, rot=0., xmask=None):
    if alpha <= 0.001: return
    rgba = lay['rgba']
    if scale != 1. or rot:
        pim = Image.fromarray((rgba * 255).astype(np.uint8), 'RGBA')
        nw, nh = max(1, int(lay['w'] * scale)), max(1, int(lay['h'] * scale))
        pim = pim.resize((nw, nh), Image.BILINEAR)
        if rot: pim = pim.rotate(rot, Image.BILINEAR, expand=True)
        rgba = np.asarray(pim).astype(np.float32) / 255
    h, w = rgba.shape[:2]
    cx, cy = lay['x'] + lay['w'] / 2 + dx, lay['y'] + lay['h'] / 2 + dy
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    sx0, sy0 = max(0, -x0), max(0, -y0)
    X0, Y0 = max(0, x0), max(0, y0); X1, Y1 = min(W, x0 + w), min(H, y0 + h)
    if X1 <= X0 or Y1 <= Y0: return
    src = rgba[sy0:sy0 + Y1 - Y0, sx0:sx0 + X1 - X0]
    a = src[..., 3:4] * alpha
    if xmask is not None: a = a * xmask[Y0:Y1, X0:X1, None]
    canvas[Y0:Y1, X0:X1] = canvas[Y0:Y1, X0:X1] * (1 - a) + src[..., :3] * a

def rgrad(cx, cy, r): return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)).astype(np.float32)

SX, SY = 262, 562                       # centro da lâmpada do giroflex
siren_core = rgrad(SX, SY, 70); siren_wide = rgrad(SX, SY, 380)
ang = np.arctan2(yy - SY, xx - SX); dist = np.hypot(xx - SX, yy - SY)
water = (ndi.gaussian_filter((box(1080, 640, W, H) & ~hole).astype(np.float32), 25))[..., None]
orange = np.array([1., .55, .08], np.float32)
rng = np.random.default_rng(7)
NP = 70
parts = dict(x=rng.uniform(0, W, NP), y=rng.uniform(0, H, NP), v=rng.uniform(40, 140, NP),
             r=rng.uniform(1.5, 4.5, NP), ph=rng.uniform(0, 6.28, NP))
vign = (1 - .45 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 2).astype(np.float32)[..., None]

T_ALO, T_TIT, T_BAN = .55, 1.0, 1.45    # entradas
def shake(t):
    s = 0.
    for t0, amp in ((T_ALO + .22, 22), (T_TIT + .22, 30)):
        if t >= t0: s += amp * math.exp(-(t - t0) * 9)
    return s * math.sin(t * 95), s * math.cos(t * 77)

ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
    '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '14',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', OUT], stdin=subprocess.PIPE)

N = int(FPS * DUR)
for f in range(N):
    t = f / FPS
    # fundo: acende a partir do escuro, com o mar ondulando
    shift = (np.sin(yy[:, :1] * .09 + t * 5.5) * 3).astype(np.int32)
    wb = bg[yy, np.clip(xx + shift, 0, W - 1)]
    fr = bg * (1 - water) + wb * water
    fr = fr * (.05 + .95 * ease_out(prog(t, .15, .9)))

    # giroflex: pulsos + feixe giratório
    blink = .35 + .65 * abs(math.sin(math.pi * t * 2.4)) ** 4
    on = prog(t, 0, .12)
    beam_a = t * 5.2
    d = np.angle(np.exp(1j * (ang - beam_a)))
    beam = np.exp(-(d / .28) ** 2) * np.exp(-dist / 900) * (dist > 40)
    glow = on * (siren_core * 1.3 * blink + siren_wide * .35 * blink + beam * .55)
    fr = fr + glow[..., None] * orange
    fr = fr * (1 + .06 * on * blink * np.array([1, .7, .4], np.float32))  # banho de luz laranja

    # traços amarelos de "alerta"
    pl = ease_out(prog(t, T_ALO + .2, T_ALO + .4))
    paste(fr, L['lines'], scale=.6 + .4 * pl, alpha=pl * (.75 + .25 * blink))

    # ALÔ, — impacto
    p = prog(t, T_ALO, T_ALO + .25)
    if p > 0: paste(fr, L['alo'], scale=1 + 2.2 * (1 - ease_out(p)), alpha=min(1, p * 3), rot=8 * (1 - p))
    # ESPÍRITO SANTO! — impacto maior + respiração
    p = prog(t, T_TIT, T_TIT + .25)
    if p > 0:
        breathe = 1 + .012 * math.sin((t - T_TIT) * 3.2) * prog(t, T_TIT + .6, T_TIT + 1.2)
        paste(fr, L['tit'], scale=(1 + 1.6 * (1 - ease_out(p))) * breathe, alpha=min(1, p * 3))
    # faixa — pincelada da esquerda para a direita
    p = prog(t, T_BAN, T_BAN + .4)
    if p > 0:
        edge = L['ban']['x'] - 120 + ease_out(p) * (L['ban']['w'] + 260)
        wipe = np.clip((edge - xx - (yy - 780) * .25) / 80, 0, 1).astype(np.float32)
        paste(fr, L['ban'], scale=.94 + .06 * back(p, 1.6), xmask=wipe)

    # brilho que atravessa o título e a faixa
    for t0, k in ((2.4, 'tit'), (3.0, 'ban'), (4.6, 'tit')):
        p = prog(t, t0, t0 + .55)
        if 0 < p < 1:
            lay = L[k]; pos = lay['x'] - 200 + p * (lay['w'] + 400)
            band = np.exp(-((xx - pos + (yy - lay['y']) * .45) / 45) ** 2)
            m = np.zeros((H, W), np.float32)
            m[lay['y']:lay['y'] + lay['h'], lay['x']:lay['x'] + lay['w']] = lay['rgba'][..., 3]
            fr = fr + (band * m * .55)[..., None]

    # faíscas subindo
    if t > .4:
        for i in range(NP):
            px = (parts['x'][i] + 25 * math.sin(t * 1.3 + parts['ph'][i])) % W
            py = (parts['y'][i] - parts['v'][i] * t) % H
            x0, y0 = int(px) - 12, int(py) - 12
            if 0 <= x0 < W - 24 and 0 <= y0 < H - 24:
                g = np.exp(-((xx[:24, :24] - 12) ** 2 + (yy[:24, :24] - 12) ** 2) / (2 * parts['r'][i] ** 2))
                fl = .5 + .5 * math.sin(t * 9 + parts['ph'][i])
                fr[y0:y0 + 24, x0:x0 + 24] += (g * .8 * fl * prog(t, .4, 1))[..., None] * orange

    # flashes de impacto
    for t0, a in ((T_ALO + .22, .45), (T_TIT + .22, .6)):
        if t >= t0: fr = fr + a * math.exp(-(t - t0) * 14)
    fr = np.clip(fr * vign, 0, 1)

    # câmera: zoom lento de aproximação + tremor
    z = 1.10 - .07 * ease_out(prog(t, 0, DUR))
    sx, sy = shake(t)
    pim = Image.fromarray((fr * 255).astype(np.uint8))
    cx, cy = W / 2 + sx, H / 2 + sy
    pim = pim.transform((W, H), Image.AFFINE, (1 / z, 0, cx - W / (2 * z), 0, 1 / z, cy - H / (2 * z)), Image.BILINEAR)
    ff.stdin.write(pim.tobytes())
    if f % 30 == 0: print(f'{t:.1f}s', flush=True)
ff.stdin.close(); ff.wait()
