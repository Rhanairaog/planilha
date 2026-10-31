"""Take 2 — galpão do outlet + selo "Mega Saldão Engage Eletro Outlet".
Uso: python3 take2.py fundo.webp selo.webp saida.mp4 [duracao_s] [largura altura]
Para painel vertical, passe por exemplo 1080 1920."""
import numpy as np, subprocess, sys, math
from PIL import Image, ImageFilter
from scipy import ndimage as ndi

BG_F, LOGO_F, OUT = sys.argv[1:4]
DUR = float(sys.argv[4]) if len(sys.argv) > 4 else 6.0
W, H = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (1920, 1080)
FPS = 30
K = min(W, H) / 1080                             # escala pelo lado menor: vale para horizontal e vertical
VERT = H > W

# ---------- fundo ----------
b = Image.open(BG_F).convert('RGB')
s = max(W / b.width, H / b.height)
b = b.resize((math.ceil(b.width * s), math.ceil(b.height * s)), Image.LANCZOS)
ox, oy = (b.width - W) // 2, (b.height - H) // 2
bg_im = b.crop((ox, oy, ox + W, oy + H))
if s > 1.5: bg_im = bg_im.filter(ImageFilter.UnsharpMask(radius=2, percent=90, threshold=2))   # fundo pequeno ampliado
yy, xx = np.mgrid[0:H, 0:W]

# ---------- selo ----------
LOGO_H = 880                                   # altura do selo no quadro (em 1080)
LCX, LCY = 960, 545
if VERT:                                       # vertical: selo pela largura, um pouco acima do centro
    LOGO_H = .86 * W / K * 1306 / 1205
    LCX, LCY = W / 2 / K, H * .44 / K
lg = Image.open(LOGO_F).convert('RGBA'); lg = lg.crop(lg.getbbox())
ls = LOGO_H * K / lg.height
LAMPS = [(240, 190), (605, 95), (980, 225)]    # lâmpadas vermelhas, em pixels do selo original
LAMPS = [((x - lg.width / 2) * ls, (y - lg.height / 2) * ls) for x, y in LAMPS]
lg = lg.resize((round(lg.width * ls), round(lg.height * ls)), Image.LANCZOS)
LOGO = np.asarray(lg).astype(np.float32) / 255
LW, LH = lg.size
lface = (LOGO[..., 3] > .9).astype(np.float32)

# ---------- utilidades ----------
clamp = lambda v, lo=0., hi=1.: max(lo, min(hi, v))
prog = lambda t, a, b: clamp((t - a) / (b - a))
ease_out = lambda p: 1 - (1 - p) ** 3
def back(p, s=1.8): p -= 1; return p * p * ((s + 1) * p + s) + 1

def paste(canvas, rgba, cx, cy, scale=1., rot=0., alpha=1.):
    if alpha <= .001 or scale <= .01: return
    if scale != 1. or rot:
        pim = Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), 'RGBA')
        pim = pim.resize((max(1, int(rgba.shape[1] * scale)), max(1, int(rgba.shape[0] * scale))), Image.BILINEAR)
        if rot: pim = pim.rotate(rot, Image.BICUBIC, expand=True)
        rgba = np.asarray(pim).astype(np.float32) / 255
    h, w = rgba.shape[:2]
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if X1 <= X0 or Y1 <= Y0: return
    src = rgba[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    al = np.clip(src[..., 3:4], 0, 1) * alpha
    canvas[Y0:Y1, X0:X1] = canvas[Y0:Y1, X0:X1] * (1 - al) + np.clip(src[..., :3], 0, 1) * al

def zoom(im, z, cx, cy):
    return im.transform((W, H), Image.AFFINE, (1 / z, 0, cx - W / (2 * z), 0, 1 / z, cy - H / (2 * z)), Image.BILINEAR)

red = np.array([1., .12, .05], np.float32)
gold = np.array([1., .78, .35], np.float32)
dist_c = np.hypot(xx - LCX * K, yy - LCY * K); ang_c = np.arctan2(yy - LCY * K, xx - LCX * K)
# halo escuro atrás do selo (contraste) e raios dourados giratórios
backdrop = np.exp(-(dist_c / (520 * K)) ** 2)[..., None].astype(np.float32)
burst_fall = np.exp(-dist_c / (700 * K)).astype(np.float32)
# feixes de sol entrando pelo portão do galpão (alto, à esquerda do centro)
GX, GY = (W * .5, -60 * K) if VERT else (560 * K, -60 * K)
ang_g = np.arctan2(yy - GY, xx - GX); dist_g = np.hypot(xx - GX, yy - GY)
vign = (1 - .25 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)).astype(np.float32)[..., None]

rng = np.random.default_rng(11); NP = 90
P = dict(x=rng.uniform(0, W, NP), y=rng.uniform(0, H, NP), vx=rng.uniform(-15, 15, NP) * K,
         vy=rng.uniform(-30, -8, NP) * K, r=rng.uniform(1.2, 3.5, NP) * K, ph=rng.uniform(0, 6.28, NP))
pg = int(20 * K) | 1

# ---------- linha do tempo ----------
T_LOGO = .3                 # selo entra
T_LAND = T_LOGO + .45       # impacto
GLINTS = (1.7, 3.9)

ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
    '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '14',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', OUT], stdin=subprocess.PIPE)

for f in range(int(FPS * DUR)):
    t = f / FPS
    # câmera sobre o galpão: entrada em "chicote" (zoom com rastro) e depois aproximação lenta
    pin = ease_out(prog(t, 0, .45))
    z = 1.04 + .08 * prog(t, 0, DUR) + .35 * (1 - pin)
    cx, cy = W / 2 + 40 * K * prog(t, 0, DUR), H / 2
    if pin < 1:
        frames = [np.asarray(zoom(bg_im, z * (1 + .06 * k * (1 - pin)), cx, cy)).astype(np.float32) for k in range(4)]
        fr = sum(frames) / 4 / 255
    else:
        fr = np.asarray(zoom(bg_im, z, cx, cy)).astype(np.float32) / 255

    # luz do sol: feixes que respiram + poeira dourada
    beams = (.5 + .5 * np.cos(ang_g * 22 + t * .8)) ** 6 * np.exp(-dist_g / (1300 * K))
    fr = fr + (beams * .18)[..., None] * gold

    # contraste atrás do selo + raios dourados girando
    lp = prog(t, T_LOGO, T_LAND)
    fr = fr * (1 - backdrop * .55 * lp)
    burst = (.5 + .5 * np.cos(ang_c * 14 + t * .7)) ** 8 * burst_fall * prog(t, T_LAND, T_LAND + .5)
    fr = fr + (burst * .3)[..., None] * gold

    # selo: cai girando, passa do tamanho e assenta; depois flutua
    if t >= T_LOGO:
        p = prog(t, T_LOGO, T_LAND)
        sc = back(p) * (1 + .015 * math.sin((t - T_LAND) * 2.6) * prog(t, T_LAND + .3, T_LAND + .8))
        rot = -14 * (1 - ease_out(p)) + .8 * math.sin(t * 1.7) * prog(t, T_LAND, T_LAND + .6)
        lx = LCX * K
        ly = LCY * K - 260 * K * (1 - ease_out(p)) + 7 * K * math.sin(t * 2) * prog(t, T_LAND, T_LAND + .6)
        rgba = LOGO
        for g0 in GLINTS:
            gp = prog(t, g0, g0 + .6)
            if 0 < gp < 1:
                gy, gx = np.mgrid[0:LH, 0:LW]
                band = np.exp(-((gx - (-300 * K + gp * (LW + 600 * K)) + (gy - LH / 2) * .5) / (55 * K)) ** 2) * lface
                rgba = rgba.copy(); rgba[..., :3] += (band * .55)[..., None]
        paste(fr, rgba, lx, ly, scale=sc, rot=rot, alpha=min(1, p * 4))

        # giroflexes do selo: acendem em sequência (esq → centro → dir) e jogam feixe vermelho na cena
        if t >= T_LAND:
            cr, sr = math.cos(math.radians(-rot)), math.sin(math.radians(-rot))
            for i, (dx, dy) in enumerate(LAMPS):
                px = lx + (dx * cr - dy * sr) * sc; py = ly + (dx * sr + dy * cr) * sc
                on = .25 + .75 * max(0., math.cos((t * 4.2 - i / 3) * 2 * math.pi)) ** 3
                d = np.hypot(xx - px, yy - py)
                halo = np.exp(-(d / (60 * K)) ** 2) * 1.1 + np.exp(-d / (260 * K)) * .35
                a = np.arctan2(yy - py, xx - px)
                beam = np.exp(-(np.angle(np.exp(1j * (a - t * 4.5 - i * 2.1))) / .22) ** 2) * np.exp(-d / (1100 * K)) * (d > 50 * K)
                fr = fr + ((halo * on + beam * .22) * prog(t, T_LAND, T_LAND + .2))[..., None] * red

    # poeira dourada flutuando na luz
    for i in range(NP):
        px = (P['x'][i] + P['vx'][i] * t + 18 * K * math.sin(t * .9 + P['ph'][i])) % W
        py = (P['y'][i] + P['vy'][i] * t) % H
        x0, y0 = int(px) - pg // 2, int(py) - pg // 2
        if 0 <= x0 < W - pg and 0 <= y0 < H - pg:
            g = np.exp(-((xx[:pg, :pg] - pg // 2) ** 2 + (yy[:pg, :pg] - pg // 2) ** 2) / (2 * P['r'][i] ** 2))
            fl = .55 + .45 * math.sin(t * 5 + P['ph'][i])
            fr[y0:y0 + pg, x0:x0 + pg] += (g * .55 * fl)[..., None] * gold

    # impacto: onda de choque dourada + clarão
    if t >= T_LAND:
        k = t - T_LAND
        ring = np.exp(-((dist_c - k * 2600 * K) / (40 * K)) ** 2) * math.exp(-k * 4)
        fr = fr + (ring * .8)[..., None] * gold + .5 * math.exp(-k * 14)
    fr = np.clip(fr * vign, 0, 1)

    # tremor de câmera no impacto
    out = Image.fromarray((fr * 255).astype(np.uint8))
    if t >= T_LAND and t - T_LAND < .6:
        amp = 26 * K * math.exp(-(t - T_LAND) * 9)
        sx, sy = amp * math.sin(t * 95), amp * math.cos(t * 77)
        out = out.transform((W, H), Image.AFFINE, (1 / 1.02, 0, W / 2 + sx - W / 2.04, 0, 1 / 1.02, H / 2 + sy - H / 2.04), Image.BILINEAR)
    ff.stdin.write(out.tobytes())
ff.stdin.close(); ff.wait()
