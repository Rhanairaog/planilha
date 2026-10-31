"""Take 3 — entrada triunfal de "Eletrodomésticos com até 70% de desconto!".
Uso: python3 take3.py fundo.webp chamada.png selo70.webp saida.mp4 [duracao_s] [largura altura]"""
import numpy as np, subprocess, sys, math
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

BG_F, HEAD_F, BADGE_F, OUT = sys.argv[1:5]
DUR = float(sys.argv[5]) if len(sys.argv) > 5 else 6.0
W, H = (int(sys.argv[6]), int(sys.argv[7])) if len(sys.argv) > 7 else (1920, 1080)
FPS = 30
K = min(W, H) / 1080                  # escala pelo lado menor: vale para horizontal e vertical
VERT = H > W

# ---------- fundo (escurecido) ----------
b = Image.open(BG_F).convert('RGB')
s = max(W / b.width, H / b.height)
b = b.resize((math.ceil(b.width * s), math.ceil(b.height * s)), Image.LANCZOS)
ox, oy = (b.width - W) // 2, (b.height - H) // 2
bg_im = b.crop((ox, oy, ox + W, oy + H))
yy, xx = np.mgrid[0:H, 0:W]

# ---------- camadas ----------
HEAD_W, HEAD_CY = 1450, 215          # chamada branca no topo
BADGE_W, BADGE_C = 1380, (960, 665)  # selo 70% abaixo
if VERT:                             # vertical: chamada em duas linhas no terço de cima, selo logo abaixo
    HEAD_W, HEAD_CY = .88 * W / K, H * .33 / K
    BADGE_W, BADGE_C = .9 * W / K, (W / 2 / K, H * .53 / K)

h = Image.open(HEAD_F).convert('RGBA'); h = h.crop(h.getbbox())
hs = HEAD_W * K / h.width
h = h.resize((round(h.width * hs), round(h.height * hs)), Image.LANCZOS)
HA = np.asarray(h).astype(np.float32) / 255
hx0, hy0 = int(W / 2 - h.width / 2), int(HEAD_CY * K - h.height / 2)
lab, n = ndi.label(HA[..., 3] > .5)
sz = ndi.sum(np.ones(lab.shape), lab, range(1, n + 1))
for k, v in enumerate(sz, 1):
    if v < sz.max() * .15: lab[lab == k] = 0                   # acento do É acompanha a letra de baixo
idx = ndi.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
near = lab[idx[0], idx[1]]
LETTERS = []
for k in np.unique(lab[lab > 0]):
    a = HA.copy(); a[..., 3] *= (near == k)
    ys, xs = np.where(a[..., 3] > .01)
    LETTERS.append(dict(rgba=a[ys.min():ys.max() + 1, xs.min():xs.max() + 1],
                        cx=hx0 + (xs.min() + xs.max()) / 2, cy=hy0 + (ys.min() + ys.max()) / 2))
mid = (min(L['cy'] for L in LETTERS) + max(L['cy'] for L in LETTERS)) / 2
LETTERS.sort(key=lambda L: (L['cy'] > mid + 10 * K, L['cx']))   # linha por linha, da esquerda para a direita

bd = Image.open(BADGE_F).convert('RGBA'); bd = bd.crop(bd.getbbox())
bs = BADGE_W * K / bd.width
bd = bd.resize((round(bd.width * bs), round(bd.height * bs)), Image.LANCZOS)
BADGE = np.asarray(bd).astype(np.float32) / 255
# sombra projetada: dá volume a selos chapados (sem relevo) sobre o fundo escuro
pad, off = int(40 * K), (int(10 * K), int(16 * K))
sh = np.zeros((BADGE.shape[0] + 2 * pad, BADGE.shape[1] + 2 * pad, 4), np.float32)
sh[pad + off[1]:pad + off[1] + BADGE.shape[0], pad + off[0]:pad + off[0] + BADGE.shape[1], 3] = BADGE[..., 3]
sh[..., 3] = ndi.gaussian_filter(sh[..., 3], 14 * K) * .75
core = sh[pad:pad + BADGE.shape[0], pad:pad + BADGE.shape[1]]
a = BADGE[..., 3:4]
core[..., :3] = BADGE[..., :3] * a / np.maximum(a + core[..., 3:4] * (1 - a), 1e-4)
core[..., 3:4] = a + core[..., 3:4] * (1 - a)
BADGE = sh
BH, BW = BADGE.shape[:2]
BCX, BCY = BADGE_C[0] * K, BADGE_C[1] * K
bface = (BADGE[..., 3] > .9).astype(np.float32)

# ---------- utilidades ----------
clamp = lambda v, lo=0., hi=1.: max(lo, min(hi, v))
prog = lambda t, a, b: clamp((t - a) / (b - a))
ease_out = lambda p: 1 - (1 - p) ** 3
ease_in = lambda p: p ** 3
def back(p, s=1.8): p -= 1; return p * p * ((s + 1) * p + s) + 1

def paste(canvas, rgba, cx, cy, scale=1., rot=0., alpha=1.):
    if alpha <= .001 or scale <= .01: return
    if scale != 1. or rot:
        pim = Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), 'RGBA')
        pim = pim.resize((max(1, int(rgba.shape[1] * scale)), max(1, int(rgba.shape[0] * scale))), Image.BILINEAR)
        if rot: pim = pim.rotate(rot, Image.BICUBIC, expand=True)
        rgba = np.asarray(pim).astype(np.float32) / 255
    hh, ww = rgba.shape[:2]
    x0, y0 = int(round(cx - ww / 2)), int(round(cy - hh / 2))
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x0 + ww), min(H, y0 + hh)
    if X1 <= X0 or Y1 <= Y0: return
    src = rgba[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    al = np.clip(src[..., 3:4], 0, 1) * alpha
    canvas[Y0:Y1, X0:X1] = canvas[Y0:Y1, X0:X1] * (1 - al) + np.clip(src[..., :3], 0, 1) * al

gold = np.array([1., .8, .3], np.float32)
warm = np.array([1., .93, .78], np.float32)
dist_b = np.hypot(xx - BCX, yy - BCY); ang_b = np.arctan2(yy - BCY, xx - BCX)
# escurecimento: mais forte nas bordas e atrás dos textos
darken = .5 + .25 * np.exp(-(((xx - W / 2) / (W * .45)) ** 2 + ((yy - H * .55) / (H * .5)) ** 2))
darken = darken * (1 - .4 * np.exp(-((yy - HEAD_CY * K) / max(95 * K, h.height * .75)) ** 2))   # faixa escura para a chamada
behind = np.exp(-(((xx - W / 2) / (W * .42)) ** 2 + ((yy - H * .52) / (H * .4)) ** 2))
SPOTS = [(-150 * K, H + 120 * K), (W + 150 * K, H + 120 * K)]   # holofotes nos cantos de baixo
spot_geo = [(np.arctan2(yy - sy, xx - sx), np.hypot(xx - sx, yy - sy)) for sx, sy in SPOTS]
vign = (1 - .3 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)).astype(np.float32)[..., None]

# ---------- linha do tempo ----------
T_HEAD, STAG = .35, .028     # letras da chamada sobem uma a uma
T_HOLD = 1.25                # respiro: luz baixa, holofotes miram o centro
T_BADGE = 1.45               # selo 70% desce de frente para a câmera
T_HIT = T_BADGE + .3         # impacto
BEATS = (2.6, 3.6, 4.6)      # pulsos de "coração" do selo
GLINTS = (2.9, 4.9)

# confete disparado no impacto
rng = np.random.default_rng(5); NC = 260
ca = rng.uniform(0, 2 * np.pi, NC); cv = rng.uniform(500, 1500, NC) * K
C = dict(vx=np.cos(ca) * cv, vy=np.sin(ca) * cv * .8 - 450 * K, w=rng.uniform(8, 18, NC) * K,
         h=rng.uniform(4, 9, NC) * K, spin=rng.uniform(4, 14, NC), ph=rng.uniform(0, 6.28, NC),
         rot=rng.uniform(0, 6.28, NC), col=rng.choice(4, NC))
CCOL = [(255, 214, 0), (255, 255, 255), (255, 150, 20), (255, 236, 140)]
NS = 50
S = dict(x=rng.uniform(0, W, NS), y=rng.uniform(0, H, NS), ph=rng.uniform(0, 6.28, NS), f=rng.uniform(2, 5, NS))

ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
    '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '14',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', OUT], stdin=subprocess.PIPE)

for f in range(int(FPS * DUR)):
    t = f / FPS
    # câmera: travelling lateral lento sobre os produtos
    z = 1.06 + .05 * prog(t, 0, DUR)
    cx, cy = W / 2 - 50 * K + 100 * K * prog(t, 0, DUR), H / 2
    fr = np.asarray(bg_im.transform((W, H), Image.AFFINE, (1 / z, 0, cx - W / (2 * z), 0, 1 / z, cy - H / (2 * z)),
                                    Image.BILINEAR)).astype(np.float32) / 255
    dim = prog(t, 0, .4) * (1 - .35 * prog(t, T_HOLD, T_BADGE) + .35 * prog(t, T_HIT, T_HIT + .3))
    fr = fr * (darken * (1 - .25 * behind * prog(t, T_BADGE, T_HIT)))[..., None] * (.15 + .85 * dim)

    # holofotes: varrem a cena e convergem para o selo no impacto
    conv = ease_out(prog(t, T_HOLD, T_HIT))
    for i, ((sx, sy), (ang, dist)) in enumerate(zip(SPOTS, spot_geo)):
        target = math.atan2(BCY - sy, BCX - sx)
        sweep = target + (.45 if i == 0 else -.45) * math.sin(t * 2.2 + i * 1.5)
        a = sweep * (1 - conv) + (target + .05 * math.sin(t * 1.6 + i)) * conv
        beam = np.exp(-(np.angle(np.exp(1j * (ang - a))) / .09) ** 2) * np.exp(-dist / (2200 * K))
        fr = fr + (beam * (.22 + .1 * conv) * prog(t, 0, .4))[..., None] * warm

    # chamada: letras sobem com rastro de luz
    hb = 1 + .006 * math.sin(t * 3)
    for i, L in enumerate(LETTERS):
        p = prog(t, T_HEAD + i * STAG, T_HEAD + i * STAG + .3)
        if p > 0:
            paste(fr, L['rgba'], L['cx'], L['cy'] + 90 * K * (1 - back(p, 1.5)) - 4 * K * math.sin(t * 3),
                  scale=hb, alpha=min(1, p * 2.5))
            if p < 1:   # brilho no momento em que a letra chega
                fr = fr + (np.exp(-((xx - L['cx']) ** 2 + (yy - L['cy']) ** 2) / (2 * (45 * K) ** 2)) * .5 * (1 - p))[..., None] * warm

    # raios dourados atrás do selo
    rays = (.5 + .5 * np.cos(ang_b * 18 - t * .6)) ** 10 * np.exp(-dist_b / (750 * K))
    fr = fr + (rays * .35 * prog(t, T_HIT, T_HIT + .4))[..., None] * gold

    # selo 70%: vem de trás da câmera, gigante, e crava no lugar
    if t >= T_BADGE:
        p = prog(t, T_BADGE, T_HIT)
        sc = 1 + 3.2 * (1 - ease_in(p)) if p < 1 else 1.
        beat = sum(.05 * math.exp(-((t - bt) / .07) ** 2) + .03 * math.exp(-((t - bt - .2) / .07) ** 2) for bt in BEATS)
        settle = .06 * math.exp(-(t - T_HIT) * 7) * math.sin((t - T_HIT) * 30) if t > T_HIT else 0
        rgba = BADGE
        for g0 in GLINTS:
            gp = prog(t, g0, g0 + .7)
            if 0 < gp < 1:
                gy, gx = np.mgrid[0:BH, 0:BW]
                band = np.exp(-((gx - (-300 * K + gp * (BW + 600 * K)) + (gy - BH / 2) * .55) / (70 * K)) ** 2) * bface
                rgba = rgba.copy(); rgba[..., :3] += (band * .6)[..., None]
        paste(fr, rgba, BCX, BCY - 4 * K * math.sin(t * 2.4) * prog(t, T_HIT, T_HIT + .5),
              scale=sc + beat + settle, rot=-6 * (1 - p) + .6 * math.sin(t * 1.9) * prog(t, T_HIT, T_HIT + .5),
              alpha=min(1, p * 3))

    # impacto: clarão, dupla onda de choque
    if t >= T_HIT:
        k = t - T_HIT
        for d0, spd in ((0, 2400), (.12, 1700)):
            if k > d0:
                ring = np.exp(-((dist_b - (k - d0) * spd * K) / (35 * K)) ** 2) * math.exp(-(k - d0) * 3.5)
                fr = fr + (ring * .7)[..., None] * gold
        fr = fr + .75 * math.exp(-k * 10)
    # cintilas sobre os produtos depois do impacto
    if t > T_HIT:
        for i in range(NS):
            v = max(0, math.sin(t * S['f'][i] + S['ph'][i])) ** 12 * prog(t, T_HIT, T_HIT + .5)
            if v > .02:
                x0, y0 = int(S['x'][i]), int(S['y'][i])
                r = int(26 * K)
                if r <= x0 < W - r and r <= y0 < H - r:
                    gx, gy = xx[:2 * r, :2 * r] - r, yy[:2 * r, :2 * r] - r
                    star = (np.exp(-(gy / 1.5) ** 2 - (gx / (r * .8)) ** 2) + np.exp(-(gx / 1.5) ** 2 - (gy / (r * .8)) ** 2)
                            + np.exp(-(gx ** 2 + gy ** 2) / 18)) * v
                    fr[y0 - r:y0 + r, x0 - r:x0 + r] += star[..., None] * warm
    fr = np.clip(fr * vign, 0, 1)

    out = Image.fromarray((fr * 255).astype(np.uint8))
    # confete (retângulos girando com gravidade; o "achatamento" simula o giro em 3D)
    if t > T_HIT:
        k = t - T_HIT
        dr = ImageDraw.Draw(out)
        px = BCX + C['vx'] * k * np.exp(-k * .9)
        py = BCY + (C['vy'] * k) * np.exp(-k * .9) + 520 * K * k * k
        for i in range(NC):
            if not (-20 < px[i] < W + 20 and -20 < py[i] < H + 20): continue
            r = C['rot'][i] + C['spin'][i] * k * .5
            flat = abs(math.cos(C['spin'][i] * k + C['ph'][i]))
            w2, h2 = C['w'][i] / 2, C['h'][i] / 2 * max(.15, flat)
            cr, sr = math.cos(r), math.sin(r)
            pts = [(px[i] + dx * cr - dy * sr, py[i] + dx * sr + dy * cr) for dx, dy in ((-w2, -h2), (w2, -h2), (w2, h2), (-w2, h2))]
            col = tuple(int(c * (.6 + .4 * flat)) for c in CCOL[C['col'][i]])
            dr.polygon(pts, fill=col)
    # tremor no impacto
    if T_HIT <= t < T_HIT + .7:
        amp = 30 * K * math.exp(-(t - T_HIT) * 8)
        sx, sy = amp * math.sin(t * 95), amp * math.cos(t * 77)
        out = out.transform((W, H), Image.AFFINE, (1 / 1.03, 0, W / 2 + sx - W / 2.06, 0, 1 / 1.03, H / 2 + sy - H / 2.06), Image.BILINEAR)
    ff.stdin.write(out.tobytes())
ff.stdin.close(); ff.wait()
