"""Take 1 — "Alô, Espírito Santo" em três batidas que terminam numa composição empilhada (horizontal ou vertical).
Uso: python3 render3.py fundo.webp alo.png espirito_santo.png faixa.png saida.mp4 [duracao_s] [largura altura]
As camadas de texto são PNG 1920x1080 com transparência; só o recorte (bbox) de cada uma é usado."""
import numpy as np, subprocess, sys, math
from PIL import Image
from scipy import ndimage as ndi

BG_F, ALO_F, ES_F, BAN_F, OUT = sys.argv[1:6]
DUR = float(sys.argv[6]) if len(sys.argv) > 6 else 6.0
W, H = (int(sys.argv[7]), int(sys.argv[8])) if len(sys.argv) > 8 else (1920, 1080)
FPS = 30
K = min(W, H) / 1080                          # escala pelo lado menor: vale para horizontal e vertical
VERT = H > W

# ---------- fundo (cobre o quadro inteiro) ----------
b = Image.open(BG_F).convert('RGB')
s = max(W / b.width, H / b.height)
b = b.resize((math.ceil(b.width * s), math.ceil(b.height * s)), Image.LANCZOS)
ox, oy = (b.width - W) // 2, (b.height - H) // 2
bg = np.asarray(b.crop((ox, oy, ox + W, oy + H))).astype(np.float32) / 255
yy, xx = np.mgrid[0:H, 0:W]
lum = bg.mean(-1)
gl = ndi.gaussian_filter(lum * ((yy > H * .5) if VERT else (xx < W * .3)), 25 * K)   # vertical: giroflex embaixo
SY, SX = np.unravel_index(gl.argmax(), gl.shape)          # centro do giroflex

# ---------- camadas ----------
# composição final (em 1920x1080): centro x, centro y e escala sobre o PNG original
LOCK = dict(alo=(1150, 245, 1.1), es=(1150, 470, .85), ban=(1150, 725, .8))
ALO_MID_X = None
if VERT:   # vertical: ESPÍRITO SANTO e a faixa ficam onde a arte colocou; o ALÔ sobe para cima deles
    def art(f):
        im = Image.open(f); return im.getbbox(), W / im.width
    (eb, cs), (bb, bs_) , (ab, as_) = art(ES_F), art(BAN_F), art(ALO_F)
    LOCK['es'] = ((eb[0] + eb[2]) / 2 * cs / K, (eb[1] + eb[3]) / 2 * cs / K, cs / K)
    LOCK['ban'] = ((bb[0] + bb[2]) / 2 * bs_ / K, (bb[1] + bb[3]) / 2 * bs_ / K, bs_ / K)
    alo_h = (ab[3] - ab[1]) * as_
    LOCK['alo'] = ((ab[0] + ab[2]) / 2 * as_ / K, (eb[1] * cs - 4 * K - alo_h / 2) / K, as_ / K)
    ALO_MID_X = W / 2 / K

def load(f, key):
    im = Image.open(f).convert('RGBA'); im = im.crop(im.getbbox())
    cx, cy, sc = LOCK[key]; sc *= K
    im = im.resize((round(im.width * sc), round(im.height * sc)), Image.LANCZOS)
    return np.asarray(im).astype(np.float32) / 255, int(cx * K - im.width / 2), int(cy * K - im.height / 2)

def piece(rgba, x0, y0, face):
    ys, xs = np.where(rgba[..., 3] > .01)
    y_0, y_1, x_0, x_1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return dict(rgba=rgba[y_0:y_1, x_0:x_1], face=face[y_0:y_1, x_0:x_1].astype(np.float32),
                x=x0 + x_0, y=y0 + y_0, w=x_1 - x_0, h=y_1 - y_0)

def split(rgba, x0, y0, lab):
    """Cada pixel com alfa vai para o rótulo mais próximo (sombra e acentos acompanham a letra)."""
    idx = ndi.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
    near = lab[idx[0], idx[1]]
    out = []
    for k in np.unique(lab[lab > 0]):
        a = rgba.copy(); a[..., 3] *= (near == k)
        out.append(piece(a, x0, y0, lab == k))
    # linha por linha: corta no maior salto entre os centros verticais, se ele for maior que meia letra
    cys = sorted(p['y'] + p['h'] / 2 for p in out); hm = np.median([p['h'] for p in out])
    gaps = np.diff(cys) if len(cys) > 1 else np.array([0])
    cut_y = cys[int(np.argmax(gaps))] + gaps.max() / 2 if gaps.max() > .5 * hm else 1e9
    return sorted(out, key=lambda p: (p['y'] + p['h'] / 2 > cut_y, p['x']))

A, ax, ay = load(ALO_F, 'alo')
ALO = piece(A, ax, ay, A[..., 3] > .9)

E, ex, ey = load(ES_F, 'es')
lab, n = ndi.label(E[..., 3] > .9)
sizes = ndi.sum(np.ones(lab.shape), lab, range(1, n + 1))
for k, sz in enumerate(sizes, 1):
    if sz < sizes.max() * .15: lab[lab == k] = 0              # acento e ruído viram parte da letra vizinha
LETTERS = split(E, ex, ey, lab)

Bn, bx, by = load(BAN_F, 'ban')
c, a = Bn[..., :3], Bn[..., 3]
dark = (a > .5) & (c.max(-1) < .35)
# contorno preto da faixa (se houver): o componente escuro que abraça a faixa inteira não é texto
dl, dn = ndi.label(ndi.binary_dilation(dark, iterations=1))
if dn:
    dsl = ndi.find_objects(dl)
    big = [k + 1 for k, sl in enumerate(dsl) if (sl[1].stop - sl[1].start) > .8 * a.shape[1]]
    OUTLINE = np.isin(dl, big)
else:
    OUTLINE = np.zeros_like(dark)
dark = dark & ~OUTLINE
yellow = np.median(c[(a > .9) & ~dark], axis=0)
ink = ndi.binary_dilation((a > .1) & (c.max(-1) < .8) & ~OUTLINE, iterations=2) & ~OUTLINE   # letras + bordas suavizadas
bar = Bn.copy(); bar[..., :3] = np.where(ink[..., None], yellow, c)     # faixa lisa, sem letras
BAR = piece(bar, bx, by, (a > .9) & ~ink)
# palavras: letras ligadas por uma dilatação horizontal curta (menor que o espaço entre palavras)
hs_ = ndi.find_objects(ndi.label(dark)[0]); r = max(1, round(.09 * np.median([sl[0].stop - sl[0].start for sl in hs_])))   # ~espaço entre letras
wl, nw = ndi.label(ndi.binary_dilation(dark, structure=np.ones((1, 2 * r + 1)), iterations=1))
wsz = ndi.sum(np.ones(wl.shape), wl, range(1, nw + 1))
for k, sz in enumerate(wsz, 1):
    if sz < wsz.max() * .1: wl[wl == k] = 0                   # acento do Ú vai para a palavra embaixo
WORDS = split(np.dstack([c, a * dark]), bx, by, np.where(dark, wl, 0))

# ---------- utilidades ----------
clamp = lambda v, lo=0., hi=1.: max(lo, min(hi, v))
prog = lambda t, a, b: clamp((t - a) / (b - a))
ease_out = lambda p: 1 - (1 - p) ** 3
ease_io = lambda p: 3 * p * p - 2 * p ** 3
def back(p, s=2.0): p -= 1; return p * p * ((s + 1) * p + s) + 1

def paste(canvas, lay, scale=1., alpha=1., dx=0, dy=0, rot=0., xmask=None, extra=None):
    if alpha <= .001 or scale <= .01: return
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

def glint(lay, p, offset=0, width=45):
    """faixa de brilho diagonal; offset = posição x da peça dentro do grupo, para o brilho correr contínuo."""
    if not 0 < p < 1: return None
    ly, lx = np.mgrid[0:lay['h'], 0:lay['w']]
    pos = -200 * K + p * (1500 * K) - offset
    band = np.exp(-((lx - pos + ly * .45) / (width * K)) ** 2) * lay['face']
    return (band * .6)[..., None]

orange = np.array([1., .55, .08], np.float32)
ang = np.arctan2(yy - SY, xx - SX); dist = np.hypot(xx - SX, yy - SY)
siren_core = np.exp(-(dist / (70 * K)) ** 2 / 2); siren_wide = np.exp(-(dist / (380 * K)) ** 2 / 2)
water = 0 if VERT else ndi.gaussian_filter(((yy > H * .43) & (xx > W * .34) & (xx < W * .72)).astype(np.float32), 30 * K)[..., None]
lights = 0 * lum if VERT else ((lum > .62) & (yy > H * .28) & (yy < H * .5) & (xx > W * .3)).astype(np.float32)
lights = ndi.gaussian_filter(lights, 1.2)[..., None]
lphase = ndi.gaussian_filter(np.random.default_rng(3).uniform(0, 6.28, (H, W)).astype(np.float32), 2)[..., None]
rays = ((bg[..., 0] > .8) & (bg[..., 1] > .65) & (bg[..., 2] < .3) & ((yy > H * .5) if VERT else (xx < W * .3)) & (dist > 160 * K)).astype(np.float32)
rays = ndi.gaussian_filter(rays, 3 * K)[..., None]
vign = (1 - .22 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)).astype(np.float32)[..., None]
# véu escuro atrás do bloco de texto, para leitura à distância no painel
vbox = ((xx > W * .04) & (xx < W * .96) & (yy > H * .17) & (yy < H * .74)) if VERT else ((xx > 520 * K) & (xx < 1780 * K) & (yy > 120 * K) & (yy < 840 * K))
veil = ndi.gaussian_filter(vbox.astype(np.float32), 90 * K)[..., None]

rng = np.random.default_rng(7); NP = 70
P = dict(x=rng.uniform(0, W, NP), y=rng.uniform(0, H, NP), v=rng.uniform(40, 140, NP) * K,
         r=rng.uniform(1.5, 4.5, NP) * K, ph=rng.uniform(0, 6.28, NP))
pg = int(24 * K) | 1

# ---------- linha do tempo ----------
T_ALO, T_ALO_UP = .45, 1.55          # ALÔ grande no centro, depois sobe para o topo
T_ES, STAG = 1.65, .035              # letras de ESPÍRITO SANTO sobem uma a uma
T_BAR, T_WORDS = 2.75, 2.95          # faixa se estende, palavras saltam
ES_LAND = T_ES + STAG * (len(LETTERS) - 1) + .25
IMPACTS = ((T_ALO + .22, 24, .45), (ES_LAND, 16, .25), (T_WORDS + .09 * (len(WORDS) - 1) + .2, 8, .0))
ALO_BIG, ALO_MID_Y = 1.9 / 1.1, 520  # escala e altura do ALÔ na primeira batida
if VERT: ALO_BIG, ALO_MID_Y = 1.55, H * .42 / K
if ALO_MID_X is None: ALO_MID_X = LOCK['alo'][0]

def shake(t):
    s = sum(amp * math.exp(-(t - t0) * 9) for t0, amp, _ in IMPACTS if t >= t0) * K
    return s * math.sin(t * 95), s * math.cos(t * 77)

ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
    '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '14',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', OUT], stdin=subprocess.PIPE)

for f in range(int(FPS * DUR)):
    t = f / FPS
    shift = np.round(np.sin(yy[:, :1] * .09 / K + t * 5.5) * 3 * K).astype(np.int32)
    fr = bg * (1 - water) + bg[yy, np.clip(xx + shift, 0, W - 1)] * water
    fr = fr + lights * fr * .5 * np.sin(lphase * 3 + t * 7)
    fr = fr * (.04 + .96 * ease_out(prog(t, .2, 1.0)))
    fr = fr * (1 - veil * .35 * prog(t, .4, 1.2))

    # giroflex: duas piscadas de ignição, depois pulso + feixe giratório
    if t < .45: blink = 1. if (.05 < t < .12 or .22 < t < .3 or t > .38) else .1
    else: blink = .35 + .65 * abs(math.sin(math.pi * (t - .45) * 2.4 + 1.57)) ** 4
    d = np.angle(np.exp(1j * (ang - t * 5.2)))
    beam = np.exp(-(d / .28) ** 2) * np.exp(-dist / (900 * K)) * (dist > 40 * K) * prog(t, .4, .8)
    glow = siren_core * 1.2 * blink + siren_wide * .4 * blink + beam * .5
    fr = fr + glow[..., None] * orange + rays * .5 * blink * orange
    fr = fr * (1 + .06 * blink * np.array([1, .7, .4], np.float32))

    breathe = 1 + .01 * math.sin((t - 3.6) * 3.2) * prog(t, 3.6, 4.2)

    # 1) ALÔ — impacto no centro, empurrão lento, sobe para o topo
    p = prog(t, T_ALO, T_ALO + .25)
    if p > 0:
        m = ease_io(prog(t, T_ALO_UP, T_ALO_UP + .4))
        big = ALO_BIG * (1 + .05 * prog(t, T_ALO + .25, T_ALO_UP))
        sc = (big + 1.4 * (1 - ease_out(p))) * (1 - m) + m * 1.0
        dy = (ALO_MID_Y - LOCK['alo'][1]) * K * (1 - m)
        dx = (ALO_MID_X - LOCK['alo'][0]) * K * (1 - m)
        paste(fr, ALO, scale=sc * breathe, alpha=min(1, p * 3), dx=dx, dy=dy, rot=12 * (1 - p),
              extra=glint(ALO, prog(t, 5.0, 5.5), width=35))

    # 2) ESPÍRITO SANTO — letras sobem em cascata e assentam com quique
    for i, L in enumerate(LETTERS):
        p = prog(t, T_ES + i * STAG, T_ES + i * STAG + .25)
        if p > 0:
            off = L['x'] - LETTERS[0]['x']
            paste(fr, L, scale=breathe, alpha=min(1, p * 4), dy=380 * K * (1 - back(p, 1.7)),
                  rot=20 * (1 - ease_out(p)), extra=glint(L, prog(t, 4.0, 4.7), offset=off))

    # 3) faixa — a barra se estende ao longo da inclinação, palavras saltam
    p = prog(t, T_BAR, T_BAR + .3)
    if p > 0:
        edge = BAR['x'] - 40 * K + ease_out(p) * (BAR['w'] + 80 * K)
        wipe = np.clip((edge - xx) / (30 * K), 0, 1).astype(np.float32) if p < 1 else None
        paste(fr, BAR, scale=breathe, xmask=wipe, extra=glint(BAR, prog(t, 4.5, 5.2)))
    for i, wd in enumerate(WORDS):
        p = prog(t, T_WORDS + i * .09, T_WORDS + i * .09 + .22)
        if p > 0: paste(fr, wd, scale=back(p, 2.6) * breathe, alpha=min(1, p * 3))

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
        if t >= t0 and a: fr = fr + a * math.exp(-(t - t0) * 14)
    fr = np.clip(fr * vign, 0, 1)

    z = 1.08 - .06 * ease_out(prog(t, 0, DUR)); sx, sy = shake(t)
    cx, cy = W / 2 + sx, H / 2 + sy
    pim = Image.fromarray((fr * 255).astype(np.uint8)).transform(
        (W, H), Image.AFFINE, (1 / z, 0, cx - W / (2 * z), 0, 1 / z, cy - H / (2 * z)), Image.BILINEAR)
    ff.stdin.write(pim.tobytes())
ff.stdin.close(); ff.wait()
