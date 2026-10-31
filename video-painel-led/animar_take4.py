"""Take 4 — categorias (Mostruário, Saldos, Fora de linha) + "Até 10x sem juros!".
Uso: python3 take4.py fundo.webp cartoes.webp faixa10x.webp saida.mp4 [duracao_s] [largura altura]"""
import numpy as np, subprocess, sys, math
from PIL import Image
from scipy import ndimage as ndi

BG_F, CARDS_F, BAR_F, OUT = sys.argv[1:5]
DUR = float(sys.argv[5]) if len(sys.argv) > 5 else 6.0
W, H = (int(sys.argv[6]), int(sys.argv[7])) if len(sys.argv) > 7 else (1920, 1080)
FPS = 30
K = min(W, H) / 1080                             # escala pelo lado menor: vale para horizontal e vertical
VERT = H > W

b = Image.open(BG_F).convert('RGB')
s = max(W / b.width, H / b.height)
b = b.resize((math.ceil(b.width * s), math.ceil(b.height * s)), Image.LANCZOS)
ox, oy = (b.width - W) // 2, (b.height - H) // 2
bg_im = b.crop((ox, oy, ox + W, oy + H))
yy, xx = np.mgrid[0:H, 0:W]

# ---------- cartões: o PNG já vem posicionado em 1920x1080; separo os três pelo vão entre eles ----------
def fit_canvas(im):
    """Camada que já vem posicionada numa tela: escala inteira para caber no quadro (sem cortar nem distorcer)."""
    im = im.convert('RGBA')
    if im.size == (W, H): return im
    sc = min(W / im.width, H / im.height)
    im = im.resize((round(im.width * sc), round(im.height * sc)), Image.LANCZOS)
    out = Image.new('RGBA', (W, H)); out.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
    return out

cards_im = fit_canvas(Image.open(CARDS_F))
CA = np.asarray(cards_im).astype(np.float32) / 255
lab, n = ndi.label(CA[..., 3] > .5)
boxes = sorted([sl for sl in ndi.find_objects(lab) if sl[1].stop - sl[1].start > 100 * K], key=lambda sl: (sl[0].start // int(60 * K), sl[1].start))   # linha por linha
CARDS = []
pad = int(25 * K)                                   # inclui o brilho dourado em volta da borda
for k, sl in enumerate(boxes):
    own = ndi.binary_dilation(lab == lab[sl][lab[sl] > 0][0], iterations=pad)   # só este cartão e o brilho dele
    y0, y1 = max(0, sl[0].start - pad), min(H, sl[0].stop + pad)
    x0, x1 = max(0, sl[1].start - pad), min(W, sl[1].stop + pad)
    rgba = CA[y0:y1, x0:x1].copy(); rgba[..., 3] *= own[y0:y1, x0:x1]
    CARDS.append(dict(rgba=rgba, face=(rgba[..., 3] > .9).astype(np.float32), cx=(x0 + x1) / 2, cy=(y0 + y1) / 2))
CARDS_DY = 0 if VERT else 20 * K                                   # desce um pouco a fileira para equilibrar com a faixa

# ---------- faixa "ATÉ 10X SEM JUROS!" ----------
BAR_W, BAR_C = 1560, (960, 790)
br = Image.open(BAR_F).convert('RGBA')
if VERT:   # no vertical a arte já traz a posição do selo: mantém onde o designer colocou
    full = fit_canvas(br); bb = full.getbbox(); br = full.crop(bb)
    BAR_W, BAR_C = br.width / K, ((bb[0] + bb[2]) / 2 / K, (bb[1] + bb[3]) / 2 / K)
else:
    br = br.crop(br.getbbox())
    bs = BAR_W * K / br.width
    br = br.resize((round(br.width * bs), round(br.height * bs)), Image.LANCZOS)
BAR = np.asarray(br).astype(np.float32) / 255
BAR_FACE = (BAR[..., 3] > .9).astype(np.float32)
BCX, BCY = BAR_C[0] * K, BAR_C[1] * K

# ---------- utilidades ----------
clamp = lambda v, lo=0., hi=1.: max(lo, min(hi, v))
prog = lambda t, a, b: clamp((t - a) / (b - a))
ease_out = lambda p: 1 - (1 - p) ** 3
def back(p, s=1.8): p -= 1; return p * p * ((s + 1) * p + s) + 1

def paste(canvas, rgba, cx, cy, sx=1., sy=1., alpha=1.):
    if alpha <= .001 or sx <= .01 or sy <= .01: return
    if sx != 1. or sy != 1.:
        pim = Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), 'RGBA')
        pim = pim.resize((max(1, int(rgba.shape[1] * sx)), max(1, int(rgba.shape[0] * sy))), Image.BILINEAR)
        rgba = np.asarray(pim).astype(np.float32) / 255
    hh, ww = rgba.shape[:2]
    x0, y0 = int(round(cx - ww / 2)), int(round(cy - hh / 2))
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x0 + ww), min(H, y0 + hh)
    if X1 <= X0 or Y1 <= Y0: return
    src = rgba[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    al = np.clip(src[..., 3:4], 0, 1) * alpha
    canvas[Y0:Y1, X0:X1] = canvas[Y0:Y1, X0:X1] * (1 - al) + np.clip(src[..., :3], 0, 1) * al

def with_glint(rgba, face, p, width=60, k=.6):
    if not 0 < p < 1: return rgba
    hh, ww = rgba.shape[:2]
    gy, gx = np.mgrid[0:hh, 0:ww]
    band = np.exp(-((gx - (-250 * K + p * (ww + 500 * K)) + (gy - hh / 2) * .5) / (width * K)) ** 2) * face
    out = rgba.copy(); out[..., :3] += (band * k)[..., None]
    return out

gold = np.array([1., .8, .3], np.float32)
warm = np.array([1., .9, .7], np.float32)
vign = (1 - .3 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)).astype(np.float32)[..., None]
# bokeh dourado (círculos desfocados) em três profundidades
rng = np.random.default_rng(21); NB = 38
B = dict(x=rng.uniform(0, W, NB), y=rng.uniform(0, H, NB), r=rng.uniform(14, 48, NB) * K,
         v=rng.uniform(8, 35, NB) * K, ph=rng.uniform(0, 6.28, NB), a=rng.uniform(.06, .2, NB))
NS = 40   # faíscas quando a faixa assenta
sa = rng.uniform(0, 2 * np.pi, NS); sv = rng.uniform(300, 900, NS) * K
SP = dict(vx=np.cos(sa) * sv, vy=np.sin(sa) * sv * .5, ox=rng.uniform(-BAR_W / 2, BAR_W / 2, NS) * K)

# ---------- linha do tempo ----------
T_CARDS, CSTAG = .3, .28     # cartões giram e entram um a um
T_BAR = 1.45                 # faixa se abre do centro
T_BAR_HIT = T_BAR + .35
T_PUNCH = 2.25               # "soco" de destaque na faixa
FOCUS = (3.0, 3.35, 3.7)     # destaque passa por cada cartão
GLINT_BAR, GLINT_CARDS = (2.6, 4.6), 4.1

ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
    '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '14',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', OUT], stdin=subprocess.PIPE)

for f in range(int(FPS * DUR)):
    t = f / FPS
    z = 1.03 + .05 * prog(t, 0, DUR)
    fr = np.asarray(bg_im.transform((W, H), Image.AFFINE, (1 / z, 0, W / 2 - W / (2 * z), 0, 1 / z, H / 2 - H / (2 * z)),
                                    Image.BILINEAR)).astype(np.float32) / 255
    # luz do fundo do galpão "respira"
    fr = fr + (np.exp(-(((xx - W * .52) / (W * .2)) ** 2 + ((yy - H * .08) / (H * .25)) ** 2)) *
               (.12 + .06 * math.sin(t * 2.3)))[..., None] * warm
    for i in range(NB):
        bx = (B['x'][i] + 20 * K * math.sin(t * .7 + B['ph'][i])) % W
        by = (B['y'][i] - B['v'][i] * t) % H
        r = B['r'][i]; x0, y0 = int(bx - r * 1.3), int(by - r * 1.3); sz = int(r * 2.6)
        if 0 <= x0 < W - sz and 0 <= y0 < H - sz:
            d = np.hypot(xx[:sz, :sz] - sz / 2, yy[:sz, :sz] - sz / 2)
            disc = np.clip((r - d) / (2.5 * K), 0, 1) * (.7 + .3 * (d / r))     # borda levemente mais clara
            fr[y0:y0 + sz, x0:x0 + sz] += (disc * B['a'][i] * (.7 + .3 * math.sin(t * 2 + B['ph'][i])))[..., None] * gold

    # cartões: giram no eixo vertical (frente surge do perfil) e descem levemente
    for i, c in enumerate(CARDS):
        t0 = T_CARDS + i * CSTAG
        p = prog(t, t0, t0 + .45)
        if p <= 0: continue
        flip = abs(math.cos((1 - ease_out(p)) * math.pi / 2 * 1.0)) if p < 1 else 1.
        focus = sum(.07 * math.exp(-((t - FOCUS[i]) / .12) ** 2) for _ in [0])
        sc = back(p, 1.4) if p < 1 else 1.
        rgba = with_glint(c['rgba'], c['face'], prog(t, GLINT_CARDS + i * .15, GLINT_CARDS + i * .15 + .6))
        if focus > .005:
            rgba = rgba.copy(); rgba[..., :3] += (c['face'] * focus * 2.5)[..., None] * gold * .4
        paste(fr, rgba, c['cx'], c['cy'] + CARDS_DY - 50 * K * (1 - ease_out(p)) - 3 * K * math.sin(t * 2.2 + i),
              sx=max(.02, flip) * (sc + focus), sy=sc + focus, alpha=min(1, p * 3))
        if 0 < p < .5:   # lampejo quando o cartão "abre"
            fr = fr + (np.exp(-(((xx - c['cx']) / (200 * K)) ** 2 + ((yy - c['cy'] - CARDS_DY) / (130 * K)) ** 2))
                       * .5 * (1 - p * 2))[..., None] * gold

    # faixa: abre do centro para as pontas, quica e leva um "soco" de destaque
    if t >= T_BAR:
        p = prog(t, T_BAR, T_BAR_HIT)
        sx = back(p, 1.6)
        punch = .07 * math.exp(-((t - T_PUNCH) / .09) ** 2)
        rgba = BAR
        for g0 in GLINT_BAR: rgba = with_glint(rgba, BAR_FACE, prog(t, g0, g0 + .7), width=70, k=.5)
        paste(fr, rgba, BCX, BCY - 4 * K * math.sin(t * 2.4) * prog(t, T_BAR_HIT, T_BAR_HIT + .5),
              sx=max(.02, sx) * (1 + punch), sy=(.6 + .4 * ease_out(p)) * (1 + punch), alpha=min(1, p * 4))
        if t >= T_BAR_HIT:
            k = t - T_BAR_HIT
            fr = fr + .35 * math.exp(-k * 12)
            for i in range(NS):
                px = BCX + SP['ox'][i] + SP['vx'][i] * k; py = BCY + SP['vy'][i] * k + 400 * K * k * k
                a = math.exp(-k * 2.5)
                if a > .05 and 6 < px < W - 6 and 6 < py < H - 6:
                    x0, y0 = int(px) - 6, int(py) - 6
                    g = np.exp(-((xx[:12, :12] - 6) ** 2 + (yy[:12, :12] - 6) ** 2) / 6)
                    fr[y0:y0 + 12, x0:x0 + 12] += (g * a)[..., None] * gold
    fr = np.clip(fr * vign, 0, 1)

    out = Image.fromarray((fr * 255).astype(np.uint8))
    if T_BAR_HIT <= t < T_BAR_HIT + .5:
        amp = 14 * K * math.exp(-(t - T_BAR_HIT) * 10)
        sx_, sy_ = amp * math.sin(t * 95), amp * math.cos(t * 77)
        out = out.transform((W, H), Image.AFFINE, (1 / 1.015, 0, W / 2 + sx_ - W / 2.03, 0, 1 / 1.015, H / 2 + sy_ - H / 2.03), Image.BILINEAR)
    ff.stdin.write(out.tobytes())
ff.stdin.close(); ff.wait()
