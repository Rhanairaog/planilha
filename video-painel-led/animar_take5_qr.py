"""Take 5 (versão QR Code) — fitas rolando, QR Code "escaneado" na tela, chamada e endereço digitado.
Uso: python3 take5_qr.py fundo.jpg|preto fitas.png bloco_qr.png[,outra_camada.png]  Anton.ttf saida.mp4 [duracao_s] [largura altura]
O QR Code nunca é distorcido depois de pronto: fica parado, nítido e com margem branca para a leitura."""
import numpy as np, subprocess, sys, math
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

BG_F, TAPE_F, BLOCK_F, FONT_F, OUT = sys.argv[1:6]
DUR = float(sys.argv[6]) if len(sys.argv) > 6 else 6.0
W, H = (int(sys.argv[7]), int(sys.argv[8])) if len(sys.argv) > 8 else (1920, 1080)
FPS = 30
K = min(W, H) / 1080                  # escala pelo lado menor: vale para horizontal e vertical
VERT = H > W
yy, xx = np.mgrid[0:H, 0:W]

def fit(im):
    im = im.convert('RGBA')
    return im if im.size == (W, H) else im.resize((W, H), Image.LANCZOS)

PRETO = BG_F == 'preto'                                      # 'preto' no lugar do arquivo = fundo preto liso, sem brilho
bg = np.zeros((H, W, 3), np.float32) if PRETO else \
    np.asarray(Image.open(BG_F).convert('RGB').resize((W, H), Image.LANCZOS)).astype(np.float32) / 255

# ---------- fitas: medidas tiradas da arte (em 1920x1080) ----------
tape_a = np.asarray(fit(Image.open(TAPE_F))).astype(np.float32) / 255
solid = tape_a[..., 3] > .5
YELLOW = tuple(int(v * 255) for v in np.median(tape_a[solid & (tape_a[..., 0] > .8)][:, :3], 0))
INK = tuple(int(v * 255) for v in np.median(tape_a[solid & (tape_a[..., :3].max(-1) < .35)][:, :3], 0))
ys, xs = np.where(solid)
top, bot = ys < H / 2, ys >= H / 2
def axis(sel):                                               # direção principal (PCA) de cada fita
    v = np.linalg.eigh(np.cov(np.vstack([xs[sel], ys[sel]])))[1][:, 1]
    return v[1] / v[0]
SLOPE = (axis(top) + axis(bot)) / 2
ANG = math.degrees(math.atan(-SLOPE))                        # inclinação (sobe para a direita)
def centerline(sel):                                         # y do eixo da fita em x = W/2
    return float(np.mean(ys[sel] - SLOPE * (xs[sel] - W / 2)))
TAPES = [centerline(top), centerline(bot)]
runs = solid[: H // 2].sum(0)                                # espessura vertical onde a fita está inteira
THICK = float(np.percentile(runs[runs > 0], 90)) * math.cos(math.radians(ANG))

font = ImageFont.truetype(FONT_F, round(41 * K))
ITEMS = ['CADASTRO ANTECIPADO', 'EVENTO PRESENCIAL', 'DESCONTOS']
GAP = 96 * K                                                 # da palavra até o centro da estrela

def star(d, cx, cy, r):                                      # estrela de 4 pontas, como a da arte
    pts = []
    for i in range(8):
        a = i * math.pi / 4 - math.pi / 2
        rr = r if i % 2 == 0 else r * .28
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    d.polygon(pts, fill=INK)

def tape_tile(offset_items=0):
    """Um período do texto da fita (palavra ✦ palavra ✦ palavra ✦), horizontal."""
    th = int(THICK)
    words = ITEMS[offset_items:] + ITEMS[:offset_items]
    widths = [font.getbbox(w)[2] for w in words]
    P = int(sum(widths) + len(words) * 2 * GAP)
    tile = Image.new('RGBA', (P, th), YELLOW + (255,))
    d = ImageDraw.Draw(tile)
    x = 0
    asc = font.getbbox('E')
    ty = (th - (asc[3] - asc[1])) / 2 - asc[1]
    for w, wd in zip(words, widths):
        d.text((x, ty), w, font=font, fill=INK)
        star(d, x + wd + GAP, th / 2, 14 * K)
        x += wd + 2 * GAP
    return tile, P

TILES = [tape_tile(0), tape_tile(1)]
SPAN = int(math.hypot(W, H) + 200 * K)                       # comprimento que cobre o quadro na diagonal

def tape_layer(i, scroll, slide):
    tile, P = TILES[i]
    strip = Image.new('RGBA', (SPAN + P, tile.size[1]))
    for k in range(SPAN // P + 2): strip.paste(tile, (k * P, 0))
    off = int(scroll % P)
    strip = strip.crop((off, 0, off + SPAN, tile.size[1]))
    # sombra suave embaixo da fita
    pad = int(30 * K)
    lay = Image.new('RGBA', (SPAN, tile.size[1] + 2 * pad), (0, 0, 0, 0))
    sh = np.zeros((lay.size[1], SPAN), np.float32); sh[pad + int(8 * K):pad + int(8 * K) + tile.size[1]] = 1
    sh = ndi.gaussian_filter(sh, 10 * K) * .6
    lay = Image.fromarray(np.dstack([np.zeros((lay.size[1], SPAN, 3), np.uint8), (sh * 255).astype(np.uint8)]), 'RGBA')
    lay.alpha_composite(strip, (0, pad))
    rot = lay.rotate(ANG, Image.BICUBIC, expand=True)
    cy = TAPES[i] - SLOPE * slide * math.cos(math.radians(ANG))
    cx = W / 2 + slide * math.cos(math.radians(ANG))
    return rot, (int(cx - rot.size[0] / 2), int(cy - rot.size[1] / 2))

# ---------- bloco: QR Code | linha 1 | linha 2 | endereço ----------
# o bloco pode vir numa camada só ou em várias (separadas por vírgula), todas na mesma tela
blk_im = Image.new('RGBA', (W, H))
for f_ in BLOCK_F.split(','): blk_im.alpha_composite(fit(Image.open(f_)))
blk = np.asarray(blk_im).astype(np.float32) / 255
ba = blk[..., 3] > .1
mx_, mn_ = blk[..., :3].max(-1), blk[..., :3].min(-1)
sat = (mx_ - mn_) / (mx_ + 1e-3)
# o QR é a maior mancha "preto e branco" do bloco
bw = ndi.binary_closing(ba & (sat < .3), iterations=max(2, int(8 * K)))
qlab, qn = ndi.label(bw)
qsl = ndi.find_objects(qlab)[int(np.argmax(ndi.sum(bw, qlab, range(1, qn + 1))))]
QY0, QY1, QX0, QX1 = qsl[0].start, qsl[0].stop, qsl[1].start, qsl[1].stop
QR = blk[QY0:QY1, QX0:QX1].copy()
QR_SHIFT = 0 if VERT else -60 * K                    # no horizontal abre espaço para a margem branca de leitura
QPAD = int(34 * K)                                   # "zona de silêncio" exigida pelos leitores de QR
qm = int(10 * K) + 4                                 # inclui a moldura dourada fina do QR
qbox = np.zeros_like(ba); qbox[max(0, QY0 - qm):QY1 + qm, max(0, QX0 - qm):QX1 + qm] = True
text_area = ba & ~qbox
yel = text_area & (sat > .45)
wht = text_area & (mn_ > .8) & (sat < .2)
def cut(mask):
    a = blk.copy(); a[..., 3] *= mask
    ys_, xs_ = np.where(a[..., 3] > .01)
    y0, y1, x0, x1 = ys_.min(), ys_.max() + 1, xs_.min(), xs_.max() + 1
    return dict(rgba=a[y0:y1, x0:x1], x=x0, y=y0, w=x1 - x0, h=y1 - y0, face=(a[y0:y1, x0:x1, 3] > .9).astype(np.float32))
grow = lambda m: ndi.binary_dilation(m, iterations=2) & text_area
# linhas amarelas da chamada, separadas pelos vãos verticais (acentos e cedilhas ficam com a linha)
rows = np.where(yel.any(1))[0]
br_ = np.where(np.diff(rows) > 5 * K)[0]
bands = [(rows[i0], rows[i1] + 1) for i0, i1 in zip(np.r_[0, br_ + 1], np.r_[br_, len(rows) - 1])]
main = [bd for bd in bands if bd[1] - bd[0] > 15 * K]
edges = [main[0][0] - 1] + [(main[i][1] + main[i + 1][0]) / 2 for i in range(len(main) - 1)] + [H]
LINES = [cut(grow(yel & (yy >= e0) & (yy < e1))) for e0, e1 in zip(edges[:-1], edges[1:])]
URL = cut(grow(wht)) if wht.sum() > 50 * K else None    # endereço escrito é opcional
if URL:
    lab, n = ndi.label(wht)
    CHAR_END = sorted(s[1].stop - URL['x'] + 2 for s in ndi.find_objects(lab))
BLOCK_DY = 0

# ---------- utilidades ----------
clamp = lambda v, lo=0., hi=1.: max(lo, min(hi, v))
prog = lambda t, a, b: clamp((t - a) / (b - a))
ease_out = lambda p: 1 - (1 - p) ** 3
def back(p, s=1.8): p -= 1; return p * p * ((s + 1) * p + s) + 1

def paste(canvas, rgba, cx, cy, sx=1., sy=1., alpha=1., xmask=None):
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
    if xmask is not None: al = al * xmask[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0, None]
    canvas[Y0:Y1, X0:X1] = canvas[Y0:Y1, X0:X1] * (1 - al) + np.clip(src[..., :3], 0, 1) * al


gold = np.array([1., .82, .2], np.float32)
vign = (1 - .25 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)).astype(np.float32)[..., None]
QCX, QCY = (QX0 + QX1) / 2 + QR_SHIFT, (QY0 + QY1) / 2 + BLOCK_DY
QW, QH = QX1 - QX0, QY1 - QY0
CARD = np.ones((QH + 2 * QPAD, QW + 2 * QPAD, 4), np.float32)          # cartão branco com cantos arredondados
cy_, cx_ = np.mgrid[0:CARD.shape[0], 0:CARD.shape[1]]
r = 18 * K
dx = np.maximum(0, np.maximum(r - cx_, cx_ - (CARD.shape[1] - 1 - r))); dy = np.maximum(0, np.maximum(r - cy_, cy_ - (CARD.shape[0] - 1 - r)))
CARD[..., 3] = np.clip(r - np.hypot(dx, dy) + .5, 0, 1)

# ---------- linha do tempo ----------
T_TAPES = 0.
T_CARD = .3                  # cartão branco abre
T_SCAN, T_SCAN_END = .55, 1.35   # QR "impresso" por um laser de cima para baixo
T_L1, T_L2 = 1.0, 1.3
T_TYPE, T_TYPE_END = 1.7, 2.8
T_BRACKETS = 1.45            # cantoneiras de "aponte a câmera" pulsando em volta do QR
GLINT_CTA = 3.6
TAPE_SPEED = (-110 * K, 110 * K)

ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
    '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '14',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', OUT], stdin=subprocess.PIPE)

for f in range(int(FPS * DUR)):
    t = f / FPS
    fr = bg * (1 + .08 * math.sin(t * 1.5))
    if not PRETO: fr = fr + (np.exp(-(((xx - W / 2) / (W * .35)) ** 2 + ((yy - H * .5) / (H * .3)) ** 2)) * (.07 + .03 * math.sin(t * 2.2)))[..., None] * gold

    # cartão branco + QR revelado por um laser que desce
    p = prog(t, T_CARD, T_CARD + .3)
    if p > 0:
        paste(fr, CARD, QCX, QCY, sx=back(p, 1.5), sy=back(p, 1.5), alpha=min(1, p * 3))
    if t >= T_SCAN:
        q = prog(t, T_SCAN, T_SCAN_END)
        line = int(QH * q)
        if line > 0:
            m = np.zeros((QH, QW), np.float32); m[:line] = 1
            paste(fr, QR, QCX, QCY, xmask=m)
        if q < 1:   # feixe do laser com brilho
            ly = int(QCY - QH / 2 + line)
            band = np.exp(-((yy - ly) / (6 * K)) ** 2) * (np.abs(xx - QCX) < QW / 2 + QPAD)
            fr = fr + (band * .9)[..., None] * gold

    # cantoneiras em volta do QR, pulsando como mira de câmera
    if t >= T_BRACKETS:
        p = ease_out(prog(t, T_BRACKETS, T_BRACKETS + .35))
        puls = 1 + .025 * math.sin((t - T_BRACKETS) * 6)
        half = (QW / 2 + QPAD + 12 * K) * (1.4 - .4 * p) * puls
        L_, th = 60 * K, max(3, int(9 * K))
        col = np.array([.99, .8, .05], np.float32)
        for sxn in (-1, 1):
            for syn in (-1, 1):
                cx0, cy0 = QCX + sxn * half, QCY + syn * half
                xa, xb = sorted((int(cx0), int(cx0 - sxn * L_))); ya, yb = sorted((int(cy0), int(cy0 - syn * L_)))
                ty0 = int(cy0) - (th if syn > 0 else 0); tx0 = int(cx0) - (th if sxn > 0 else 0)
                fr[max(0, ty0):ty0 + th, max(0, xa):xb] = fr[max(0, ty0):ty0 + th, max(0, xa):xb] * (1 - p) + col * p
                fr[max(0, ya):yb, max(0, tx0):tx0 + th] = fr[max(0, ya):yb, max(0, tx0):tx0 + th] * (1 - p) + col * p

    # linhas da chamada
    for li, Lr in enumerate(LINES):
        t0 = T_L1 + li * (T_L2 - T_L1)
        p = prog(t, t0, t0 + .4)
        if p > 0:
            rgba = Lr['rgba']
            gp = prog(t, GLINT_CTA + .15 * li, GLINT_CTA + .7 + .15 * li)
            if 0 < gp < 1:
                gy, gx = np.mgrid[0:Lr['h'], 0:Lr['w']]
                band = np.exp(-((gx - (-200 * K + gp * (Lr['w'] + 400 * K)) + gy * .5) / (50 * K)) ** 2) * Lr['face']
                rgba = rgba.copy(); rgba[..., :3] += (band * .6)[..., None]
            lx = np.arange(Lr['w'])[None, :].repeat(Lr['h'], 0)
            edge = ease_out(p) * (Lr['w'] + 60 * K)
            m = np.clip((edge - lx) / (50 * K), 0, 1).astype(np.float32) if p < 1 else None
            paste(fr, rgba, Lr['x'] + Lr['w'] / 2, Lr['y'] + Lr['h'] / 2 + BLOCK_DY + 25 * K * (1 - ease_out(p)), xmask=m)

    # endereço digitado, com cursor amarelo
    if URL and t >= T_TYPE:
        nchar = int(len(CHAR_END) * prog(t, T_TYPE, T_TYPE_END) + .999)
        cut_x = CHAR_END[nchar - 1] if nchar else 0
        if nchar:
            m = (np.arange(URL['w'])[None, :] < cut_x).astype(np.float32).repeat(URL['h'], 0)
            paste(fr, URL['rgba'], URL['x'] + URL['w'] / 2, URL['y'] + URL['h'] / 2 + BLOCK_DY, xmask=m)
        if t < T_TYPE_END + .1 or int((t - T_TYPE_END) * 2.5) % 2 == 0:
            x0 = int(URL['x'] + cut_x + 6 * K); y0 = int(URL['y'] + BLOCK_DY)
            fr[y0:y0 + URL['h'], x0:x0 + max(2, int(5 * K))] = np.array([.98, .78, .05], np.float32)

    fr = np.clip(fr * vign, 0, 1)
    # o QR pronto é recolocado sem vinheta nem brilhos por cima: máximo contraste para a leitura
    if t >= T_SCAN_END:
        paste(fr, CARD, QCX, QCY); paste(fr, QR, QCX, QCY)
    out = Image.fromarray((fr * 255).astype(np.uint8)).convert('RGBA')

    for i in (0, 1):
        p = ease_out(prog(t, T_TAPES + i * .12, T_TAPES + i * .12 + .5))
        slide = (-1 if i == 0 else 1) * 1.2 * W * (1 - p)
        lay, pos = tape_layer(i, TAPE_SPEED[i] * t, slide)
        tmp = Image.new('RGBA', (W, H)); tmp.paste(lay, pos)
        out.alpha_composite(tmp)
    ff.stdin.write(out.convert('RGB').tobytes())
ff.stdin.close(); ff.wait()
