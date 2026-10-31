"""Take 5 — chamada final: fitas de "evento" rolando, "Faça o seu cadastro no site" e o endereço digitado.
Uso: python3 take5.py fundo.jpg fitas.png chamada.png url.png Anton.ttf saida.mp4 [duracao_s] [largura altura]
As fitas são redesenhadas (mesma fonte, cor e inclinação da arte) para o texto poder rolar sem fim."""
import numpy as np, subprocess, sys, math
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

BG_F, TAPE_F, CTA_F, URL_F, FONT_F, OUT = sys.argv[1:7]
DUR = float(sys.argv[7]) if len(sys.argv) > 7 else 6.0
W, H = (int(sys.argv[8]), int(sys.argv[9])) if len(sys.argv) > 9 else (1920, 1080)
FPS = 30
K = W / 1920
yy, xx = np.mgrid[0:H, 0:W]

def fit(im):
    im = im.convert('RGBA')
    return im if im.size == (W, H) else im.resize((W, H), Image.LANCZOS)

bg = np.asarray(Image.open(BG_F).convert('RGB').resize((W, H), Image.LANCZOS)).astype(np.float32) / 255

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

# ---------- chamada: globo | linha 1 | linha 2 ----------
CTA_SCALE, CTA_DY = .9, -95 * K
cta = np.asarray(fit(Image.open(CTA_F))).astype(np.float32) / 255
ca = cta[..., 3] > .1
cys, cxs = np.where(ca)
text_x0 = cxs.min() + (cxs.max() - cxs.min()) * .21           # o texto começa depois do globo
rows = np.where(ca[:, int(text_x0):].any(1))[0]
gaps = np.where(np.diff(rows) > 5 * K)[0]
LINE_SPLIT = (rows[gaps[0]] + rows[gaps[0] + 1]) / 2 if len(gaps) else H
def cut(mask):
    a = cta.copy(); a[..., 3] *= mask
    ys_, xs_ = np.where(a[..., 3] > .01)
    y0, y1, x0, x1 = ys_.min(), ys_.max() + 1, xs_.min(), xs_.max() + 1
    return dict(rgba=a[y0:y1, x0:x1], x=x0, y=y0, w=x1 - x0, h=y1 - y0, face=(a[y0:y1, x0:x1, 3] > .9).astype(np.float32))
GLOBE = cut(xx < text_x0)
LINE1 = cut((xx >= text_x0) & (yy < LINE_SPLIT))
LINE2 = cut((xx >= text_x0) & (yy >= LINE_SPLIT))
CTA_CX, CTA_CY = (cxs.min() + cxs.max()) / 2, (cys.min() + cys.max()) / 2

# ---------- endereço: caixa branca + letras digitadas ----------
URL_DY = 175 * K
ua = np.asarray(fit(Image.open(URL_F))).astype(np.float32) / 255
dark = (ua[..., 3] > .5) & (ua[..., :3].max(-1) < .4)
lab, n = ndi.label(dark)
objs = ndi.find_objects(lab)
box_l = 1 + int(np.argmax([(s[1].stop - s[1].start) * (s[0].stop - s[0].start) for s in objs]))
ink = ndi.binary_dilation(dark & (lab != box_l), iterations=2) & ~(lab == box_l)
inside = ndi.binary_fill_holes(lab == box_l)
box = ua.copy(); box[..., :3] = np.where(ink[..., None], np.median(ua[inside & ~ink & ~(lab == box_l)][:, :3], 0), ua[..., :3])
uys, uxs = np.where(ua[..., 3] > .1)
UY0, UY1, UX0, UX1 = uys.min(), uys.max() + 1, uxs.min(), uxs.max() + 1
BOX = box[UY0:UY1, UX0:UX1]
TXT = ua.copy(); TXT[..., 3] *= ink; TXT = TXT[UY0:UY1, UX0:UX1]
gl = sorted([objs[k] for k in range(n) if k + 1 != box_l], key=lambda s: s[1].start)
CHAR_END = [s[1].stop - UX0 + 2 for s in gl]                 # x local do fim de cada caractere
URL_CX, URL_CY = (UX0 + UX1) / 2, (UY0 + UY1) / 2 + URL_DY
TXT_TOP, TXT_BOT = min(s[0].start for s in gl) - UY0, max(s[0].stop for s in gl) - UY0

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

def place(part):   # centro da peça já na posição final da composição
    cx = CTA_CX + (part['x'] + part['w'] / 2 - CTA_CX) * CTA_SCALE
    cy = CTA_CY + (part['y'] + part['h'] / 2 - CTA_CY) * CTA_SCALE + CTA_DY
    return cx, cy

gold = np.array([1., .82, .2], np.float32)
vign = (1 - .25 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)).astype(np.float32)[..., None]

# ---------- linha do tempo ----------
T_TAPES = 0.                # fitas entram deslizando
T_GLOBE, T_L1, T_L2 = .45, .6, .95
T_BOX = 1.3
T_TYPE, T_TYPE_END = 1.6, 2.75
T_CLICK = 3.35              # cursor do mouse clica no endereço
GLINT_CTA = 4.0
TAPE_SPEED = (-110 * K, 110 * K)   # rolagem em sentidos opostos

ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
    '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '14',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', OUT], stdin=subprocess.PIPE)

for f in range(int(FPS * DUR)):
    t = f / FPS
    fr = bg * (1 + .08 * math.sin(t * 1.5))
    # brilho dourado que respira atrás do bloco
    fr = fr + (np.exp(-(((xx - W / 2) / (W * .35)) ** 2 + ((yy - H * .47) / (H * .3)) ** 2)) * (.07 + .03 * math.sin(t * 2.2)))[..., None] * gold

    # globo: gira como um globo 3D (achatamento horizontal) e assenta
    p = prog(t, T_GLOBE, T_GLOBE + .5)
    if p > 0:
        cx, cy = place(GLOBE)
        spin = abs(math.cos((1 - ease_out(p)) * math.pi * 1.5)) if p < 1 else 1.
        pulse = .08 * math.exp(-((t - T_CLICK - .05) / .1) ** 2)
        paste(fr, GLOBE['rgba'], cx, cy, sx=CTA_SCALE * back(p) * max(.05, spin) * (1 + pulse),
              sy=CTA_SCALE * back(p) * (1 + pulse), alpha=min(1, p * 3))
    # linhas da chamada: revelação da esquerda para a direita com leve subida
    for L, t0 in ((LINE1, T_L1), (LINE2, T_L2)):
        p = prog(t, t0, t0 + .4)
        if p > 0:
            cx, cy = place(L)
            rgba = L['rgba']
            gp = prog(t, GLINT_CTA + (.15 if L is LINE2 else 0), GLINT_CTA + .7 + (.15 if L is LINE2 else 0))
            if 0 < gp < 1:
                gy, gx = np.mgrid[0:L['h'], 0:L['w']]
                band = np.exp(-((gx - (-200 + gp * (L['w'] + 400)) + gy * .5) / 50) ** 2) * L['face']
                rgba = rgba.copy(); rgba[..., :3] += (band * .6)[..., None]
            hh, ww = rgba.shape[:2]
            lx = np.arange(int(ww * CTA_SCALE))[None, :].repeat(int(hh * CTA_SCALE), 0)
            edge = ease_out(p) * (ww * CTA_SCALE + 60 * K)
            m = np.clip((edge - lx) / (50 * K), 0, 1).astype(np.float32) if p < 1 else None
            paste(fr, rgba, cx, cy + 25 * K * (1 - ease_out(p)), sx=CTA_SCALE, sy=CTA_SCALE, xmask=m)

    # caixa do endereço: abre com quique; letras aparecem como digitadas, com cursor piscando
    p = prog(t, T_BOX, T_BOX + .35)
    if p > 0:
        click = .06 * math.exp(-((t - T_CLICK - .05) / .09) ** 2) - .03 * math.exp(-((t - T_CLICK) / .04) ** 2)
        sx = back(p, 1.5) * (1 + click); sy = (.5 + .5 * ease_out(p)) * (1 + click)
        paste(fr, BOX, URL_CX, URL_CY, sx=sx, sy=sy, alpha=min(1, p * 3))
        if t >= T_TYPE:
            nchar = int(len(CHAR_END) * prog(t, T_TYPE, T_TYPE_END) + .999)
            if nchar > 0:
                cut_x = CHAR_END[nchar - 1]
                m = (np.arange(TXT.shape[1])[None, :] < cut_x).astype(np.float32).repeat(TXT.shape[0], 0)
                paste(fr, TXT, URL_CX, URL_CY, sx=sx, sy=sy, xmask=m if sx == 1 and sy == 1 else None)
            else:
                cut_x = CHAR_END[0] - 45 * K
            # cursor de texto: amarelo, pisca depois de terminar a digitação
            blink_on = t < T_TYPE_END + .1 or (int((t - T_TYPE_END) * 2.5) % 2 == 0)
            if blink_on:
                x0 = int(URL_CX - BOX.shape[1] * sx / 2 + (cut_x + 6 * K) * sx)
                y0 = int(URL_CY - BOX.shape[0] * sy / 2 + TXT_TOP * sy); y1 = int(URL_CY - BOX.shape[0] * sy / 2 + TXT_BOT * sy)
                fr[y0:y1, x0:x0 + max(2, int(5 * K))] = np.array([.98, .78, .05], np.float32)

    # ponteiro do mouse: entra pelo canto, clica no endereço (onda dourada no clique)
    if t >= T_CLICK - .6 and t < DUR:
        k = ease_out(prog(t, T_CLICK - .6, T_CLICK - .05))
        tx, ty = URL_CX + 380 * K, URL_CY + 15 * K
        mx, my = tx + (W * .95 - tx) * (1 - k), ty + (H * 1.05 - ty) * (1 - k)
        if t >= T_CLICK:
            kk = t - T_CLICK
            ring = np.exp(-((np.hypot(xx - tx, yy - ty) - kk * 900 * K) / (10 * K)) ** 2) * math.exp(-kk * 5)
            fr = fr + (ring * .9)[..., None] * gold
    fr = np.clip(fr * vign, 0, 1)
    out = Image.fromarray((fr * 255).astype(np.uint8)).convert('RGBA')

    # fitas por cima de tudo: deslizam para dentro e rolam sem parar
    for i in (0, 1):
        p = ease_out(prog(t, T_TAPES + i * .12, T_TAPES + i * .12 + .5))
        slide = (-1 if i == 0 else 1) * 1.2 * W * (1 - p)
        lay, pos = tape_layer(i, TAPE_SPEED[i] * t, slide)
        tmp = Image.new('RGBA', (W, H)); tmp.paste(lay, pos)   # paste recorta o que sai do quadro
        out.alpha_composite(tmp)

    if t >= T_CLICK - .6:
        press = .85 if 0 <= t - T_CLICK < .12 else 1.
        s = 62 * K * press
        arrow = [(0, 0), (0, 1.0), (.27, .76), (.45, 1.15), (.6, 1.08), (.42, .7), (.75, .7)]
        pts = [(mx + x * s, my + y * s) for x, y in arrow]
        d = ImageDraw.Draw(out)
        d.polygon([(x + 4 * K, y + 5 * K) for x, y in pts], fill=(0, 0, 0, 110))
        d.polygon(pts, fill=(255, 255, 255, 255), outline=(15, 15, 15, 255), width=max(2, int(3 * K)))
    ff.stdin.write(out.convert('RGB').tobytes())
ff.stdin.close(); ff.wait()
