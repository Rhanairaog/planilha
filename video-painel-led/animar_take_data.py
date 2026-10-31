"""Take da data — "06, 07 e 08 de novembro / Preços que você nunca viu!" sobre o galpão escurecido.
Uso: python3 take_data.py fundo.webp cartao.png saida.mp4 [duracao_s] [largura altura]"""
import numpy as np, subprocess, sys, math
from PIL import Image
from scipy import ndimage as ndi

BG_F, CARD_F, OUT = sys.argv[1:4]
DUR = float(sys.argv[4]) if len(sys.argv) > 4 else 6.0
W, H = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (1920, 1080)
FPS = 30
K = min(W, H) / 1080                  # escala pelo lado menor: vale para horizontal e vertical
yy, xx = np.mgrid[0:H, 0:W]

b = Image.open(BG_F).convert('RGB')
s = max(W / b.width, H / b.height)
b = b.resize((math.ceil(b.width * s), math.ceil(b.height * s)), Image.LANCZOS)
ox, oy = (b.width - W) // 2, (b.height - H) // 2
bg_im = b.crop((ox, oy, ox + W, oy + H))

# ---------- cartão: separa a moldura vazia de cada elemento ----------
cim = Image.open(CARD_F).convert('RGBA')
if cim.size != (W, H): cim = cim.resize((W, H), Image.LANCZOS)
C = np.asarray(cim).astype(np.float32) / 255
ca = C[..., 3] > .5
cys, cxs = np.where(ca)
CX0, CY0, CX1, CY1 = cxs.min(), cys.min(), cxs.max() + 1, cys.max() + 1
border = int(22 * K)
inner = np.zeros_like(ca); inner[CY0 + border:CY1 - border, CX0 + border:CX1 - border] = True
content = inner & (C[..., :3].max(-1) > .3)
hole = ndi.binary_dilation(content, iterations=int(4 * K) + 1) & inner
known = (inner & ~hole).astype(np.float32)
fill = C[..., :3].copy()
for sig in (60, 20, 6):     # preenchimento por convolução normalizada (o fundo do cartão é um degradê suave)
    num = np.stack([ndi.gaussian_filter(C[..., i] * known, sig * K) for i in range(3)], -1)
    den = ndi.gaussian_filter(known, sig * K)[..., None]
    f2 = num / (den + 1e-6); w = np.clip(den * 4, 0, 1)
    fill = np.where(hole[..., None], fill * (1 - w) + f2 * w, C[..., :3])
EMPTY = C.copy(); EMPTY[..., :3] = np.where(hole[..., None], fill, C[..., :3])
EMPTY = EMPTY[CY0:CY1, CX0:CX1]
CARD_CX, CARD_CY = (CX0 + CX1) / 2, (CY0 + CY1) / 2

def region(mask):
    a = C.copy(); a[..., 3] = a[..., 3] * ndi.binary_dilation(mask, iterations=2)
    ys, xs = np.where(a[..., 3] > .01)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    # dentro do cartão, o "alfa" do elemento é o quanto ele se destaca do fundo do cartão
    lum = np.clip((a[..., :3].max(-1) - fill.max(-1)) / .35, 0, 1) * (a[..., 3] > .01)
    out = np.dstack([a[..., :3], lum])[y0:y1, x0:x1]
    return dict(rgba=out, x=x0, y=y0, w=x1 - x0, h=y1 - y0, face=(lum[y0:y1, x0:x1] > .6).astype(np.float32))

lab, n = ndi.label(content)
objs = ndi.find_objects(lab)
area = lambda sl: (sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start)
# ícone: o maior contorno (o calendário) + tudo que está dentro dele ou logo acima (as argolas)
ib = objs[int(np.argmax([area(sl) for sl in objs]))]
ih = ib[0].stop - ib[0].start
in_icon = lambda sl: sl[1].start >= ib[1].start - 2 and sl[1].stop <= ib[1].stop + 2 and \
    sl[0].start >= ib[0].start - ih * .2 and sl[0].stop <= ib[0].stop + 2
ICON_IDS = [k + 1 for k, sl in enumerate(objs) if in_icon(sl)]
ICON = region(np.isin(lab, ICON_IDS))
# divisória: componente muito mais largo que alto
div_ids = [k + 1 for k, sl in enumerate(objs) if (sl[1].stop - sl[1].start) > 15 * (sl[0].stop - sl[0].start)]
DIV = region(np.isin(lab, div_ids))
DIV_Y = DIV['y']
text = content & ~np.isin(lab, ICON_IDS + div_ids)
# linhas de texto acima da divisória (separadas pelos vãos na projeção vertical)
def rows_of(mask):
    r = np.where(mask.any(1))[0]
    if len(r) == 0: return []
    br = np.where(np.diff(r) > 3 * K + 1)[0]
    return [(r[i0], r[i1] + 1) for i0, i1 in zip(np.r_[0, br + 1], np.r_[br, len(r) - 1])]
above = text & (yy < DIV_Y)
rows = [rw for rw in rows_of(above) if rw[1] - rw[0] > 12 * K]      # ignora acentos e vírgulas soltas
yellow = (C[..., 0] > .7) & (C[..., 2] < .45) & content
GROUPS, POPS, MONTH_LETTERS = [], [], []
for y0, y1 in rows:
    # junta acentos/descendentes que ficaram fora da faixa da linha
    rmask = above & np.isin(lab, np.unique(lab[y0:y1][above[y0:y1]]))
    rh = y1 - y0
    words, nw = ndi.label(ndi.binary_dilation(rmask, structure=np.ones((1, 2 * max(1, int(rh * .09)) + 1))))
    if (yellow & rmask).sum() > 20:          # linha das datas: números giram, palavras brancas só aparecem
        for k in range(1, nw + 1):
            wm = rmask & (words == k)
            (GROUPS if (yellow & wm).sum() > .3 * wm.sum() else POPS).append(region(wm))
    else:                                     # linha do mês: letras sobem uma a uma
        MONTH_LETTERS += [region(lab == k) for k in np.unique(lab[rmask]) if k]
GROUPS.sort(key=lambda g: (g['y'], g['x'])); POPS.sort(key=lambda g: (g['y'], g['x']))
MONTH_LETTERS.sort(key=lambda r: (r['y'] // max(1, int(40 * K)), r['x']))
BOTTOM = region(text & (yy > DIV_Y))

# ---------- utilidades ----------
clamp = lambda v, lo=0., hi=1.: max(lo, min(hi, v))
prog = lambda t, a, b: clamp((t - a) / (b - a))
ease_out = lambda p: 1 - (1 - p) ** 3
def back(p, s=1.8): p -= 1; return p * p * ((s + 1) * p + s) + 1

def paste(canvas, rgba, cx, cy, sx=1., sy=1., alpha=1., clip=None):
    """clip = (y0, y1) em coordenadas do quadro: só desenha dentro dessa faixa (efeito de janela)."""
    if alpha <= .001 or sx <= .01 or sy <= .01: return
    if sx != 1. or sy != 1.:
        pim = Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), 'RGBA')
        pim = pim.resize((max(1, int(rgba.shape[1] * sx)), max(1, int(rgba.shape[0] * sy))), Image.BILINEAR)
        rgba = np.asarray(pim).astype(np.float32) / 255
    hh, ww = rgba.shape[:2]
    x0, y0 = int(round(cx - ww / 2)), int(round(cy - hh / 2))
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x0 + ww), min(H, y0 + hh)
    if clip: Y0, Y1 = max(Y0, int(clip[0])), min(Y1, int(clip[1]))
    if X1 <= X0 or Y1 <= Y0: return
    src = rgba[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    al = np.clip(src[..., 3:4], 0, 1) * alpha
    canvas[Y0:Y1, X0:X1] = canvas[Y0:Y1, X0:X1] * (1 - al) + np.clip(src[..., :3], 0, 1) * al

gold = np.array([1., .8, .2], np.float32)
vign = (1 - .3 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)).astype(np.float32)[..., None]
# perímetro da moldura (para a luz que corre pela borda)
PW, PH = CX1 - CX0, CY1 - CY0
PERIM = 2 * (PW + PH)
def perim_point(d):
    d %= PERIM
    if d < PW: return CX0 + d, CY0
    d -= PW
    if d < PH: return CX1, CY0 + d
    d -= PH
    if d < PW: return CX1 - d, CY1
    return CX0, CY1 - (d - PW)

rng = np.random.default_rng(31); NP = 60
P = dict(x=rng.uniform(0, W, NP), y=rng.uniform(0, H, NP), v=rng.uniform(15, 45, NP) * K, r=rng.uniform(1.2, 3, NP) * K, ph=rng.uniform(0, 6.28, NP))
pg = int(18 * K) | 1

# ---------- linha do tempo ----------
T_CARD = .25                 # cartão se abre de uma linha dourada
T_ICON = .7
T_REELS, REEL_STAG = .85, .17    # datas giram como rolos de caça-níquel e param em sequência
T_MONTH, MSTAG = 1.75, .045      # letras de NOVEMBRO sobem por uma "janela"
T_DIV = 2.35
T_BOTTOM = 2.5
T_LIGHT = 2.9                # luz dourada corre pela moldura
GLINT = 3.8

ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
    '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '14',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', OUT], stdin=subprocess.PIPE)

for f in range(int(FPS * DUR)):
    t = f / FPS
    z = 1.04 + .05 * prog(t, 0, DUR)
    cxp = W / 2 + 40 * K * (prog(t, 0, DUR) - .5)
    fr = np.asarray(bg_im.transform((W, H), Image.AFFINE, (1 / z, 0, cxp - W / (2 * z), 0, 1 / z, H / 2 - H / (2 * z)),
                                    Image.BILINEAR)).astype(np.float32) / 255
    fr = fr * (.1 + .9 * ease_out(prog(t, 0, .5)))
    for i in range(NP):
        px = (P['x'][i] + 15 * K * math.sin(t + P['ph'][i])) % W; py = (P['y'][i] - P['v'][i] * t) % H
        x0, y0 = int(px) - pg // 2, int(py) - pg // 2
        if 0 <= x0 < W - pg and 0 <= y0 < H - pg:
            g = np.exp(-((xx[:pg, :pg] - pg // 2) ** 2 + (yy[:pg, :pg] - pg // 2) ** 2) / (2 * P['r'][i] ** 2))
            fr[y0:y0 + pg, x0:x0 + pg] += (g * .5 * (.6 + .4 * math.sin(t * 4 + P['ph'][i])))[..., None] * gold
    fr = np.clip(fr * vign, 0, 1)

    # cartão: nasce como uma linha dourada e abre na vertical
    p = prog(t, T_CARD, T_CARD + .4)
    if p > 0:
        sy = max(.012, back(p, 1.3)) if p > .15 else .012
        sx = ease_out(prog(t, T_CARD, T_CARD + .15))
        paste(fr, EMPTY, CARD_CX, CARD_CY, sx=sx, sy=sy)
        if p < .5:
            fr = fr + (np.exp(-((yy - CARD_CY) / (8 * K)) ** 2) * (np.abs(xx - CARD_CX) < PW / 2 * sx) * (1 - p * 2))[..., None] * gold

    # ícone do calendário: cai e quica; depois os quadradinhos piscam em onda
    p = prog(t, T_ICON, T_ICON + .4)
    if p > 0:
        rgba = ICON['rgba']
        if t > T_REELS + 3 * REEL_STAG + .3:
            wave = (.5 + .5 * np.sin(t * 5 - (np.arange(ICON["w"])[None, :] / ICON["w"]) * 6)) * ICON['face']
            rgba = rgba.copy(); rgba[..., :3] += (wave * .25)[..., None]
        paste(fr, rgba, ICON['x'] + ICON['w'] / 2, ICON['y'] + ICON['h'] / 2 - 120 * K * (1 - back(p, 1.6)), alpha=min(1, p * 3))

    # datas: cada grupo gira dentro de uma janela e para, um depois do outro
    for i, g in enumerate(GROUPS):
        t_stop = T_REELS + .45 + i * REEL_STAG
        if t < T_REELS + i * .05: continue
        spin = 1 - ease_out(prog(t, T_REELS + i * .05, t_stop))     # 1 = girando rápido, 0 = parado
        step = g['h'] + 20 * K
        off = (spin * 6 * step) % step                             # passa por ~6 cópias antes de parar
        rgba = g['rgba']
        if spin > .05:   # borrão vertical de movimento
            rgba = rgba.copy(); rgba[..., 3] = ndi.uniform_filter1d(rgba[..., 3], max(1, int(spin * 30 * K)), axis=0)
        win = (g['y'] - 4 * K, g['y'] + g['h'] + 4 * K)
        for k in (-1, 0):
            paste(fr, rgba, g['x'] + g['w'] / 2, g['y'] + g['h'] / 2 + off + k * step, clip=win)
        if 0 < t - t_stop < .25:   # estalo de luz quando o rolo trava
            fr = fr + (np.exp(-(((xx - g['x'] - g['w'] / 2) / (90 * K)) ** 2 + ((yy - g['y'] - g['h'] / 2) / (70 * K)) ** 2))
                       * .45 * (1 - (t - t_stop) / .25))[..., None] * gold
    # palavras brancas da linha das datas ("E", "DE"): só aparecem com um pulo
    for i, g in enumerate(POPS):
        p = prog(t, T_REELS + .5 + i * REEL_STAG, T_REELS + .8 + i * REEL_STAG)
        if p > 0: paste(fr, g['rgba'], g['x'] + g['w'] / 2, g['y'] + g['h'] / 2, sx=back(p, 2.2), sy=back(p, 2.2), alpha=min(1, p * 3))

    # NOVEMBRO: letras sobem de baixo por uma janela, em sequência
    for i, L in enumerate(MONTH_LETTERS):
        p = prog(t, T_MONTH + i * MSTAG, T_MONTH + i * MSTAG + .3)
        if p > 0:
            gp = prog(t, GLINT + i * .04, GLINT + .3 + i * .04)
            rgba = L['rgba']
            if 0 < gp < 1:
                rgba = rgba.copy(); rgba[..., :3] += (L['face'] * .5 * math.sin(gp * math.pi))[..., None] * gold
            paste(fr, rgba, L['x'] + L['w'] / 2, L['y'] + L['h'] / 2 + L['h'] * (1 - back(p, 1.4)),
                  clip=(L['y'] - 4 * K, L['y'] + L['h'] + 4 * K))

    # divisória: desenha do centro para as pontas
    p = ease_out(prog(t, T_DIV, T_DIV + .35))
    if p > 0: paste(fr, DIV['rgba'], DIV['x'] + DIV['w'] / 2, DIV['y'] + DIV['h'] / 2, sx=max(.01, p))

    # frase: entra com as letras "abertas" (espaçamento maior) e se fecha no lugar
    p = prog(t, T_BOTTOM, T_BOTTOM + .5)
    if p > 0:
        paste(fr, BOTTOM['rgba'], BOTTOM['x'] + BOTTOM['w'] / 2, BOTTOM['y'] + BOTTOM['h'] / 2,
              sx=1 + .18 * (1 - ease_out(p)), alpha=ease_out(p))

    # luz dourada correndo pela moldura (dois cometas em lados opostos)
    if t >= T_LIGHT:
        k = prog(t, T_LIGHT, T_LIGHT + .4)
        for head0 in (0, PERIM / 2):
            for j in range(10):
                d = head0 + (t - T_LIGHT) * 1400 * K - j * 22 * K
                px, py = perim_point(d)
                r = int(30 * K); x0, y0 = int(px) - r, int(py) - r
                if 0 <= x0 < W - 2 * r and 0 <= y0 < H - 2 * r:
                    gg = np.exp(-((xx[:2 * r, :2 * r] - r) ** 2 + (yy[:2 * r, :2 * r] - r) ** 2) / (2 * (7 * K) ** 2))
                    fr[y0:y0 + 2 * r, x0:x0 + 2 * r] += (gg * .7 * (1 - j / 10) * k)[..., None] * np.array([1, .9, .55], np.float32)

    ff.stdin.write((np.clip(fr, 0, 1) * 255).astype(np.uint8).tobytes())
ff.stdin.close(); ff.wait()
