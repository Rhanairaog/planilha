"""Take do local — "Av. 800, S/N - Quadro 18 / TIMS - Carpina" sobre a foto da entrada do TIMS.
Uso: python3 take_local.py fundo.jpg endereco.png saida.mp4 [duracao_s] [largura altura]"""
import numpy as np, subprocess, sys, math
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

BG_F, ADDR_F, OUT = sys.argv[1:4]
DUR = float(sys.argv[4]) if len(sys.argv) > 4 else 6.0
W, H = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (1920, 1080)
FPS = 30
K = W / 1920
yy, xx = np.mgrid[0:H, 0:W]

bg_im = Image.open(BG_F).convert('RGB').resize((W, H), Image.LANCZOS)
bgl = np.asarray(bg_im.convert('L')).astype(np.float32) / 255
PHOTO_TOP = int(np.argmax(bgl.mean(1) > .08))            # onde a foto começa (acima é preto)

# placa do TIMS na foto (medida na arte, em 1920x1080): x 910–1092, y 612–740
SIGN_X, SIGN_Y = 1001 * K, 676 * K
SIGN_TOP, SIGN_W = 612 * K, 182 * K

# ---------- endereço: pino | linha amarela | linha branca ----------
A = np.asarray(Image.open(ADDR_F).convert('RGBA').resize((W, H), Image.LANCZOS)).astype(np.float32) / 255
al = A[..., 3] > .3
mx_, mn_ = A[..., :3].max(-1), A[..., :3].min(-1)
sat = (mx_ - mn_) / (mx_ + 1e-3)
cols = np.where(al.any(0))[0]
PIN_X1 = cols[np.where(np.diff(cols) > 20 * K)[0][0]] + 1          # fim do pino (primeiro vão grande)
ys_, _ = np.where(al)
DY = (PHOTO_TOP - 40 * K) / 2 - (ys_.min() + ys_.max()) / 2       # centraliza o bloco na faixa preta
def cut(mask):
    a = A.copy(); a[..., 3] *= ndi.binary_dilation(mask, iterations=2)
    ys, xs = np.where(a[..., 3] > .01)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return dict(rgba=a[y0:y1, x0:x1], x=x0, y=y0 + DY, w=x1 - x0, h=y1 - y0, face=(a[y0:y1, x0:x1, 3] > .9).astype(np.float32))
PIN = cut(al & (xx < PIN_X1))
LINE1 = cut(al & (xx >= PIN_X1) & (sat > .4))
LINE2 = cut(al & (xx >= PIN_X1) & (mn_ > .75) & (sat < .25))

# ---------- utilidades ----------
clamp = lambda v, lo=0., hi=1.: max(lo, min(hi, v))
prog = lambda t, a, b: clamp((t - a) / (b - a))
ease_out = lambda p: 1 - (1 - p) ** 3
ease_io = lambda p: 3 * p * p - 2 * p ** 3
def back(p, s=1.8): p -= 1; return p * p * ((s + 1) * p + s) + 1
def bounce(p):   # queda com dois quiques (pino de mapa)
    if p < .55: return (p / .55) ** 2
    if p < .85: q = (p - .7) / .15; return 1 - .12 * (1 - q * q)
    q = (p - .925) / .075; return 1 - .03 * (1 - q * q)

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
    a = np.clip(src[..., 3:4], 0, 1) * alpha
    if xmask is not None: a = a * xmask[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0, None]
    canvas[Y0:Y1, X0:X1] = canvas[Y0:Y1, X0:X1] * (1 - a) + np.clip(src[..., :3], 0, 1) * a

def glint(L, p):
    if not 0 < p < 1: return L['rgba']
    gy, gx = np.mgrid[0:L['h'], 0:L['w']]
    band = np.exp(-((gx - (-200 * K + p * (L['w'] + 400 * K)) + gy * .5) / (50 * K)) ** 2) * L['face']
    out = L['rgba'].copy(); out[..., :3] += (band * .55)[..., None]
    return out

yellow = np.array([1., .85, .05], np.float32)
PIN_TIP = (PIN['x'] + PIN['w'] / 2, PIN['y'] + PIN['h'])
# rota tracejada do pino até a placa (curva de Bézier passando por baixo do texto)
P0 = np.array([PIN_TIP[0], PIN_TIP[1] + 25 * K]); P3 = np.array([SIGN_X, SIGN_TOP - 18 * K])
P1 = P0 + np.array([60 * K, (P3[1] - P0[1]) * 1.3]); P2 = np.array([(P0[0] + P3[0]) * .55, P3[1] - 150 * K])
ts = np.linspace(0, 1, 400)[:, None]
ROUTE = ((1 - ts) ** 3 * P0 + 3 * (1 - ts) ** 2 * ts * P1 + 3 * (1 - ts) * ts ** 2 * P2 + ts ** 3 * P3)
seg = np.r_[0, np.cumsum(np.hypot(*np.diff(ROUTE, axis=0).T))]
ROUTE_LEN = seg[-1]

# ---------- linha do tempo ----------
T_PIN = .35                 # pino cai e quica
T_L1, T_L2 = .75, 1.15
T_ROUTE, T_ROUTE_END = 1.6, 2.7
GLINT = 3.6

ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
    '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '14',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', OUT], stdin=subprocess.PIPE)

for f in range(int(FPS * DUR)):
    t = f / FPS
    # câmera de drone: aproximação lenta centrada na placa (a placa fica parada no quadro)
    z = 1. + .1 * ease_io(prog(t, 0, DUR))
    fr = np.asarray(bg_im.transform((W, H), Image.AFFINE, (1 / z, 0, SIGN_X - SIGN_X / z, 0, 1 / z, SIGN_Y - SIGN_Y / z),
                                    Image.BILINEAR)).astype(np.float32) / 255
    fr = fr * ease_out(prog(t, 0, .6))

    # pino: cai do alto, quica, sombra achata; depois ondas de radar na ponta
    p = prog(t, T_PIN, T_PIN + .55)
    if p > 0:
        drop = (1 - bounce(p)) * 320 * K
        squash = 1 + .12 * math.exp(-((p - .55) / .05) ** 2)
        if p > .5:
            k = (t - T_PIN - .3) % 1.3
            for d0 in (0, .35):
                kk = k - d0
                if kk > 0:
                    rr = kk * 140 * K
                    ring = np.exp(-((np.hypot((xx - PIN_TIP[0]) / 2.2, yy - PIN_TIP[1]) - rr / 2.2) / (2.5 * K)) ** 2) * max(0, 1 - kk / .9)
                    fr = fr + (ring * .8)[..., None] * yellow
        paste(fr, PIN['rgba'], PIN['x'] + PIN['w'] / 2, PIN['y'] + PIN['h'] / 2 - drop + PIN['h'] * (1 - 1 / squash) / 2,
              sx=squash, sy=1 / squash, alpha=min(1, p * 4))

    # endereço: revelação da esquerda para a direita; cidade sobe por baixo
    p = prog(t, T_L1, T_L1 + .45)
    if p > 0:
        lx = np.arange(LINE1['w'])[None, :].repeat(LINE1['h'], 0)
        m = np.clip((ease_out(p) * (LINE1['w'] + 60 * K) - lx) / (50 * K), 0, 1).astype(np.float32) if p < 1 else None
        paste(fr, glint(LINE1, prog(t, GLINT, GLINT + .8)), LINE1['x'] + LINE1['w'] / 2, LINE1['y'] + LINE1['h'] / 2, xmask=m)
    p = prog(t, T_L2, T_L2 + .4)
    if p > 0:
        paste(fr, glint(LINE2, prog(t, GLINT + .2, GLINT + 1.)), LINE2['x'] + LINE2['w'] / 2,
              LINE2['y'] + LINE2['h'] / 2 + 30 * K * (1 - ease_out(p)), alpha=ease_out(p))

    # pulso na placa quando a rota chega
    if t >= T_ROUTE_END:
        k = (t - T_ROUTE_END) % 1.5
        rr = SIGN_W * .6 + k * 260 * K
        ring = np.exp(-((np.hypot(xx - SIGN_X, (yy - SIGN_Y) * 1.6) - rr) / (4 * K)) ** 2) * max(0, 1 - k / 1.2)
        glow = np.exp(-(((xx - SIGN_X) / (SIGN_W * .9)) ** 2 + ((yy - SIGN_Y) / (SIGN_W * .7)) ** 2)) * .25 * (.6 + .4 * math.sin(t * 5))
        fr = fr + ((ring * .9 + glow) * prog(t, T_ROUTE_END, T_ROUTE_END + .3))[..., None] * yellow

    out = Image.fromarray((np.clip(fr, 0, 1) * 255).astype(np.uint8))
    # rota tracejada: desenha do pino até a placa; depois os traços "andam" em direção à placa
    if t >= T_ROUTE:
        reach = ease_io(prog(t, T_ROUTE, T_ROUTE_END)) * ROUTE_LEN
        dash, gap = 26 * K, 16 * K
        march = (t - T_ROUTE) * 90 * K
        d = ImageDraw.Draw(out)
        s0 = -((march) % (dash + gap))
        while s0 < reach:
            a, b = max(0, s0), min(reach, s0 + dash)
            if b > a:
                i0, i1 = np.searchsorted(seg, a), np.searchsorted(seg, b)
                pts = [tuple(p) for p in ROUTE[i0:max(i1, i0 + 2)]]
                d.line(pts, fill=(255, 216, 10), width=max(3, int(8 * K)), joint='curve')
            s0 += dash + gap
        # ponta luminosa da rota
        if reach < ROUTE_LEN:
            hx, hy = ROUTE[np.searchsorted(seg, reach) - 1]
            d.ellipse((hx - 9 * K, hy - 9 * K, hx + 9 * K, hy + 9 * K), fill=(255, 245, 200))
    ff.stdin.write(out.tobytes())
ff.stdin.close(); ff.wait()
