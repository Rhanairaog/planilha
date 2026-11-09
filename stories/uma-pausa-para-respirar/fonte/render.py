"""Renderiza o story 1080x1920 (15 s, 30 fps) com as linhas em movimento."""
import json, subprocess, sys
import numpy as np, cv2
from PIL import Image

I, L, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PREVIEW = len(sys.argv) > 4  # gera só alguns quadros-chave
W, H, FPS, DUR = 1080, 1920, 30, 15.0

f32 = lambda im: np.asarray(im).astype(np.float32) / 255
bgA = f32(Image.open(I + '1.webp').convert('RGB'))
bgB = f32(Image.open(L + 'bg5_clean.png').convert('RGB'))
mat = f32(Image.open(L + 'mat_mask.png').convert('L'))
alpha_of = lambda f: f32(Image.open(L + f))[..., 3]
t_orange, t_para, t_resp = alpha_of('title_orange.png'), alpha_of('title_para.png'), alpha_of('title_respirar.png')
cta = alpha_of('cta.png')
rows = np.where(cta.sum(1) > 0)[0]
gap = rows[np.argmax(np.diff(rows))] + 1 if np.diff(rows).max() > 1 else (rows.min() + rows.max()) // 2
# divide o CTA na linha mais vazia entre as duas frases
prof = cta[rows.min():rows.max()].sum(1)
mid = rows.min() + 15 + int(np.argmin(prof[15:-15]))
cta_top, cta_bot = cta.copy(), cta.copy()
cta_top[mid:] = 0; cta_bot[:mid] = 0

ORANGE = np.array([220, 143, 7], np.float32) / 255
WHITE = np.array([252, 252, 246], np.float32) / 255
paths = []
for p in json.load(open(L + 'paths.json')):
    P = np.array(p['pts'], np.float32)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    tg = np.gradient(P, axis=0); tg /= np.linalg.norm(tg, axis=1, keepdims=True) + 1e-6
    nrm = np.stack([-tg[:, 1], tg[:, 0]], 1)
    paths.append(dict(P=P, s=s, n=nrm, orange=p['color'] == 'orange', L=s[-1]))
# ordem: laranja superior, branca superior, branca inferior, laranja inferior
paths.sort(key=lambda q: q['P'][len(q['P']) // 2, 1])

def ease(x):  # easeInOutCubic
    x = np.clip(x, 0, 1)
    return np.where(x < .5, 4 * x ** 3, 1 - (-2 * x + 2) ** 3 / 2)

def ease_out(x):
    x = np.clip(x, 0, 1)
    return 1 - (1 - x) ** 3

def lines_layer(t, head, tail):
    """head/tail: frações [0,1] por linha. Devolve alfas (laranja, branco)."""
    out = [np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)]
    for i, p in enumerate(paths):
        h, tl = float(head[i]), float(tail[i])
        if h <= tl + 1e-4:
            continue
        s, Lp = p['s'], p['L']
        # ondulação orgânica: duas ondas viajando ao longo da linha + respiração lenta
        ph = i * 1.7
        off = (7.0 * np.sin(2 * np.pi * s / 760 - 2 * np.pi * t / 4.2 + ph)
               + 3.2 * np.sin(2 * np.pi * s / 310 + 2 * np.pi * t / 2.9 + ph * 2.3)
               + 2.0 * np.sin(2 * np.pi * t / 6.0 + ph))
        Q = p['P'] + p['n'] * off[:, None]
        keep = (s >= tl * Lp) & (s <= h * Lp)
        Q = Q[keep]
        if len(Q) < 2:
            continue
        layer = out[0] if p['orange'] else out[1]
        m = np.zeros((H, W), np.uint8)
        cv2.polylines(m, [np.round(Q * 16).astype(np.int32)], False, 255, 3, cv2.LINE_AA, 4)
        layer[:] = np.maximum(layer, m.astype(np.float32) / 255)
    return out

def over(base, color, a):
    a = a[..., None]
    return base * (1 - a) + color * a

def shift_rows(a, dy):
    """desloca verticalmente um alfa em dy pixels (subpixel)."""
    M = np.float32([[1, 0, 0], [0, 1, dy]])
    return cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR)

def staggered_reveal(a, t, t0, dur, spread, rise=26, x0=180, x1=890):
    """cada coluna sobe e aparece com atraso proporcional à posição x (efeito 'letra a letra')."""
    xs = np.arange(W, dtype=np.float32)
    delay = np.clip((xs - x0) / (x1 - x0), 0, 1) * spread
    k = ease_out((t - t0 - delay) / dur)          # 0..1 por coluna
    res = np.zeros_like(a)
    # quantiza em poucos níveis de deslocamento para ficar rápido
    for lv in np.linspace(0, 1, 13):
        cols = np.abs(k - lv) <= (1 / 24 + 1e-6)
        if not cols.any() or lv == 0:
            continue
        sh = shift_rows(a, rise * (1 - lv))
        res[:, cols] = sh[:, cols] * lv
    return res

def wipe(a, t, t0, dur, x0=300, x1=885, soft=40):
    """revelação da esquerda para a direita, como uma caneta escrevendo."""
    edge = x0 - soft + (x1 - x0 + 2 * soft) * float(ease((t - t0) / dur))
    xs = np.arange(W, dtype=np.float32)
    m = np.clip((edge - xs) / soft, 0, 1)
    return a * m[None, :]

def zoom(img, sc, cx=W / 2, cy=H * 0.45):
    M = np.float32([[sc, 0, cx * (1 - sc)], [0, sc, cy * (1 - sc)]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def lin(t, a, b):
    return float(np.clip((t - a) / (b - a), 0, 1))

def frame(t):
    n = len(paths)
    # ---------------- Cena A: fundo da arte 1 ----------------
    A = bgA.copy()
    draw_on = [(0.5, 2.3), (0.8, 2.3), (1.0, 3.0), (1.3, 2.6)]
    retract = [(3.7, 4.7), (3.8, 4.8), (3.9, 5.0), (3.8, 4.6)]
    head = [ease((t - a) / (b - a)) for a, b in draw_on]
    tail = [ease((t - a) / (b - a)) for a, b in retract]
    if t < 5.1:
        lo, lw = lines_layer(t, head, tail)
        A = over(A, ORANGE, lo)
        A = over(A, WHITE, lw)
    title_fade = 1 - ease((t - 8.5) / 0.7)
    if t > 4.2 and title_fade > 0:
        o = staggered_reveal(t_orange, t, 4.3, 0.75, 0.55)
        pa = staggered_reveal(t_para, t, 5.25, 0.6, 0.15, rise=18)
        rs = wipe(t_resp, t, 5.55, 1.7)
        lift = -14 * float(ease((t - 8.5) / 0.7))
        if lift:
            o, pa, rs = shift_rows(o, lift), shift_rows(pa, lift), shift_rows(rs, lift)
        A = over(A, ORANGE, o * title_fade)
        A = over(A, WHITE, np.maximum(pa, rs) * title_fade)
    # zoom: aproximação lenta + "respiração" (inspira/expira) enquanto o título aparece
    scA = 1.06 - 0.06 * float(ease(t / 4.3))
    if t > 4.3:  # um ciclo completo de respiração termina em 1.0, alinhado com a cena seguinte
        scA = 1.0 + 0.035 * (0.5 - 0.5 * np.cos(2 * np.pi * min(t - 4.3, 4.6) / 4.6))
    A = zoom(A, scA)

    xf = float(ease((t - 8.8) / 0.9))
    if xf <= 0:
        frameimg = A
    else:
        # ---------------- Cena B: arte 5 com a modelo ----------------
        B = bgB.copy()
        draw_on_B = [(9.3, 11.0), (9.6, 11.0), (9.5, 11.6), (9.9, 11.0)]
        headB = [ease((t - a) / (b - a)) for a, b in draw_on_B]
        lo, lw = lines_layer(t, headB, [0] * n)
        B = over(B, ORANGE, lo * (1 - mat))   # laranja passa por trás do tapete, como na arte
        B = over(B, WHITE, lw)                # branca passa por cima
        B = over(B, WHITE, staggered_reveal(cta_top, t, 11.25, 0.7, 0.3, rise=22, x0=350, x1=710))
        B = over(B, WHITE, staggered_reveal(cta_bot, t, 11.72, 0.7, 0.3, rise=18, x0=380, x1=680))
        scB = 1.0 + 0.045 * float(ease((t - 8.9) / 6.1))
        B = zoom(B, scB)
        frameimg = A * (1 - xf) + B * xf if xf < 1 else B
    # entrada suave do preto no primeiro instante
    fin = lin(t, 0, 0.35)
    return np.clip(frameimg * fin, 0, 1)

if PREVIEW:
    for t in [1.6, 3.2, 4.4, 5.0, 6.4, 7.6, 9.2, 10.4, 12.0, 14.5]:
        Image.fromarray((frame(t) * 255).astype(np.uint8)).save(f'{OUT}_{t:05.2f}.png')
    sys.exit()

ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                       '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                       '-f', 'lavfi', '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',
                       '-shortest', '-c:v', 'libx264', '-preset', 'slow', '-crf', '15', '-profile:v', 'high',
                       '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', OUT],
                      stdin=subprocess.PIPE)
N = int(DUR * FPS)
for i in range(N):
    ff.stdin.write((frame(i / FPS) * 255 + 0.5).astype(np.uint8).tobytes())
    if i % 60 == 0:
        print(f'{i}/{N}', flush=True)
ff.stdin.close(); ff.wait()
print('done', OUT)
