"""Extrai camadas a partir das artes: traçados vetoriais das linhas, título, CTA e fundos limpos."""
import json, sys
import numpy as np, cv2
from PIL import Image
from scipy.ndimage import map_coordinates, gaussian_filter1d
from skimage.measure import label, regionprops

I, O = sys.argv[1], sys.argv[2]
ld = lambda f: np.asarray(Image.open(I + f).convert('RGB')).astype(np.float32)
a, b, e = ld('1.webp'), ld('2.webp'), ld('5.webp')
H, W = a.shape[:2]

# ---------- 1. Linhas: máscara pela diferença entre arte 1 e arte 2 ----------
d = np.abs(a - b).max(2)
mask = d > 60
lab = label(mask, connectivity=2)
field = cv2.GaussianBlur(np.clip(d, 0, 255) / 255.0, (0, 0), 1.6)

def sample(p):
    return map_coordinates(field, [[p[1]], [p[0]]], order=1, mode='constant')[0]

def centroid(p, n):
    """Recentraliza o ponto no eixo da linha, ao longo da normal."""
    offs = np.linspace(-4, 4, 33)
    pts = p[None] + offs[:, None] * n[None]
    w = map_coordinates(field, [pts[:, 1], pts[:, 0]], order=1) ** 2
    if w.sum() < 1e-6:
        return p
    return (pts * w[:, None]).sum(0) / w.sum()

def trace(start, direction, step=2.0):
    p, t = np.array(start, float), np.array(direction, float)
    t /= np.linalg.norm(t)
    out = [p.copy()]
    for _ in range(6000):
        best, bestv = None, -1
        for ang in np.radians(np.arange(-35, 36, 2.5)):
            c, s = np.cos(ang), np.sin(ang)
            nt = np.array([c * t[0] - s * t[1], s * t[0] + c * t[1]])
            v = sample(p + nt * 5) - abs(ang) * 0.05  # prefere seguir reto (atravessa cruzamentos)
            if v > bestv:
                best, bestv = nt, v
        if bestv < 0.12:
            break
        q = p + best * step
        q = centroid(q, np.array([-best[1], best[0]]))
        nt = q - p
        if np.linalg.norm(nt) < 1e-3:
            break
        t = 0.7 * t + 0.3 * nt / np.linalg.norm(nt)
        t /= np.linalg.norm(t)
        p = q
        out.append(p.copy())
        if not (-1 <= p[0] <= W and -1 <= p[1] <= H) or (len(out) > 20 and (p[0] < 1.5 or p[0] > W - 2.5 or p[1] < 1.5 or p[1] > H - 2.5)):
            break
    return np.array(out)

paths = []
for r in regionprops(lab):
    if r.area < 400:
        continue
    ys, xs = np.where(lab == r.label)
    # ponto inicial: pixel da componente mais próximo de uma borda
    dist = np.minimum.reduce([xs, W - 1 - xs, ys, H - 1 - ys])
    i = np.argmin(dist)
    sx, sy = xs[i], ys[i]
    inward = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}[int(np.argmin([sx, W - 1 - sx, sy, H - 1 - sy]))]
    # direção inicial: média dos pixels próximos
    near = (np.abs(xs - sx) < 25) & (np.abs(ys - sy) < 25)
    v = np.array([xs[near].mean() - sx, ys[near].mean() - sy])
    if np.linalg.norm(v) < 1:
        v = np.array(inward, float)
    pts = trace((sx, sy), v)
    col = b[ys, xs]
    is_orange = np.median(col[:, 2]) < 150
    # suavização e reamostragem uniforme
    pts[:, 0] = gaussian_filter1d(pts[:, 0], 3, mode='nearest')
    pts[:, 1] = gaussian_filter1d(pts[:, 1], 3, mode='nearest')
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = int(s[-1] / 2)
    ss = np.linspace(0, s[-1], n)
    pts = np.stack([np.interp(ss, s, pts[:, 0]), np.interp(ss, s, pts[:, 1])], 1)
    # prolonga as pontas para fora do quadro (o movimento não revela o fim da linha)
    for end in (0, -1):
        k = 1 if end == 0 else -2
        dv = pts[end] - pts[k]
        dv /= np.linalg.norm(dv)
        ext = pts[end] + dv * np.arange(2, 60, 2)[:, None]
        pts = np.vstack([ext[::-1], pts]) if end == 0 else np.vstack([pts, ext])
    paths.append({'color': 'orange' if is_orange else 'white', 'pts': pts.round(2).tolist(),
                  'area': int(r.area), 'length': float(s[-1])})
    print(r.label, 'orange' if is_orange else 'white', 'area', r.area, 'traced length', round(s[-1]), 'width≈', round(r.area / s[-1], 2), 'end', pts[-1].round())

keep = []
for p in sorted(paths, key=lambda q: -q['area']):
    P = np.array(p['pts'])
    dup = any(np.mean([np.min(np.linalg.norm(np.array(k['pts']) - x, axis=1)) for x in P[::25]]) < 6 for k in keep)
    if not dup:
        keep.append(p)
paths = keep
print('linhas finais:', len(paths))
json.dump(paths, open(O + 'paths.json', 'w'))

# ---------- 2. Título (arte 3 já tem alfa): separa laranja e branco ----------
t = np.asarray(Image.open(I + '3.png')).astype(np.float32)
rgb, al = t[..., :3], t[..., 3] / 255
orange_px = (rgb[..., 2] < 140) & (al > 0.02)
lab_t = label(al > 0.02, connectivity=2)
# rótulos por linha do título: laranja = "Uma pausa"; branco "para"; branco cursivo "respirar"
o_layer = np.zeros((H, W, 4), np.uint8)
w_layer = np.zeros((H, W, 4), np.uint8)
for r in regionprops(lab_t):
    ys, xs = r.coords[:, 0], r.coords[:, 1]
    orange = orange_px[ys, xs].mean() > 0.5
    tgt = o_layer if orange else w_layer
    tgt[ys, xs, :3] = rgb[ys, xs]
    tgt[ys, xs, 3] = t[ys, xs, 3]
# separa "para" (x<320) de "respirar"
para = np.zeros_like(w_layer); resp = np.zeros_like(w_layer)
for r in regionprops(lab_t):
    ys, xs = r.coords[:, 0], r.coords[:, 1]
    if orange_px[ys, xs].mean() > 0.5:
        continue
    tgt = para if (r.bbox[3] < 325 and r.bbox[0] < 400) else resp
    tgt[ys, xs] = w_layer[ys, xs]
# o "r" inicial de respirar começa antes de x=330? confere
Image.fromarray(o_layer).save(O + 'title_orange.png')
Image.fromarray(para).save(O + 'title_para.png')
Image.fromarray(resp).save(O + 'title_respirar.png')

# ---------- 3. Arte 5: remove linhas e CTA para obter fundo limpo ----------
line_mask = np.zeros((H, W), np.uint8)
for r in regionprops(lab):
    if r.area >= 400:
        line_mask[r.coords[:, 0], r.coords[:, 1]] = 1
line_mask = cv2.dilate(((d > 18) & (cv2.dilate(line_mask, np.ones((11, 11), np.uint8)) > 0)).astype(np.uint8), np.ones((5, 5), np.uint8))
cta_box = (slice(1725, 1835), slice(340, 720))
cta_mask = np.zeros((H, W), np.uint8)
sub = (e[cta_box].min(2) > 150).astype(np.uint8)
cta_mask[cta_box] = cv2.dilate(sub, np.ones((17, 17), np.uint8))
hole = np.maximum(line_mask, cta_mask) > 0
clean = e.copy()
# preenchimento por textura: copia grama vizinha (sem borrão)
filled = np.zeros((H, W), bool)
offs = [(0, 12), (12, 0), (0, -12), (-12, 0), (12, 12), (-12, -12), (12, -12), (-12, 12), (0, 24), (24, 0), (0, -24), (-24, 0)]
ctaoffs = [(-118, 0), (118, 0)]
yy, xx = np.mgrid[0:H, 0:W]
for dy, dx in ctaoffs + offs:
    sy, sx = np.clip(yy + dy, 0, H - 1), np.clip(xx + dx, 0, W - 1)
    ok = hole & ~filled & ~hole[sy, sx]
    if (dy, dx) in ctaoffs:
        ok &= cta_mask > 0
    clean[ok] = e[sy[ok], sx[ok]]
    filled |= ok
# dentro do tapete (superfície lisa) usa inpaint
roi = np.zeros((H, W), bool); roi[495:1245, 335:685] = True
inner = (hole & roi).astype(np.uint8)
cl8 = cv2.inpaint(cv2.cvtColor(clean.astype(np.uint8), cv2.COLOR_RGB2BGR), inner, 5, cv2.INPAINT_TELEA)
clean = cv2.cvtColor(cl8, cv2.COLOR_BGR2RGB).astype(np.float32)
# borda suave entre preenchimento e original
soft = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), 1.2)[..., None]
soft = np.clip(soft * 1.6, 0, 1)
clean = clean * soft + e * (1 - soft)
# logotipo e área do CTA: recompõe com a mesma região da arte 1 (sem linhas nem texto), com borda suave
def patch(y0, y1, x0, x1, f=14):
    m = np.zeros((H, W), np.float32); m[y0 + f:y1 - f, x0 + f:x1 - f] = 1
    m = cv2.GaussianBlur(m, (0, 0), f / 2)[..., None]
    return m
m = patch(70, 200, 360, 760)
clean = a * m + clean * (1 - m)
m = patch(1700, 1860, 320, 750, 20)
clean = a * m + clean * (1 - m)
clean = np.clip(clean, 0, 255).astype(np.uint8)
Image.fromarray(clean).save(O + 'bg5_clean.png')
# camada do CTA (branco) com alfa estimado sobre o fundo limpo
bg = clean.astype(np.float32)
alpha = np.clip(((e - bg) / np.maximum(255 - bg, 1)).mean(2), 0, 1)
alpha[cta_mask == 0] = 0
cta = np.zeros((H, W, 4), np.uint8)
cta[..., :3] = 255
cta[..., 3] = (alpha * 255).astype(np.uint8)
Image.fromarray(cta).save(O + 'cta.png')

# máscara do tapete + modelo (oclui a linha laranja na cena final, como na arte)
roi = np.zeros((H, W), bool); roi[495:1245, 335:685] = True
g = clean.astype(np.float32)
non_green = ~((g[..., 1] > g[..., 0] * 1.08) & (g[..., 1] > g[..., 2] * 1.25))
mat = (non_green & roi).astype(np.uint8)
mat = cv2.morphologyEx(mat, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
mat = cv2.morphologyEx(mat, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
lab_m = label(mat)
big = max(regionprops(lab_m), key=lambda r: r.area).label
mat = (lab_m == big).astype(np.uint8) * 255
mat = cv2.GaussianBlur(mat, (0, 0), 1.0)
Image.fromarray(mat).save(O + 'mat_mask.png')
print('ok')
