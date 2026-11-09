"""Trilha original em estilo bossa nova (violão dedilhado + percussão + respiração + vento),
sincronizada com o story "Uma pausa para respirar" (15 s).

Andamento: semínima = 128 em 2/4  ->  1 semicolcheia = 0,1172 s; 1 compasso harmônico
(dois compassos de 2/4) = 1,875 s; 8 compassos harmônicos = 15 s exatos.
"""
import sys
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve
from scipy.io import wavfile

SR = 44100
DUR = 15.0
N = int(SR * DUR)
S16 = 60 / 128 / 4          # semicolcheia
BAR = 16 * S16              # compasso harmônico (dois 2/4) = 1,875 s
rng = np.random.default_rng(7)

def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], 'band', fs=SR, output='sos'), x)

def lp(x, f, order=2):
    return sosfilt(butter(order, f, 'low', fs=SR, output='sos'), x)

def hp(x, f, order=2):
    return sosfilt(butter(order, f, 'high', fs=SR, output='sos'), x)

def mtof(m):
    return 440 * 2 ** ((m - 69) / 12)

def add(buf, sig, t):
    i = max(0, int(round(t * SR)))
    if i >= len(buf):
        return
    j = min(len(buf), i + len(sig))
    buf[i:j] += sig[:j - i]

# ---------------- violão de náilon (Karplus-Strong vetorizado por blocos) ----------------
def pluck(freq, dur, bright=0.5, decay=0.996, pos=0.18):
    target = SR / freq - 0.5                 # o filtro de média adiciona meio sample de atraso
    P = max(2, int(np.floor(target)))
    ratio = (P + 0.5) / (SR / freq)          # correção fina de afinação por reamostragem
    n = int(dur * SR * ratio) + 2
    exc = rng.uniform(-1, 1, P)
    exc = lp(exc, 1200 + 5000 * bright, 1)          # dedo (náilon): ataque mais macio
    k = max(1, int(P * pos))                          # posição do dedilhado (filtro pente)
    exc = exc - np.roll(exc, k) * 0.9
    y = np.zeros(n + P + 1)
    y[:P] = exc
    b = 1
    while b * P < n:
        prev = y[(b - 1) * P:b * P]
        prev1 = np.concatenate([[y[b * P - P - 1] if b > 1 else 0.0], prev[:-1]])
        y[b * P:(b + 1) * P] = decay * 0.5 * (prev + prev1)
        b += 1
    y = y[:n]
    m = int(dur * SR)
    y = np.interp(np.arange(m) * ratio, np.arange(n), y)   # leitura mais lenta -> afinação exata
    return y / (np.abs(y).max() + 1e-9)

def note_env(n, cut_at=None, rel=0.035):
    env = np.ones(n)
    if cut_at is not None and cut_at < n:
        r = int(rel * SR)
        env[cut_at:cut_at + r] = np.linspace(1, 0, min(r, n - cut_at))
        env[cut_at + r:] = 0
    return env

guitar = np.zeros(N + SR * 3)
bass = np.zeros(N + SR * 3)

# acordes (harmonia original): baixo fundamental, baixo alternado (quinta), vozes do acorde
CH = {
    'Cmaj9':  (48, 43, [52, 59, 62, 67]),
    'Am9':    (45, 40, [55, 60, 64, 71]),
    'Dm9':    (50, 45, [53, 60, 64, 69]),
    'G13':    (43, 50, [53, 59, 64, 69]),
    'Em7':    (40, 47, [55, 59, 62, 67]),
    'A7b13':  (45, 40, [55, 61, 65, 69]),
    'Fmaj9':  (41, 48, [52, 57, 60, 67]),
    'Fm6':    (41, 48, [50, 56, 60, 65]),
    'Cfinal': (48, 43, [52, 59, 62, 67, 71, 76]),
}
# 8 compassos harmônicos de 1,875 s (16 semicolcheias); cada um dividido em duas metades de 8
HARM = [('Cmaj9', 'Cmaj9'), ('Am9', 'Am9'), ('Dm9', 'Dm9'), ('G13', 'G13'),
        ('Em7', 'A7b13'), ('Fmaj9', 'Fmaj9'), ('Fm6', 'Fm6'), ('Cfinal', 'Cfinal')]

def chord_at(slot):
    """slot global de semicolcheia -> acorde (com antecipação típica da bossa na semicolcheia 13)."""
    hb, half = slot // 16, (slot % 16) // 8
    if slot % 16 == 13:             # antecipação do acorde seguinte
        hb, half = hb + 1, 0
    hb = min(hb, 7)
    return HARM[hb][half]

# levada de violão: acordes no padrão da clave da bossa (0,3,6,10,13 a cada 16), baixo no 1 e no 2
CLAVE = [0, 3, 6, 10, 13]
chord_slots = [h * 16 + c for h in range(8) for c in CLAVE]
chord_slots = [s for s in chord_slots if s * S16 < DUR - 0.9]
for i, s in enumerate(chord_slots):
    t = s * S16
    name = chord_at(s)
    _, _, voices = CH[name]
    nxt = chord_slots[i + 1] * S16 if i + 1 < len(chord_slots) else t + 2.0
    length = nxt - t
    # dinâmica: mais suave durante a respiração (compassos 2-3), cheia a partir da modelo (compasso 5)
    bar = int(t // BAR)
    vel = {0: .55, 1: .65, 2: .42, 3: .42, 4: .58, 5: .85, 6: .85, 7: .8}[bar]
    accent = 1.0 if s % 16 in (0, 6, 10) else 0.85
    for k, m in enumerate(voices):
        d = 0.012 * k                                   # dedilhado (pinça dos dedos)
        nl = length + 0.06
        sig = pluck(mtof(m), nl + 0.1, bright=0.35 + 0.2 * vel, decay=0.997)
        sig *= note_env(len(sig), int((nl - d) * SR))
        sig *= np.linspace(1, .88, len(sig))
        add(guitar, sig * vel * accent * (0.9 - 0.06 * k), t + d + rng.normal(0, 0.003))

# baixo do polegar: fundamental no tempo 1 e quinta no tempo 2 de cada 2/4
for h in range(8):
    for beat, idx in ((0, 0), (4, 1), (8, 0), (12, 1)):
        s = h * 16 + beat
        t = s * S16
        if t > DUR - 0.9:
            continue
        name = HARM[h][beat // 8]
        m = CH[name][idx]
        bar = int(t // BAR)
        vel = {0: .6, 1: .7, 2: .5, 3: .5, 4: .65, 5: .95, 6: .95, 7: .85}[bar]
        sig = pluck(mtof(m), 0.6, bright=0.2, decay=0.998, pos=0.3)
        sig *= note_env(len(sig), int(S16 * 4 * SR - 0.03 * SR))
        add(bass, sig * vel, t)

# acorde final que soa até o fim
_, root, voices = CH['Cfinal']
tf = DUR - 0.9
for k, m in enumerate([36, 43] + voices):
    sig = pluck(mtof(m), 1.4, bright=0.45, decay=0.9985)
    add(guitar, sig * 0.75 * (0.95 - 0.05 * k), tf + 0.018 * k)

# corpo do violão: realce de graves suaves e leve brilho
guitar = guitar + 0.35 * bp(guitar, 90, 260) + 0.15 * bp(guitar, 2500, 5000)
bass = lp(bass, 900)

# ---------------- percussão ----------------
perc = np.zeros(N + SR)

def noise_hit(dur, lo, hi, tau):
    n = int(dur * SR)
    x = bp(rng.uniform(-1, 1, n), lo, hi)
    return x * np.exp(-np.arange(n) / (tau * SR))

def kick(dur=0.35, f0=95, f1=48):
    n = int(dur * SR); t = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-t / 0.04)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.12)

def rim():
    n = int(0.08 * SR); t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * 1750 * t) * np.exp(-t / 0.012)
    return 0.6 * tone + noise_hit(0.08, 1500, 6000, 0.008)

for s in range(int(DUR / S16)):
    t = s * S16
    if t > DUR - 0.5:
        break
    bar = int(t // BAR)
    sw = 0.012 if s % 2 else 0.0                      # balanço (swing leve nas semicolcheias pares)
    # ganzá: semicolcheias com acentos nas "e" de cada tempo -> mais vivo
    acc = {0: .45, 1: .6, 2: 1.0, 3: .7}[s % 4]
    g = {0: .35, 1: .5, 2: .35, 3: .45, 4: .65, 5: 1, 6: 1, 7: .9}[bar]
    if bar == 4:                                      # crescendo até a entrada da modelo
        g = .4 + .6 * ((t - 4 * BAR) / BAR)
    add(perc, noise_hit(0.06, 5000, 11000, 0.018) * acc * g * 0.8, t + sw)
    # surdo/bumbo: tempo 1 curto, tempo 2 longo (marcação clássica da bossa)
    if bar in (1, 5, 6, 7) or (bar == 4 and t > 4 * BAR + BAR / 2):
        if s % 8 == 0:
            add(perc, kick(0.25) * 0.45, t)
        if s % 8 == 4:
            add(perc, kick(0.45, 85, 44) * 0.7, t)
    # aro (rimshot) na clave
    if (bar in (1, 5, 6, 7) or (bar == 4 and t > 4 * BAR + BAR / 2)) and s % 16 in CLAVE:
        add(perc, rim() * 0.28, t + sw)
# pequena virada antes da modelo entrar (compasso 4, últimas semicolcheias)
for k, s in enumerate(range(5 * 16 - 4, 5 * 16)):
    add(perc, rim() * (0.18 + 0.05 * k), s * S16)

# ---------------- respiração (sincronizada com o zoom de 4,3 s a 8,9 s) ----------------
breath = np.zeros(N)
def breath_seg(t0, t1, kind):
    n = int((t1 - t0) * SR)
    x = rng.normal(0, 1, n)
    u = np.linspace(0, 1, n)
    if kind == 'in':     # inspiração pelo nariz: mais aguda, cresce e para
        x = bp(x, 900, 5200) + 0.5 * bp(x, 2400, 3200, 1)
        env = np.sin(np.pi * np.clip(u / 0.92, 0, 1) * 0.5) ** 1.3 * np.clip((1 - u) / 0.12, 0, 1)
    else:                # expiração pela boca: mais grave e aberta ("haaa"), cai devagar
        x = bp(x, 250, 2200) + 0.7 * bp(x, 600, 950, 1) + 0.3 * bp(x, 1100, 1500, 1)
        env = np.clip(u / 0.08, 0, 1) * (1 - u) ** 1.4
    x = x / (np.abs(x).max() + 1e-9)
    add(breath, x * env, t0)
breath_seg(4.35, 6.55, 'in')
breath_seg(6.65, 8.85, 'out')

# ---------------- vento nas palmeiras ----------------
t = np.arange(N) / SR
brown = np.cumsum(rng.normal(0, 1, N)); brown -= lp(brown, 0.5, 1)
wind = lp(brown, 700) / 40
gust = bp(rng.normal(0, 1, N), 400, 1800)
lfo = 0.55 + 0.45 * np.sin(2 * np.pi * 0.18 * t) * np.sin(2 * np.pi * 0.07 * t + 1)
def swell(a, b):
    return np.clip(np.sin(np.pi * np.clip((t - a) / (b - a), 0, 1)), 0, 1)
level = 0.35 + 0.65 * (swell(0.3, 3.2) + swell(9.2, 11.8)) + 0.4 * swell(3.6, 5.2)  # linhas se movendo
wind = (wind / (np.abs(wind).max() + 1e-9) * 0.8 + gust / (np.abs(gust).max() + 1e-9) * 0.25) * lfo * level
# "whoosh" da transição para a modelo (8,8 s)
n = int(1.3 * SR); u = np.linspace(0, 1, n)
wh = rng.normal(0, 1, n)
wh = sum(bp(wh[i::1], 300 + 2500 * c, 600 + 3500 * c) * ((u > c - .15) & (u < c + .15)) for c in np.linspace(.1, .9, 6) for i in [0])
wh = wh / (np.abs(wh).max() + 1e-9) * np.sin(np.pi * u) ** 2
whoosh = np.zeros(N); add(whoosh, wh, 8.35)

# ---------------- reverb (sala pequena) e mixagem ----------------
irn = int(1.1 * SR)
ir = rng.normal(0, 1, irn) * np.exp(-np.arange(irn) / (0.28 * SR))
ir = lp(ir, 5000); ir /= np.sqrt((ir ** 2).sum())
music = guitar[:N] * 0.55 + bass[:N] * 0.55 + perc[:N] * 0.6
music = music + 0.18 * fftconvolve(music, ir)[:N]
# abaixa a música durante a respiração (ducking), para a pausa ser ouvida
duck = 1 - 0.45 * np.clip(np.minimum((t - 4.1) / 0.4, (8.95 - t) / 0.4), 0, 1)
mix = music * duck + breath * 0.32 + wind * 0.17 + whoosh * 0.14
mix = hp(mix, 30)
# fade de entrada e saída
mix *= np.clip(t / 0.25, 0, 1) * np.clip((DUR - t) / 0.6, 0, 1)
mix = np.tanh(mix / (np.abs(mix).max() + 1e-9) * 1.4) / np.tanh(1.4) * 0.89
stereo = np.stack([mix + 0.08 * np.roll(guitar[:N], 220) * duck * 0.3,
                   mix + 0.08 * np.roll(perc[:N], 180) * 0.3], 1)
stereo = stereo / np.abs(stereo).max() * 0.89
wavfile.write(sys.argv[1], SR, (stereo * 32767).astype(np.int16))
print('ok', sys.argv[1])
