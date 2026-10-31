import numpy as np, wave, os
SR, DUR = 48000, 10.0
n = int(SR * DUR); t = np.arange(n) / SR
rng = np.random.default_rng(3)
mix = np.zeros((n, 2))

def env(a, r, start=0.0, end=DUR):
    e = np.clip((t - start) / a, 0, 1) * np.clip((end - t) / r, 0, 1)
    return e * e * (3 - 2 * e)

def note(f, amp, start, end, a=1.5, r=1.8, pan=0.0, det=0.0015):
    s = 0
    for d in (-det, 0, det):
        ph = rng.uniform(0, 6.28)
        s = s + np.sin(2 * np.pi * f * (1 + d) * t + ph) + 0.18 * np.sin(4 * np.pi * f * (1 + d) * t + ph)
    s = s / 3 * amp * env(a, r, start, end) * (1 + 0.08 * np.sin(2 * np.pi * 0.25 * t + f))
    mix[:, 0] += s * (1 - pan) / 2 * 2 ** 0.5
    mix[:, 1] += s * (1 + pan) / 2 * 2 ** 0.5

# pad quente: Fmaj9 -> Dm9 (abre na chegada ao balcao)
for f, p in [(87.31, 0), (174.61, -.3), (220.0, .3), (261.63, -.2), (329.63, .2), (392.0, .4)]:
    note(f, 0.07, 0, 5.4, a=1.6, r=1.4, pan=p)
for f, p in [(73.42, 0), (146.83, -.3), (174.61, .3), (220.0, -.2), (261.63, .2), (329.63, .4)]:
    note(f, 0.07, 4.6, 10.0, a=1.2, r=1.1, pan=p)

# whoosh do zoom (ruido filtrado com corte que sobe)
noise = rng.normal(0, 1, n)
cut = 300 + 3200 * np.clip((t - 2.4) / 4.9, 0, 1) ** 2
y = np.zeros(n); acc = 0.0
alpha = 1 - np.exp(-2 * np.pi * cut / SR)
for i in range(n):
    acc += alpha[i] * (noise[i] - acc); y[i] = acc
w_env = np.exp(-((t - 6.6) / 1.7) ** 2) * (t < 7.4) + np.exp(-((t - 6.6) / 1.7) ** 2) * np.exp(-(t - 7.4) * 6) * (t >= 7.4)
mix[:, 0] += y * w_env * 0.10
mix[:, 1] += np.roll(y, 240) * w_env * 0.10

# sinos nos brilhos das joias e no cupom
def bell(f, start, amp, pan):
    tt = np.clip(t - start, 0, None)
    s = np.zeros(n)
    for k, (m, d) in enumerate([(1, 1.6), (2.76, 0.9), (5.4, 0.5), (8.9, 0.3)]):
        s += np.sin(2 * np.pi * f * m * tt) * np.exp(-tt / d) / (k + 1)
    s *= (t >= start) * amp * np.clip(tt / 0.004, 0, 1)
    mix[:, 0] += s * (1 - pan); mix[:, 1] += s * (1 + pan)
bell(1046.5, 7.35, 0.05, 0.3); bell(1318.5, 7.55, 0.04, -0.2)
bell(1568.0, 7.75, 0.035, 0.1); bell(1760.0, 8.0, 0.03, -0.3)
bell(1046.5, 8.65, 0.05, 0.0); bell(1568.0, 9.45, 0.035, 0.2)

# reverb simples (delays com feedback)
out = mix.copy()
for d, g in [(0.031, .35), (0.047, .3), (0.073, .25), (0.113, .2), (0.167, .15)]:
    k = int(d * SR)
    for ch, off in ((0, 0), (1, 7)):
        tmp = np.zeros(n); tmp[k + off:] = mix[:-(k + off), ch] * g
        out[:, ch] += tmp
out *= np.clip(t / 0.4, 0, 1)[:, None] * np.clip((DUR - t) / 0.6, 0, 1)[:, None]
out = np.tanh(out * 1.4) / np.max(np.abs(np.tanh(out * 1.4))) * 0.85
pcm = (out * 32767).astype(np.int16)
with wave.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trilha.wav'), 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
