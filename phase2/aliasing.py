"""Track 7: Fourier aliasing of non-cyclic categorical features under a chosen cyclic labelling of 12 concepts.
For a feature vector f over Z_12 (centered), the paired-mode energy shares are computed; we tabulate, for
(a) the accidental indicator under semitone and fifths labellings, (b) all 2^12 binary partitions, (c) contiguous blocks
of every width under the fifths labelling, (d) random categorical partitions with k classes: the distribution of the
maximum paired-mode share and how often a single pair carries > 50% / > 70% of the energy."""
import numpy as np, itertools, json, os
N = 12
def paired_shares(f):
    f = np.asarray(f, float); f = f - f.mean(); F = np.fft.fft(f) / np.sqrt(N); E = np.abs(F) ** 2
    v = np.array([E[m] + E[N - m] for m in range(1, N // 2)] + [E[N // 2]]); return v / v.sum() if v.sum() > 0 else v
fifths = (7 * np.arange(N)) % 12   # semitone index -> fifths position
black = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0])
print("accidental indicator in semitone labelling: paired shares P1..P5,E6 =", np.round(paired_shares(black), 3))
# same feature under the fifths labelling (relabel x -> 7x): shares permute P1<->P5
order = np.argsort(fifths); print("in fifths labelling:", np.round(paired_shares(black[order]), 3))
# (b) all binary partitions
res = []
for bits in itertools.product([0, 1], repeat=N):
    f = np.array(bits)
    if f.sum() in (0, N): continue
    s = paired_shares(f); res.append((s.max(), int(np.argmax(s)), f.sum()))
res = np.array(res)
print(f"all {len(res)} nontrivial binary partitions: max paired share mean {res[:,0].mean():.2f}, median {np.median(res[:,0]):.2f}; P(max share > 0.5) = {np.mean(res[:,0] > 0.5 + 1e-9):.3f} (exactly 0.5: {np.mean(np.abs(res[:,0] - 0.5) < 1e-9):.3f}; >= 0.5: {np.mean(res[:,0] > 0.5 - 1e-9):.3f}), > 0.7: {np.mean(res[:,0] > 0.7 + 1e-9):.3f}  [tie-aware; an earlier version printed 0.294 for '> 0.5' because 276 partitions sit exactly at 0.5 and float rounding split them]")
# (c) contiguous blocks of width w on the circle
print("contiguous block (width w) on a 12-cycle: share in the fundamental pair:")
for w in range(1, 7):
    f = np.zeros(N); f[:w] = 1; s = paired_shares(f); print(f"   w={w}: fundamental share {s[0]:.3f}, max share {s.max():.3f} (pair {np.argmax(s)+1})")
# (d) random k-class categorical partitions, dummy-coded energy summed over classes
rng = np.random.default_rng(0)
for k in (2, 3, 4):
    mx = []
    for _ in range(5000):
        lab = rng.integers(0, k, N)
        if len(set(lab)) < 2: continue
        tot = np.zeros(N // 2)
        for c in set(lab):
            f = (lab == c).astype(float); f -= f.mean(); F = np.fft.fft(f) / np.sqrt(N); E = np.abs(F) ** 2; tot += np.array([E[m] + E[N - m] for m in range(1, N // 2)] + [E[N // 2]])
        mx.append((tot / tot.sum()).max())
    mx = np.array(mx); print(f"random {k}-class partitions: max paired share median {np.median(mx):.2f}; P(>0.5) = {np.mean(mx > 0.5):.3f}, P(>0.7) = {np.mean(mx > 0.7):.3f}")
# (e) alternating and letter groupings
alt = np.array([i % 2 for i in range(N)]); print("alternating labels (semitone): shares", np.round(paired_shares(alt), 3))
letters = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6])  # canonical spelling root letters C Db D Eb E F F# G Ab A Bb B -> C,D,D,E,E,F,F,G,A,A,B,B
tot = np.zeros(N // 2)
for c in set(letters):
    f = (letters == c).astype(float); f -= f.mean(); F = np.fft.fft(f) / np.sqrt(N); E = np.abs(F) ** 2; tot += np.array([E[m] + E[N - m] for m in range(1, N // 2)] + [E[N // 2]])
print("root-letter identity (7 classes, semitone labelling): shares", np.round(tot / tot.sum(), 3))
os.makedirs("results/phase2/aliasing", exist_ok=True)
