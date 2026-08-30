"""15 standard major-key spellings on the signed line of fifths (key-signature count s: negative = flats), their neutral
pitch classes, and candidate geometries (distance matrices) from music theory and from orthography.
Verified facts: pc(s) = 7 s mod 12; enharmonic pairs (Cb,B)=(-7,+5), (Gb,F#)=(-6,+6), (Db,C#)=(-5,+7).
"""
import numpy as np

KEYS15 = ["Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#"]
S = np.arange(-7, 8)                                  # signed line-of-fifths coordinate
PC = (7 * S) % 12                                     # neutral pitch class of the tonic (C=0)
WORDS = {"Cb": "C-flat", "Gb": "G-flat", "Db": "D-flat", "Ab": "A-flat", "Eb": "E-flat", "Bb": "B-flat", "F": "F", "C": "C",
         "G": "G", "D": "D", "A": "A", "E": "E", "B": "B", "F#": "F-sharp", "C#": "C-sharp"}
LETTER = np.array([ord(k[0]) - 65 for k in KEYS15])
GLYPH = np.array([-1 if k.endswith("b") else (1 if k.endswith("#") else 0) for k in KEYS15])   # flat / natural / sharp
ENH_PAIRS = [(0, 12), (1, 13), (2, 14)]               # (Cb,B), (Gb,F#), (Db,C#)
assert all(PC[i] == PC[j] for i, j in ENH_PAIRS)
n = 15


def _pair(f):
    return np.array([[f(i, j) for j in range(n)] for i in range(n)], float)


def circ12(a, b):
    d = abs(int(a) - int(b)) % 12; return min(d, 12 - d)


def levenshtein(a, b):
    m, k = len(a), len(b); D = np.zeros((m + 1, k + 1), int); D[:, 0] = range(m + 1); D[0, :] = range(k + 1)
    for i in range(1, m + 1):
        for j in range(1, k + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    return D[m, k]


# --- Krumhansl & Kessler (1982) major key profile; inter-key similarity = correlation between transposed profiles
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def kk_corr(pc_a, pc_b, prof_a=KK_MAJOR, prof_b=KK_MAJOR):
    pa = np.roll(prof_a, int(pc_a)); pb = np.roll(prof_b, int(pc_b)); return float(np.corrcoef(pa, pb)[0, 1])


# --- Chew's Spiral Array (2000): pitch k on the line of fifths at (r sin(k pi/2), r cos(k pi/2), k h);
#     major chord center = w1 P(k) + w2 P(k+1) + w3 P(k+4); major key center = o1 C(k) + o2 C(k+1) + o3 C(k-1).
SA_R, SA_H = 1.0, np.sqrt(2.0 / 15.0); SA_W = (0.6025, 0.2930, 0.1145); SA_O = (0.6025, 0.2930, 0.1145)


def sa_pitch(k): return np.array([SA_R * np.sin(k * np.pi / 2), SA_R * np.cos(k * np.pi / 2), k * SA_H])
def sa_major_chord(k): return SA_W[0] * sa_pitch(k) + SA_W[1] * sa_pitch(k + 1) + SA_W[2] * sa_pitch(k + 4)
def sa_major_key(k): return SA_O[0] * sa_major_chord(k) + SA_O[1] * sa_major_chord(k + 1) + SA_O[2] * sa_major_chord(k - 1)


def candidate_geometries(tokcounts=None, logfreq=None):
    """Return dict name -> 15x15 DISTANCE matrix (larger = less similar)."""
    G = {}
    G["circle_fifths"] = _pair(lambda i, j: circ12(S[i], S[j]))                       # periodic neutral-pitch-class fifths distance
    G["line_fifths"] = _pair(lambda i, j: abs(S[i] - S[j]))                            # open tonal-pitch-class coordinate
    G["chromatic"] = _pair(lambda i, j: circ12(PC[i], PC[j]) if False else min((PC[i] - PC[j]) % 12, (PC[j] - PC[i]) % 12))
    G["kk_major"] = _pair(lambda i, j: 1.0 - kk_corr(PC[i], PC[j]))                    # Krumhansl–Kessler major-major (periodic, shaped)
    G["spiral_array"] = _pair(lambda i, j: np.linalg.norm(sa_major_key(S[i]) - sa_major_key(S[j])))  # Chew: open helix
    G["glyph_class"] = _pair(lambda i, j: float(GLYPH[i] != GLYPH[j]))                 # flat / natural / sharp class
    G["has_accidental"] = _pair(lambda i, j: float((GLYPH[i] != 0) != (GLYPH[j] != 0)))
    G["edit_distance"] = _pair(lambda i, j: levenshtein(KEYS15[i], KEYS15[j]))
    G["same_letter"] = _pair(lambda i, j: float(LETTER[i] != LETTER[j]))
    G["alphabet"] = _pair(lambda i, j: abs(LETTER[i] - LETTER[j]))
    G["n_accidentals"] = _pair(lambda i, j: abs(abs(S[i]) - abs(S[j])))                # unsigned key-signature size difference
    if tokcounts is not None: G["tokcount"] = _pair(lambda i, j: abs(tokcounts[i] - tokcounts[j]))
    if logfreq is not None: G["commonness"] = _pair(lambda i, j: -(logfreq[i] + logfreq[j]))   # rarer pairs = 'farther'
    return G


if __name__ == "__main__":
    print("keys:", KEYS15); print("s:", S.tolist()); print("pc:", PC.tolist()); print("glyph:", GLYPH.tolist())
    G = candidate_geometries()
    iu = np.triu_indices(n, 1)
    from scipy.stats import spearmanr
    names = list(G); print("pairwise Spearman between candidate distance matrices:")
    print("%14s" % "" + "".join("%9s" % nm[:8] for nm in names))
    for a in names: print("%14s" % a + "".join("%9.2f" % spearmanr(G[a][iu], G[b][iu]).correlation for b in names))
    print("enharmonic pairs: circle d =", [G['circle_fifths'][i, j] for i, j in ENH_PAIRS], " line d =", [G['line_fifths'][i, j] for i, j in ENH_PAIRS], " spiral d =", [round(G['spiral_array'][i, j], 2) for i, j in ENH_PAIRS])
    print("KK major-major 1-corr by fifths distance 0..6:", [round(1 - kk_corr(0, (7 * d) % 12), 3) for d in range(7)])
    print("spiral-array key distance by line step 1..8:", [round(np.linalg.norm(sa_major_key(0) - sa_major_key(d)), 3) for d in range(1, 9)])
