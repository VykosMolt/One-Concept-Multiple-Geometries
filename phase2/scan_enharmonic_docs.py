"""Targeted pass: documents mentioning a rare spelling (Cb/Gb/C#/Db/F# major); for each co-mention of an enharmonic pair
within 16 words, record the intervening text and whether 'enharmonic' appears within the window. Output counts."""
import sys, os, re, json
sys.path.insert(0, ".")
import pyarrow.parquet as pq
from corpus.concepts import KEY_RE, key_name, is_sentence_initial
PAIRS = {("Cb", "B"), ("B", "Cb"), ("Gb", "F#"), ("F#", "Gb"), ("Db", "C#"), ("C#", "Db")}
def scan(path):
    out = []; pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=2000, columns=["text"]):
        for text in batch.column("text").to_pylist():
            if not any(s in text for s in ("Cb major", "Gb major", "C# major", "C-flat major", "G-flat major", "C-sharp major", "C♭ major", "G♭ major", "C♯ major")): continue
            ms = [(m.start(), key_name(m)) for m in KEY_RE.finditer(text) if m.group("mode") == "major" and not is_sentence_initial(text, m.start())]
            for i in range(len(ms)):
                for j in range(i + 1, len(ms)):
                    if (ms[i][1], ms[j][1]) in PAIRS and ms[j][0] - ms[i][0] < 140:
                        seg = text[max(0, ms[i][0] - 60): ms[j][0] + 40].replace("\n", " ")
                        out.append({"pair": ms[i][1] + "|" + ms[j][1], "enharmonic": "enharmonic" in seg.lower(), "seg": seg})
    return out
if __name__ == "__main__":
    import multiprocessing as mp
    files = sorted(os.path.join("data/wiki", f) for f in os.listdir("data/wiki") if f.endswith(".parquet"))
    with mp.Pool(20) as pool: res = pool.map(scan, files)
    allr = [r for rr in res for r in rr]; json.dump(allr, open("results/phase2/corpus/enharmonic_pairs_context.json", "w"), indent=1)
    from collections import Counter
    c = Counter((r["pair"], r["enharmonic"]) for r in allr); print("co-mentions of enharmonic pairs (<140 chars apart):", dict(c))
    print("fraction with 'enharmonic' in the window:", sum(r["enharmonic"] for r in allr) / max(1, len(allr)))
    for r in allr[:12]: print("  -", r["pair"], "|", r["seg"][:160])
