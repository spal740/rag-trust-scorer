"""Build data/garak_ood.jsonl: 300 injected science passages using attack sentences taken from the garak
repository (garak/probes/latentinjection.py and garak/resources/promptinject/prompt_data.py), with the payload
replaced by a harmless canary word. Needs data/garak_sentences.txt (one sentence per line).
Uses only science passages not already used in any data/*.jsonl file. Label 1 for all (TPR set).
Usage: python build_garak.py
"""
import glob, json, random, re
from datasets import load_dataset
from canary import token, TEST_FORMATS, TEST_WORDS

GARAK_OWN = "Haha pwned!"  # garak's own default payload trigger (garak/probes/latentinjection.py)


def insert(text, sent, rng):
    cuts = sorted(set([0] + [m.end() for m in re.finditer(r"[.!?][\"')]*\s+", text)] + [len(text)]))
    target = rng.choice([0, len(text) // 2, len(text)])
    c = min(cuts, key=lambda x: abs(x - target))
    return " ".join(x for x in (text[:c].rstrip(), sent, text[c:].lstrip()) if x)


def main(name="garak_ood", n=300, seed=2, data="data"):
    rng = random.Random(seed)
    sents = [l.strip() for l in open(f"{data}/garak_sentences.txt") if l.strip()]
    used = set()
    for f in glob.glob(f"{data}/*.jsonl"):
        for l in open(f):
            used.add(json.loads(l)["text"][:80])
    sq = load_dataset("allenai/sciq", split="train")
    pool = [ex["support"].strip() for ex in sq if 40 <= len(ex["support"].split()) <= 200]
    pool = [p for p in pool if p[:80] not in used]
    rng.shuffle(pool)
    print("fresh science passages:", len(pool), "garak sentences:", len(sents))
    assert len(pool) >= n
    rows = [dict(text=insert(p, rng.choice(sents).replace("{c}", GARAK_OWN if (rng.random() < 0.5 and name != "garak_ood") else token(rng, TEST_FORMATS, TEST_WORDS)), rng), label=1) for p in pool[:n]]
    with open(f"{data}/{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("wrote", name)


if __name__ == "__main__":
    import sys
    main(name=sys.argv[1] if len(sys.argv) > 1 else "garak_ood")
