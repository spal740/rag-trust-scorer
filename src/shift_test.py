"""Length and position shift: does the repaired cutoff still work when passages are shorter/longer than those we calibrated on
and the injected sentence sits at the start, middle or end?

Calibration is done ONCE on unmodified target passages (cutoff repaired from N=100 labelled passages, 20 draws averaged, plus the
oracle cutoff).  We then evaluate on modified copies of held-out target_test passages:
   length: short (first sentences, >= 25 words) | original | long (the passage + two other clean passages, often > 256 tokens)
   position of the attack sentence: start | middle | end
Attack wordings are the held-out ones (standard + reworded + garak) with held-out canary formats.
Reported: FPR of clean passages in each length bucket; TPR for every (length, position); share of attacks whose sentence starts
beyond the scorer's 256-token window (those cannot be seen: a real design limit, not a modelling failure).
Usage (Colab): python src/shift_test.py             Output: results/shift_test.csv
"""
import argparse, random, re
import numpy as np

from adaptive_lite import STANDARD, REWORDED, insert_at

LENGTHS = ["short", "original", "long"]
POSITIONS = ["start", "mid", "end"]


def make_length(texts, kind, rng):
    if kind == "original":
        return list(texts)
    if kind == "short":
        out = []
        for t in texts:
            sents = re.split(r"(?<=[.!?])\s+", t.strip())
            acc = []
            for s in sents:
                acc.append(s)
                if len(" ".join(acc).split()) >= 25:
                    break
            out.append(" ".join(acc))
        return out
    if kind == "long":
        out = []
        for t in texts:
            extra = rng.sample(texts, 2)
            out.append(" ".join([t] + extra))
        return out
    raise ValueError(kind)


def inject(texts, pos, wordings, rng, tokenfn):
    res, prefixes = [], []
    for t in texts:
        sent = rng.choice(wordings).format(c=tokenfn(rng))
        new = insert_at(t, sent, pos)
        res.append(new)
        k = new.find(sent)
        prefixes.append(new[:k] if k >= 0 else new)
    return res, prefixes


def evaluate(score_fn, clean, cuts, wordings, tokenfn, n_tokens, rng, n_per=200):
    """cuts: dict name -> cutoff.  n_tokens(text) -> token count.  Returns list of result rows."""
    rows = []
    base = rng.sample(clean, min(len(clean), 3 * n_per))
    for L in LENGTHS:
        pool = make_length(base[:n_per], L, rng) if L != "long" else make_length(base[:n_per], L, rng)
        z_clean = np.asarray(score_fn(pool))
        for cname, c in cuts.items():
            rows.append(dict(length=L, position="clean", cutoff=cname, metric="FPR", value=float((z_clean > c).mean()), n=len(pool)))
        for P in POSITIONS:
            inj, pref = inject(pool, P, wordings, rng, tokenfn)
            z = np.asarray(score_fn(inj))
            visible = np.array([n_tokens(p) < 250 for p in pref])
            for cname, c in cuts.items():
                rows.append(dict(length=L, position=P, cutoff=cname, metric="TPR", value=float((z > c).mean()), n=len(inj)))
            rows.append(dict(length=L, position=P, cutoff="-", metric="share_attack_inside_256_tokens", value=float(visible.mean()), n=len(inj)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="results")
    a = ap.parse_args()
    import torch, pandas as pd
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from train import load, logits
    from canary import token, TEST_FORMATS, TEST_WORDS
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(f"{a.out}/model")
    model = AutoModelForSequenceClassification.from_pretrained(f"{a.out}/model").to(dev)
    score_fn = lambda xs: logits(model, tok, list(xs), dev)
    n_tokens = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])
    garak = [l.strip() for l in open(f"{a.data}/garak_sentences.txt") if "{c}" in l]
    wordings = STANDARD + REWORDED + garak

    S = np.load(f"{a.out}/scores.npz")
    q = lambda v: float(np.quantile(v, 0.95, method="higher"))
    prng = np.random.default_rng(0)
    pz, py = S["tgt_pool_z"], S["tgt_pool_y"]
    rep = np.mean([q(pz[prng.choice(np.where(py == 0)[0], 50, replace=False)]) for _ in range(20)])
    cuts = {"repaired N=100": float(rep), "oracle": q(S["tgt_test_z"][S["tgt_test_y"] == 0])}
    tx, ty = load(f"{a.data}/target_test.jsonl")
    clean = [t for t, y in zip(tx, ty) if y == 0]
    rng = random.Random(0)
    rows = evaluate(score_fn, clean, cuts, wordings, lambda r: token(r, TEST_FORMATS, TEST_WORDS), n_tokens, rng, a.n)
    df = pd.DataFrame(rows)
    df.to_csv(f"{a.out}/shift_test.csv", index=False)
    print("cutoffs:", cuts)
    print(df[df.metric == "FPR"].pivot_table(index="length", columns="cutoff", values="value").round(3).to_string())
    print(df[df.metric == "TPR"].pivot_table(index=["length", "position"], columns="cutoff", values="value").round(3).to_string())
    print(df[df.metric.str.startswith("share")].pivot_table(index="length", columns="position", values="value").round(3).to_string())


if __name__ == "__main__":
    main()
