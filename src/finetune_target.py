"""Fine-tuning comparison: do N labelled target (science) passages fix detection if we RETRAIN the scorer on them,
instead of only moving the cutoff?

For each N and draw:
  1. sample N labelled passages from data/target_pool.jsonl (half clean, half injected)
  2. split them 70/30: the 70% fine-tune the source-trained scorer (results/model), the 30% choose the cutoff
     (95th percentile of the clean scores in that 30%, i.e. 5% FPR), so all N labels are used and none are shared
  3. evaluate on target_test and every ood_*/garak set (never used for fitting)
Compare with sweep_extra.py (cutoff-only repair using the same N labels).
Usage (Colab, after train.py):  python src/finetune_target.py --ns 20 50 100 200 500 --draws 3
Output: results/finetune_results.csv and a printed summary.
"""
import argparse, glob, json, os
import numpy as np

TARGET_FPR = 0.05


def load(path):
    rows = [json.loads(l) for l in open(path)]
    return [r["text"] for r in rows], np.array([r["label"] for r in rows])


def sample_split(rng, y, n, ft_frac=0.7):
    """Return (finetune_idx, cutoff_idx): n labelled passages, half per class, split 70/30 within each class."""
    ft, cal = [], []
    for c in (0, 1):
        idx = rng.choice(np.where(y == c)[0], n // 2, replace=False)
        k = max(1, int(round(len(idx) * ft_frac)))
        ft.extend(idx[:k]); cal.extend(idx[k:])
    return np.array(ft), np.array(cal)


def cutoff_from(z, y):
    clean = z[y == 0]
    if len(clean) == 0:
        return np.inf
    return np.quantile(clean, 1 - TARGET_FPR, method="higher")


def evaluate(score_fn, thr, tgt_test, ood, floor):
    """Metrics at the labelled-sample cutoff `thr` (realistic) and at the oracle cutoff (5% FPR on target_test,
    suffix _orc), which isolates ranking quality from cutoff-estimation noise."""
    from sklearn.metrics import roc_auc_score
    tx, ty = tgt_test
    tz = score_fn(tx)
    thr_orc = cutoff_from(tz, ty)
    r = dict(auc_target=float(roc_auc_score(ty, tz)))
    zs = {"target": (tz, ty)}
    for name, (x, y) in ood.items():
        zs[name] = (score_fn(x), y)
    ok = True
    for name, (z, y) in zs.items():
        for suffix, t in (("", thr), ("_orc", thr_orc)):
            if name == "target":
                r["fpr_target" + suffix] = float((z[y == 0] > t).mean()); r["tpr_target" + suffix] = float((z[y == 1] > t).mean())
            elif y.min() == 1:
                r[f"tpr_{name}{suffix}"] = float((z > t).mean())
            else:
                r[f"fpr_{name}{suffix}"] = float((z > t).mean())
    ok = r["fpr_target"] <= 0.075 and r["tpr_target"] >= floor
    for name, (z, y) in zs.items():
        if name == "target":
            continue
        ok &= (r[f"tpr_{name}"] >= floor) if y.min() == 1 else (r[f"fpr_{name}"] <= 0.075)
    r["success"] = bool(ok)
    return r


def run_experiment(pool, tgt_test, ood, make_scorer, ns, draws, floor, seed=0):
    """make_scorer(texts, labels, seed) -> score_fn(list_of_texts) -> np.array of logits (fine-tuned model)."""
    rng = np.random.default_rng(seed)
    px, py = pool
    rows = []
    for n in ns:
        for k in range(draws):
            ft, cal = sample_split(rng, py, n)
            score_fn = make_scorer([px[i] for i in ft], py[ft], seed=1000 * n + k)
            cz = score_fn([px[i] for i in cal])
            thr = cutoff_from(cz, py[cal])
            row = dict(n=n, draw=k, thr=float(thr), n_ft=len(ft), n_cal=len(cal))
            row.update(evaluate(score_fn, thr, tgt_test, ood, floor))
            rows.append(row)
            print(n, k, {a: round(b, 3) for a, b in row.items() if a.startswith(("fpr", "tpr", "auc")) and not a.endswith("_orc")}, flush=True)
    return rows


def torch_make_scorer(model_dir, epochs, lr):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    from train import batches, logits
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)

    def make(texts, labels, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        steps = max(1, epochs * ((len(texts) + 15) // 16))
        sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        for _ in range(epochs):
            model.train()
            for b in batches(tok, texts, labels, 16, True):
                b = {k: v.to(dev) for k, v in b.items()}
                model(**b).loss.backward(); opt.step(); sch.step(); opt.zero_grad()
        return lambda xs: logits(model, tok, xs, dev)
    return make


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[50, 100, 200, 500])
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--floor", type=float, default=0.968)
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="results")
    a = ap.parse_args()
    pool = load(f"{a.data}/target_pool.jsonl")
    tgt_test = load(f"{a.data}/target_test.jsonl")
    ood = {os.path.basename(f)[:-6].replace("ood_", ""): load(f)
           for f in sorted(glob.glob(f"{a.data}/ood_*.jsonl")) + sorted(glob.glob(f"{a.data}/garak_ood.jsonl"))}
    rows = run_experiment(pool, tgt_test, ood, torch_make_scorer(f"{a.out}/model", a.epochs, a.lr),
                          a.ns, a.draws, a.floor)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(f"{a.out}/finetune_results.csv", index=False)
    cols = [c for c in df.columns if c.startswith(("fpr_", "tpr_", "auc_"))] + ["success"]
    print(df.groupby("n")[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
