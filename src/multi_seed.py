"""Is the main result stable across scorer training seeds?  Retrains the scorer with other seeds (same data, same settings
as train.py), scores every evaluation set, and repeats the cutoff-repair analysis for each seed.

Seed 0 is the scorer already trained (read from results/scores.npz).  Extra seeds are trained here and their scores saved to
results/seeds/seed{k}_scores.npz.  The main model in results/model is NOT touched.
Reported per seed: source AUC, target AUC, FPR at the source cutoff, oracle-cutoff TPR on target/reworded/garak sets,
mean FPR of cutoff repair at N=100 and N=200 (100 draws), share of draws within 7.5%, hard-clean FPR at N=200;
then mean and standard deviation across seeds.
Usage (Colab, GPU): python src/multi_seed.py --seeds 1 2          Output: results/multi_seed.csv
"""
import argparse, glob, json, os
import numpy as np

TPR_SETS = {"tpr_target": None, "tpr_reworded": "ood_reworded", "tpr_garak": "garak_ood", "tpr_garak_haha": "ood_garak_haha"}


def summarise(S, draws=100, seed=0):
    """S: dict-like with src_val_z/y, tgt_pool_z/y, tgt_test_z/y, extra_<set>_z. Returns a dict of metrics."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    sz, sy = S["src_val_z"], S["src_val_y"]
    pz, py, tz, ty = S["tgt_pool_z"], S["tgt_pool_y"], S["tgt_test_z"], S["tgt_test_y"]
    q = lambda v: np.quantile(v, 0.95, method="higher")
    src_cut = q(sz[sy == 0])
    orc = q(tz[ty == 0])
    out = dict(auc_source=roc_auc_score(sy, sz), auc_target=roc_auc_score(ty, tz),
               fpr_target_at_source_cut=float((tz[ty == 0] > src_cut).mean()),
               tpr_target_oracle=float((tz[ty == 1] > orc).mean()))
    for name, key in TPR_SETS.items():
        if key and f"extra_{key}_z" in S:
            out[name + "_oracle"] = float((S[f"extra_{key}_z"] > orc).mean())
    hard = S["extra_ood_hardclean_z"] if "extra_ood_hardclean_z" in S else None
    for n in (100, 200):
        cuts = np.array([q(pz[rng.choice(np.where(py == 0)[0], n // 2, replace=False)]) for _ in range(draws)])
        fprs = (tz[ty == 0][:, None] > cuts[None, :]).mean(axis=0)
        out[f"repair_fpr_N{n}"] = float(fprs.mean())
        out[f"share_within_7.5_N{n}"] = float((fprs <= 0.075).mean())
        if n == 200 and hard is not None:
            out["hardclean_fpr_N200"] = float((hard[:, None] > cuts[None, :]).mean())
    return out


def train_seed(seed, data, epochs=2, model_name="distilbert-base-uncased"):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    from train import load, batches, logits
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed); np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(dev)
    xt, yt = load(f"{data}/source_train.jsonl")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    steps = epochs * ((len(xt) + 15) // 16)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    for ep in range(epochs):
        model.train()
        for b in batches(tok, xt, yt, 16, True):
            b = {k: v.to(dev) for k, v in b.items()}
            loss = model(**b).loss
            loss.backward(); opt.step(); sch.step(); opt.zero_grad()
        print("seed", seed, "epoch", ep, "last loss", float(loss), flush=True)
    sets = {n: load(f"{data}/{n}.jsonl") for n in ["source_val", "source_test_unseen", "target_pool", "target_test"]}
    for f in sorted(glob.glob(f"{data}/ood_*.jsonl")) + sorted(glob.glob(f"{data}/garak_ood.jsonl")):
        sets[os.path.basename(f)[:-6]] = load(f)
    Z = {n: (logits(model, tok, x, dev), y) for n, (x, y) in sets.items()}
    S = dict(src_val_z=Z["source_val"][0], src_val_y=Z["source_val"][1], tgt_pool_z=Z["target_pool"][0],
             tgt_pool_y=Z["target_pool"][1], tgt_test_z=Z["target_test"][0], tgt_test_y=Z["target_test"][1])
    S.update({f"extra_{n}_z": Z[n][0] for n in Z}); S.update({f"extra_{n}_y": Z[n][1] for n in Z})
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="results")
    a = ap.parse_args()
    import pandas as pd
    os.makedirs(f"{a.out}/seeds", exist_ok=True)
    rows = []
    base = dict(np.load(f"{a.out}/scores.npz"))
    rows.append(dict(seed=0, **summarise(base)))
    for s in a.seeds:
        S = train_seed(s, a.data)
        np.savez(f"{a.out}/seeds/seed{s}_scores.npz", **S)
        rows.append(dict(seed=s, **summarise(S)))
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(f"{a.out}/multi_seed.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(f"{a.out}/multi_seed.csv", index=False)
    cols = [c for c in df.columns if c != "seed"]
    summ = pd.DataFrame({"mean": df[cols].mean(), "sd": df[cols].std(ddof=1)})
    print(df.round(3).T.to_string())
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
