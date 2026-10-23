"""Train the per-passage scorer on SOURCE data only, then save logits for every evaluation set.
Run in Colab (GPU).  pip install transformers datasets accelerate scikit-learn

Logit convention: z = logit[1] - logit[0] = evidence for "injected"; P(safe) = 1 - sigmoid(z).
Output: results/scores.npz in the format sweep.py expects, plus results/in_domain.json.
Usage: python train.py [--model distilbert-base-uncased] [--epochs 2]
"""
import argparse, json, os
import numpy as np, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score

MAXLEN = 256


def load(path):
    rows = [json.loads(l) for l in open(path)]
    return [r["text"] for r in rows], np.array([r["label"] for r in rows])


def batches(tok, texts, labels, bs, shuffle):
    idx = np.arange(len(texts))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        enc = tok([texts[k] for k in j], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(labels[j])
        yield enc


@torch.no_grad()
def logits(model, tok, texts, dev):
    model.eval(); out = []
    for b in batches(tok, texts, np.zeros(len(texts), dtype=int), 64, False):
        b = {k: v.to(dev) for k, v in b.items() if k != "labels"}
        l = model(**b).logits
        out.append((l[:, 1] - l[:, 0]).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2).to(dev)
    xt, yt = load(f"{a.data}/source_train.jsonl")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    steps = a.epochs * ((len(xt) + 15) // 16)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    for ep in range(a.epochs):
        model.train()
        for b in batches(tok, xt, yt, 16, True):
            b = {k: v.to(dev) for k, v in b.items()}
            loss = model(**b).loss
            loss.backward(); opt.step(); sch.step(); opt.zero_grad()
        print("epoch", ep, "last loss", float(loss))

    model.save_pretrained(f"{a.out}/model"); tok.save_pretrained(f"{a.out}/model")
    sets = {n: load(f"{a.data}/{n}.jsonl") for n in
            ["source_val", "source_test_unseen", "target_pool", "target_test"]}
    import glob
    for f in sorted(glob.glob(f"{a.data}/ood_*.jsonl")) + sorted(glob.glob(f"{a.data}/garak_ood.jsonl")):
        sets[os.path.basename(f)[:-6]] = load(f)  # e.g. ood_reworded, ood_hardclean, garak_ood
    Z = {n: (logits(model, tok, x, dev), y) for n, (x, y) in sets.items()}

    np.savez(f"{a.out}/scores.npz",
             src_val_z=Z["source_val"][0], src_val_y=Z["source_val"][1],
             tgt_pool_z=Z["target_pool"][0], tgt_pool_y=Z["target_pool"][1],
             tgt_test_z=Z["target_test"][0], tgt_test_y=Z["target_test"][1],
             **{f"extra_{n}_z": Z[n][0] for n in Z}, **{f"extra_{n}_y": Z[n][1] for n in Z})
    info = {}
    thr = np.quantile(Z["source_val"][0][Z["source_val"][1] == 0], 0.95, method="higher")
    for n, (z, y) in Z.items():
        has0, has1 = bool((y == 0).any()), bool((y == 1).any())
        info[n] = dict(
            auc=float(roc_auc_score(y, z)) if (has0 and has1) else None,  # needs both classes
            fpr_at_src_thr=float((z[y == 0] > thr).mean()) if has0 else None,
            tpr_at_src_thr=float((z[y == 1] > thr).mean()) if has1 else None)
    json.dump(info, open(f"{a.out}/in_domain.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
