"""Recalibration sweep: how many labelled target passages are needed to restore FPR<=5%?

Input: an .npz with arrays (label convention: 1 = injected, 0 = clean; z = logit of "injected")
    src_val_z, src_val_y     source-domain held-out scores (used to set the original 5%-FPR threshold)
    tgt_pool_z, tgt_pool_y   target-domain pool that labelled calibration samples are drawn from
    tgt_test_z, tgt_test_y   target-domain test set (never used for fitting)
Trust score is P(safe) = 1 - sigmoid(z).  A passage is filtered if z > threshold.

Methods
    uncalibrated : threshold fixed on source val at 5% FPR (no target data)
    threshold    : empirical 95th percentile of the clean scores in the N labelled target samples
    platt        : L2 logistic regression on z using the N labelled samples, cut at p=0.5
    platt_thr    : Platt fit, then threshold chosen at 5% FPR on the labelled clean samples (monotone => same
                   decisions as `threshold`; kept only for its calibrated probabilities / ECE)
    batch_shift  : label-free; shift z so target pool mean equals source val mean, then source threshold
Usage: python sweep.py scores.npz out_dir [--ns 20 50 100 200] [--draws 100] [--inj-frac 0.5]
"""
import argparse, json, os
import numpy as np
from sklearn.linear_model import LogisticRegression

TARGET_FPR = 0.05


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def thr_at_fpr(z, y, fpr=TARGET_FPR):
    clean = z[y == 0]
    return np.quantile(clean, 1 - fpr, method="higher")


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(p[m].mean() - y[m].mean())
    return out


def rates(pred, y):
    return pred[y == 0].mean(), pred[y == 1].mean()  # FPR, TPR


def fit_platt(z, y):
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(z.reshape(-1, 1), y)
    return lr


def sample_labelled(rng, z, y, n, inj_frac):
    n_inj = int(round(n * inj_frac))
    n_cl = n - n_inj
    i1 = rng.choice(np.where(y == 1)[0], n_inj, replace=False)
    i0 = rng.choice(np.where(y == 0)[0], n_cl, replace=False)
    idx = np.concatenate([i0, i1])
    return z[idx], y[idx]


def run(d, ns, draws, inj_frac, seed=0):
    rng = np.random.default_rng(seed)
    sz, sy = d["src_val_z"], d["src_val_y"]
    pz, py = d["tgt_pool_z"], d["tgt_pool_y"]
    tz, ty = d["tgt_test_z"], d["tgt_test_y"]
    t_src = thr_at_fpr(sz, sy)
    rows = []

    def add(method, n, draw, fpr, tpr, e):
        rows.append(dict(method=method, n=n, draw=draw, fpr=float(fpr), tpr=float(tpr), ece=float(e)))

    # reference points (no sampling)
    f, t = rates(tz > t_src, ty)
    add("uncalibrated", 0, 0, f, t, ece(sigmoid(tz), ty))
    t_or = thr_at_fpr(tz, ty)  # oracle upper bound: threshold from the full target test set
    f, t = rates(tz > t_or, ty)
    add("oracle_threshold", 0, 0, f, t, np.nan)
    # source-domain sanity
    sf, st = rates(sz > t_src, sy)

    # label-free baseline
    # re-centre using clean-agnostic pool statistics: match median of pool to median of source val
    # (pool prevalence may differ from source; this is a deliberately naive baseline)
    delta = np.median(sz) - np.median(pz)
    f, t = rates(tz + delta > t_src, ty)
    add("batch_shift", 0, 0, f, t, np.nan)

    for n in ns:
        for k in range(draws):
            cz, cy = sample_labelled(rng, pz, py, n, inj_frac)
            # threshold
            thr = thr_at_fpr(cz, cy)
            f, t = rates(tz > thr, ty)
            add("threshold", n, k, f, t, np.nan)
            # platt
            lr = fit_platt(cz, cy)
            p = lr.predict_proba(tz.reshape(-1, 1))[:, 1]
            f, t = rates(p > 0.5, ty)
            add("platt", n, k, f, t, ece(p, ty))
    return rows, dict(src_threshold=float(t_src), src_fpr=float(sf), src_tpr=float(st))


def summarise(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    g = df.groupby(["method", "n"])
    s = g.agg(fpr_mean=("fpr", "mean"), fpr_p10=("fpr", lambda x: np.quantile(x, .1)),
              fpr_p90=("fpr", lambda x: np.quantile(x, .9)), tpr_mean=("tpr", "mean"),
              ece_mean=("ece", "mean"),
              frac_fpr_ok=("fpr", lambda x: np.mean(x <= TARGET_FPR + 0.025))).reset_index()
    return df, s


def plot(s, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for m, c in [("threshold", "C0"), ("platt", "C1")]:
        q = s[s.method == m].sort_values("n")
        ax[0].plot(q.n, q.fpr_mean, "o-", c=c, label=m)
        ax[0].fill_between(q.n, q.fpr_p10, q.fpr_p90, color=c, alpha=.15)
        ax[1].plot(q.n, q.tpr_mean, "o-", c=c, label=m)
    for m, ls in [("uncalibrated", "--"), ("oracle_threshold", ":")]:
        q = s[s.method == m]
        if len(q):
            ax[0].axhline(q.fpr_mean.iloc[0], ls=ls, c="gray", label=m)
            ax[1].axhline(q.tpr_mean.iloc[0], ls=ls, c="gray", label=m)
    ax[0].axhline(TARGET_FPR, c="r", lw=.8)
    ax[0].set_xlabel("labelled target passages N"); ax[0].set_ylabel("target FPR (mean, 10-90%)")
    ax[1].set_xlabel("labelled target passages N"); ax[1].set_ylabel("target TPR")
    ax[0].legend(); fig.tight_layout(); fig.savefig(out, dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scores"); ap.add_argument("out")
    ap.add_argument("--ns", type=int, nargs="+", default=[20, 50, 100, 200])
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--inj-frac", type=float, default=0.5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = np.load(a.scores)
    rows, info = run(d, a.ns, a.draws, a.inj_frac)
    df, s = summarise(rows)
    s.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    json.dump(info, open(os.path.join(a.out, "source_info.json"), "w"))
    plot(s, os.path.join(a.out, "sample_efficiency.png"))
    print(info); print(s.round(3).to_string(index=False))
