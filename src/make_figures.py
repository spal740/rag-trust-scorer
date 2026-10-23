"""Confidence intervals + final figures for the cutoff-repair results, from saved scores only (no GPU).

Input : results/run3_final/scores.npz  (scores of the trained scorer on every set)
Output: results/final/repair_ci.csv, results/final/fig_fpr.png, fig_tpr.png, fig_reliability.png

For each N (labelled target passages, half clean) we repeat DRAWS times: sample N labelled passages from target_pool,
set the cutoff to the 95th percentile of the clean ones, and evaluate on target_test / the harder sets.
Reported per N:  mean over draws; 10-90th percentile band across draws (spread caused by WHICH labels were drawn);
                 95% bootstrap CI of the mean over test passages (spread caused by the finite test set; the draws' cutoffs are held fixed).
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NS = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 700]
SETS_TPR = {"target_test (injected)": None, "reworded": "ood_reworded", "garak": "garak_ood", "garak + 'Haha pwned!'": "ood_garak_haha"}


def cutoffs(rng, pool_z, pool_y, n, draws):
    out = []
    for _ in range(draws):
        i0 = rng.choice(np.where(pool_y == 0)[0], n // 2, replace=False)
        out.append(np.quantile(pool_z[i0], 0.95, method="higher"))
    return np.array(out)


def boot_mean(ind, rng, B):
    """ind: (n_passages, draws) 0/1 matrix.  Returns (2.5th, 97.5th) percentile of the mean when passages are resampled."""
    n = ind.shape[0]
    m = np.array([ind[rng.integers(0, n, n)].mean() for _ in range(B)])
    return np.percentile(m, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/run3_final/scores.npz")
    ap.add_argument("--out", default="results/final")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    S = np.load(a.scores)
    rng = np.random.default_rng(0)
    pz, py = S["tgt_pool_z"], S["tgt_pool_y"]
    tz, ty = S["tgt_test_z"], S["tgt_test_y"]
    hz = S["extra_ood_hardclean_z"]
    tpr_sets = {"target_test (injected)": tz[ty == 1]}
    for label, key in SETS_TPR.items():
        if key:
            tpr_sets[label] = S[f"extra_{key}_z"]
    clean_sets = {"target_test (clean)": tz[ty == 0], "hard clean": hz}

    rows = []
    for n in NS:
        c = cutoffs(rng, pz, py, n, a.draws)
        row = dict(N=n)
        for name, z in {**clean_sets, **tpr_sets}.items():
            ind = (z[:, None] > c[None, :]).astype(float)       # passages x draws
            per_draw = ind.mean(axis=0)
            lo, hi = boot_mean(ind, rng, a.boot)
            row[f"{name}|mean"] = per_draw.mean()
            row[f"{name}|p10"], row[f"{name}|p90"] = np.percentile(per_draw, [10, 90])
            row[f"{name}|ci_lo"], row[f"{name}|ci_hi"] = lo, hi
            if name == "target_test (clean)":
                row["share_FPR<=7.5%"] = float((per_draw <= 0.075).mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"{a.out}/repair_ci.csv", index=False)

    # oracle reference (cutoff = 95th percentile of ALL clean target_test scores)
    orc = np.quantile(tz[ty == 0], 0.95, method="higher")
    oracle = {k: float((v > orc).mean()) for k, v in {**clean_sets, **tpr_sets}.items()}
    pd.Series(oracle).to_csv(f"{a.out}/oracle_reference.csv")

    col = {"a": "#2a78d6", "b": "#eb6834", "c": "#1baf7a", "d": "#6250d6"}
    ink, grid = "#0b0b0b", "#d9d8d4"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#52514e", "axes.labelcolor": ink, "xtick.color": ink, "ytick.color": ink})

    def style(ax):
        ax.set_xscale("log"); ax.set_xticks([10, 20, 50, 100, 200, 500]); ax.set_xticklabels(["10", "20", "50", "100", "200", "500"])
        ax.grid(True, color=grid, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel("labelled target passages N (half clean)")

    x = df["N"].values
    # Figure 1: FPR
    fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=300)
    for name, c, ls, mk in [("target_test (clean)", col["a"], "-", "o"), ("hard clean", col["b"], "--", "s")]:
        ax.fill_between(x, df[f"{name}|p10"] * 100, df[f"{name}|p90"] * 100, color=c, alpha=0.15, lw=0)
        ax.plot(x, df[f"{name}|mean"] * 100, ls, marker=mk, ms=3.5, lw=1.4, color=c, label="target clean" if "target" in name else "hard-clean look-alikes")
    ax.axhline(5, color=ink, lw=0.8, ls=":"); ax.text(11, 5.6, "5% goal", fontsize=7, color=ink)
    ax.set_ylabel("false-positive rate (%)"); ax.set_ylim(0, 45); style(ax); ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(f"{a.out}/fig_fpr.png"); plt.close(fig)

    # Figure 2: TPR on every injected set vs the 96.8% floor
    fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=300)
    spec = [("target_test (injected)", col["a"], "-", "o", "target attacks"), ("reworded", col["b"], "--", "s", "reworded"),
            ("garak", col["c"], "-.", "^", "garak"), ("garak + 'Haha pwned!'", col["d"], ":", "D", "garak + Haha")]
    for name, c, ls, mk, lab in spec:
        ax.plot(x, df[f"{name}|mean"] * 100, ls, marker=mk, ms=3.5, lw=1.4, color=c, label=lab)
    ax.axhline(96.8, color=ink, lw=0.8, ls=":"); ax.text(11, 97.6, "96.8% floor", fontsize=7, color=ink)
    ax.set_ylabel("attacks detected (%)"); ax.set_ylim(35, 102); style(ax); ax.legend(frameon=False, fontsize=6.5, loc="lower left", ncol=2, columnspacing=1.0, handlelength=2.2)
    fig.tight_layout(); fig.savefig(f"{a.out}/fig_tpr.png"); plt.close(fig)

    # Figure 3: reliability
    fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=300)
    ax.plot(x, df["share_FPR<=7.5%"] * 100, "-", marker="o", ms=3.5, lw=1.4, color=col["a"])
    ax.set_ylabel("draws with FPR ≤ 7.5% (%)"); ax.set_ylim(0, 105); style(ax)
    fig.tight_layout(); fig.savefig(f"{a.out}/fig_reliability.png"); plt.close(fig)

    show = ["N", "target_test (clean)|mean", "target_test (clean)|ci_lo", "target_test (clean)|ci_hi", "share_FPR<=7.5%",
            "target_test (injected)|mean", "reworded|mean", "garak|mean", "hard clean|mean"]
    print(df[show].round(3).to_string(index=False))
    print("oracle:", {k: round(v, 3) for k, v in oracle.items()})


if __name__ == "__main__":
    main()
