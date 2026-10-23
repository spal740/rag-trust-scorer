"""Adaptive attack, end to end: attacker picks, for each question, the injection variant the DEPLOYED scorer rates
safest (query budget --cands), then we run the full RAG pipeline (retriever -> TrustFilter -> Phi) and measure the
real attack success rate (ASR = canary in the answer).
Settings:  for each deployed scorer (base scorer + repaired cutoff; fine-tuned scorer + own cutoff)
           non-adaptive = first random variant     adaptive = lowest-scoring variant out of --cands
Each setting is run with no filter and with the filter.  The attacker optimises only evasion of the scorer
(it does not know whether the LLM will obey), so the no-filter ASR of adaptive variants also shows whether adaptation
cost the attacker effectiveness against the LLM.
Usage (Colab): python src/adaptive_e2e.py --llm microsoft/Phi-3.5-mini-instruct --prompt-style context_last
Output: results/adaptive_e2e.csv
"""
import argparse, json, random
import numpy as np
from rag_pipeline import make_data, make_retriever, make_generator, run, cached
from adaptive_lite import variant_space, build_variants


def apply_variants(clean, items, chosen):
    poisoned = list(clean)
    for it, v in zip(items, chosen):
        poisoned[it["gold"]] = v
    return poisoned


def pick(variants, z, adaptive):
    return variants[int(np.argmin(z))] if adaptive else variants[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=200)
    ap.add_argument("--cands", type=int, default=50)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--ft-n", type=int, default=200)
    ap.add_argument("--llm", default="microsoft/Phi-3.5-mini-instruct")
    ap.add_argument("--prompt-style", default="context_last")
    ap.add_argument("--gen-bs", type=int, default=4)
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="results")
    a = ap.parse_args()
    import torch, pandas as pd
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from train import logits
    import finetune_target as ft

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    items, clean, _ = make_data(a.questions, 0, f"{a.data}/garak_sentences.txt", ["classic", "standard", "reworded", "garak"])
    garak = [l.strip() for l in open(f"{a.data}/garak_sentences.txt") if "{c}" in l]
    space, wordings = variant_space(garak)
    rng = random.Random(0)
    variants = [build_variants(clean[it["gold"]], it["canary"], rng, space, wordings, a.cands) for it in items]
    flat = [v for vs in variants for v in vs]
    n, kk = len(items), len(variants[0])

    S = np.load(f"{a.out}/scores.npz")
    pool_x, pool_y = ft.load(f"{a.data}/target_pool.jsonl")
    tok = AutoTokenizer.from_pretrained(f"{a.out}/model")
    base = AutoModelForSequenceClassification.from_pretrained(f"{a.out}/model").to(dev)
    base_fn = cached(lambda xs: logits(base, tok, xs, dev))
    r0 = np.random.default_rng(0)
    idx = r0.choice(np.where(pool_y == 0)[0], 50, replace=False)
    base_cut = float(np.quantile(S["tgt_pool_z"][idx], 0.95, method="higher"))
    r1 = np.random.default_rng(5000)
    fi, ci = ft.sample_split(r1, pool_y, a.ft_n)
    ft_fn = cached(ft.torch_make_scorer(f"{a.out}/model", 3, 2e-5)([pool_x[i] for i in fi], pool_y[fi], seed=7000))
    ft_cut = float(ft.cutoff_from(ft_fn([pool_x[i] for i in ci]), pool_y[ci]))
    scorers = {"base_repairedN100": (base_fn, base_cut), f"finetuned_N{a.ft_n}": (ft_fn, ft_cut)}

    gen = make_generator(a.llm, a.gen_bs)
    rows = []
    for sname, (fn, cut) in scorers.items():
        z = np.array(fn(flat)).reshape(n, kk)
        for adaptive in (False, True):
            chosen = [pick(variants[i], z[i], adaptive) for i in range(n)]
            poisoned = apply_variants(clean, items, chosen)
            retrieve = make_retriever(items, clean, poisoned, a.topk)
            conds = [dict(name="no_filter", draw=0, score_fn=None, cutoff=None),
                     dict(name="filter", draw=0, score_fn=fn, cutoff=cut)]
            for r in run(items, retrieve, gen, conds, a.topk, a.prompt_style, modes=("attacked",)):
                r.update(scorer=sname, attacker="adaptive" if adaptive else "non-adaptive", cands=kk if adaptive else 1)
                rows.append(r); print(r, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{a.out}/adaptive_e2e.csv", index=False)
    print(df[["scorer", "attacker", "condition", "asr", "poison_blocked", "acc_attacked"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
