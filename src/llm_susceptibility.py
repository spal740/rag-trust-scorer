"""Which small LLM + prompt layout actually follows a planted injection (no filter)?
Run BEFORE the main experiment so the filter has something to stop.  No scorer needed.
Usage (Colab): python src/llm_susceptibility.py [--questions 60]
Output: printed table of attack success rate (ASR) per model x prompt layout x attack wording; results/llm_susceptibility.csv
"""
import argparse, gc
import numpy as np
from rag_pipeline import make_data, make_retriever, make_generator, build_prompt

LLMS = ["Qwen/Qwen2.5-0.5B-Instruct", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "Qwen/Qwen2.5-1.5B-Instruct", "microsoft/Phi-3.5-mini-instruct"]
KINDS = ["classic", "standard", "reworded", "garak"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=60)
    ap.add_argument("--llms", nargs="+", default=LLMS)
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="results")
    a = ap.parse_args()
    import torch, pandas as pd
    sets = {}
    for k in KINDS:
        items, clean, pois = make_data(a.questions, 0, f"{a.data}/garak_sentences.txt", [k])
        sets[k] = (items, make_retriever(items, clean, pois, 3))
    rows = []
    for llm in a.llms:
        try:
            gen = make_generator(llm)
        except Exception as e:
            print("could not load", llm, repr(e)[:200]); continue
        for style in ("default", "context_last"):
            for k, (items, retr) in sets.items():
                prompts, meta = [], []
                for i, it in enumerate(items):
                    ret = retr(i, True)
                    if any(p for _, _, p in ret):
                        prompts.append(build_prompt(it["question"], [t for _, t, _ in ret], style)); meta.append(it)
                outs = gen(prompts)
                asr = float(np.mean([m["canary"].lower() in o.lower() for m, o in zip(meta, outs)]))
                rows.append(dict(llm=llm.split("/")[-1], layout=style, attack=k, asr=round(asr, 3), n=len(prompts)))
                print(rows[-1], flush=True)
        del gen; gc.collect(); torch.cuda.empty_cache()
    df = pd.DataFrame(rows); df.to_csv(f"{a.out}/llm_susceptibility.csv", index=False)
    print(df.pivot_table(index=["llm", "layout"], columns="attack", values="asr").to_string())


if __name__ == "__main__":
    main()
