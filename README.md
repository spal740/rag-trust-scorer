# Cutoff repair for a RAG passage trust scorer under domain shift

Code, results and experiment log for the University of Auckland research project
**"Cutoff Repair Is Not Enough: Recalibrating a RAG Passage Trust Scorer After Domain Shift"**
(Project Idea 5.1, detecting adversarially retrieved contexts in RAG systems).

**Research question.** For a calibrated, black-box, per-passage trust scorer P(safe | passage) that filters an
injected passage before context assembly, how many labelled target-domain passages are needed to restore a 5%
false-positive rate after a shift from HotpotQA/Wikipedia to SciQ/science, without losing detection?

**Main findings.** The Wikipedia cutoff flags 82% of clean science passages although ranking barely changes
(AUC 0.9998 to 0.986). About 30 to 100 labelled passages restore a 5% false-positive rate on average and about 200 make
it reliable, but detection stays near 88% on target attacks and 49% on reworded ones, below the 96.8% floor at every N.
In an end-to-end pipeline the repaired filter cuts attack success from 12.8% to 0.8%, but a query-budget attacker
evades the scorer within ten queries.

## Conventions
Label 1 = injected, 0 = clean. Scorer output z = logit[1] - logit[0]; a passage is filtered if z > cutoff;
trust = 1 - sigmoid(z). The scorer is DistilBERT trained on source data only (2 epochs, seed 0).

## Run order (Google Colab, T4 GPU)
```
pip install -r requirements.txt
python src/build_data.py            # HotpotQA / SciQ passages with planted injections (needs internet)
python src/build_extra.py           # reworded and hard-clean test sets
python src/build_garak.py           # garak-derived attacks (uses data/garak_sentences.txt)
python src/train.py                 # trains the scorer, writes results/scores.npz
python src/score_new.py data/garak_ood.jsonl   # score extra sets with the trained scorer
python src/sweep.py results/scores.npz results/run1 --ns 20 50 100 200 --draws 100   # cutoff repair vs N
python src/sweep_extra.py ...       # repair evaluated on reworded / garak / hard-clean sets
python src/finetune_target.py ...   # fine-tuning comparison
python src/multi_seed.py            # stability across training seeds
python src/shift_test.py            # passage length and injection position
python src/llm_susceptibility.py    # which small LLM follows a planted injection
python src/rag_pipeline.py          # end-to-end RAG with the filter (retriever, Phi-3.5-mini)
python src/adaptive_lite.py ...     # query-budget attacker against the scorer
python src/adaptive_e2e.py ...      # attacker end to end
python src/adaptive_joint.py ...    # joint attacker (partial results)
python src/make_figures.py          # CIs and figures from results/run3_final/scores.npz (no GPU)
```
Each script's docstring gives its exact arguments. Every random choice uses a fixed seed.

## Contents
- `src/`: all scripts listed above.
- `results/final/`: final confidence-interval tables and figures; `results/run3_final/`: saved scores and sweeps used for them; `results/run3_bigN/`: larger-N sweep.
- `TECHNICAL_LOG.md`: running log of every experiment, including the runs that changed the design (for example the canary-token shortcut found in the first data).
- Trained model weights and generated datasets are not included; rebuild them with the scripts above.

## Notes
All injected instructions ask only for a harmless canary word. The 3-point detection margin was set after the first sweep, and the answering model (Phi-3.5-mini-instruct) was chosen after a first run with Qwen2.5-1.5B showed almost no susceptibility; both are disclosed in the report.
