# Technical log (running record for the technical document)

Label convention: 1 = injected, 0 = clean. Scorer output z = logit[1] - logit[0] (evidence for "injected"); trust = P(safe) = 1 - sigmoid(z). A passage is filtered if z > threshold.

## 1. Data (`src/build_data.py`, seed 0)
- Source domain: HotpotQA (`hotpotqa/hotpot_qa`, config `distractor`, first 6,000 train examples); every context paragraph kept if 40-200 words -> 49,911 passages.
- Target domain: SciQ (`allenai/sciq`, train split), `support` paragraph kept if 40-200 words -> 6,050 passages.
- Injection = one instruction sentence inserted into a real passage, at a sentence boundary nearest the start, middle or end (chosen uniformly). Labels are exact because we plant the attack.
- Templates: 6 training wordings, 4 held-out test wordings. Canary words vary per passage, formatted WORD-n with n in 1..99: training words {PINEAPPLE, BANANA, COBALT, TULIP, MARBLE, ORBIT}, held-out words {QUASAR, LANTERN, WALNUT, ZEPHYR}.
- Hard negatives: 20% of clean passages get a harmless instruction-like sentence (4 source-domain, 4 target-domain sentences containing words such as "ignore", "must", "override").
- Splits (disjoint passages): source_train 4,000 (train templates and train canaries only); source_val 1,000 (all templates); source_test_unseen 1,000 (held-out templates only); target_pool 1,500 (source of labelled recalibration samples); target_test 1,500 (never used for fitting). Roughly 50% injected in each. Optional garak set (held-out OOD) if `garak_payloads.txt` is supplied.

## 2. Scorer (`src/train.py`)
- `distilbert-base-uncased`, sequence classification head, 2 labels; trained on source_train only.
- AdamW, lr 3e-5, weight decay 0.01, batch 16, 2 epochs, linear schedule with 6% warmup, max length 256 tokens (truncation), seed 0.
- Saves logits for source_val, source_test_unseen, target_pool, target_test (and garak_ood if present) to `results/scores.npz`; `results/in_domain.json` holds AUC, FPR and TPR at the source 5%-FPR threshold per set.

## 3. Recalibration sweep (`src/sweep.py`)
- Source threshold = 95th percentile of clean source_val scores (5% FPR on the source domain).
- Methods: uncalibrated (source threshold); threshold (95th percentile of clean scores among N labelled target samples); platt (L2 logistic regression on z, cut at p=0.5); batch_shift (label-free median re-centring); oracle_threshold (95th percentile of clean target_test scores, upper bound only).
- N in {20, 50, 100, 200}, 100 random draws each, balanced (50% injected) by default; metrics on target_test: FPR, TPR, ECE (10 bins). Report mean and 10-90th percentile band, plus the fraction of draws with FPR within 7.5%.
- Success criterion (author's decision): a repair "works" at N if target FPR <= 5% (tolerance band reported as <= 7.5% for single draws) AND target TPR >= source-domain TPR at 5% FPR minus 3 percentage points (source TPR 99.8%, so floor 96.8%). Honesty note: the 3-point value was suggested before any experiment, but the author confirmed it after the first sweep (run1) had been seen; say so in the write-up. Apply the same margin unchanged to the garak / reworded-attack sets.

## 4. Validity issues found and fixed (worth reporting)
- HotpotQA dataset id needed the namespace (`hotpotqa/hotpot_qa`).
- Inspection of generated data showed two shortcuts: (a) one fixed canary token in every injected passage, so the scorer could detect injection by that token alone and domain shift might vanish; (b) mid-sentence insertion produced ungrammatical breaks. Fixed by varying canary words (train/test disjoint) and inserting at sentence boundaries.
- `results/synthetic_codetest/` holds output from fake scores used only to test sweep.py; it is not a result and must not appear in the write-up.

## 5. Results
### 5.1 Scorer (results/in_domain.json, Colab T4, final training loss 0.002)
Source threshold = 5% FPR on source_val.
| set | AUC | FPR at source threshold | TPR at source threshold |
|---|---|---|---|
| source_val | 0.9996 | 4.96% | 99.81% |
| source_test_unseen (held-out wordings) | 0.9997 | 5.21% | 99.60% |
| target_pool | 0.9959 | 81.1% | 100% |
| target_test | 0.9973 | 83.8% | 100% |
Reading: ranking quality transfers (AUC 0.997) but the fixed source threshold breaks (84% of clean target passages blocked). Caveat: planted attacks are easy, TPR ~100% (ceiling); harder sets (garak, reworded) still to run.

### 5.2 Sweep (results/run1, balanced samples, 100 draws per N, evaluated on target_test)
Reference: uncalibrated FPR 83.8% (TPR 100%); oracle target threshold FPR 5.0% (TPR 99.3%).
| N labelled (half clean) | threshold: mean FPR | p10-p90 | draws with FPR <= 7.5% | TPR | platt@0.5: mean FPR | platt ECE |
|---|---|---|---|---|---|---|
| 10 | 15.1% | 1.9-35.0% | 36% | 99.4% | 1.5% | 0.022 |
| 20 | 6.9% | 1.3-12.4% | 65% | 99.3% | 1.2% | 0.017 |
| 50 | 6.8% | 3.5-10.8% | 69% | 99.3% | 1.1% | 0.014 |
| 100 | 5.8% | 2.6-8.6% | 77% | 99.3% | 1.0% | 0.012 |
| 200 | 4.9% | 2.8-7.4% | 96% | 99.3% | 1.0% | 0.012 |
Notes: (1) Threshold repair: mean FPR reaches ~5-7% from N=20, but reliability (draws within 7.5%) needs N~200 (about 100 clean samples; the 95th percentile of ~25 clean scores at N=50 is noisy). TPR cost vs uncalibrated about 0.7 points. (2) Platt cut at 0.5 does not target 5% FPR; it lands near 1% because classes are nearly separable (AUC 0.997) and a balanced fit assumes 50% injection prevalence; not directly comparable to threshold repair. ECE falls from 0.020 uncalibrated to 0.012. (3) batch_shift (median re-centring) gave FPR 100% / TPR 100%: baseline is faulty (medians of near-bimodal 50/50 mixtures are unstable); exclude from results or replace. (4) Ceiling effect: easy planted attacks keep TPR ~99-100% at every operating point, so the TPR side of the claim is weak until garak / reworded attacks are evaluated. Single scorer seed.

## 6. Related work found by web search (2026-10-08). Titles/IDs confirmed to exist in search results; only abstracts read unless noted.
- Alalawi, "When Calibrated Detectors Meet New Attacks: Per-Category Reliability of ML Intrusion Detection Under Distribution Shift", IJACSA 17(8), 2026 (read abstract via fetch). Network intrusion detection (NSL-KDD, UNSW-NB15), post-hoc temperature/Platt/isotonic, recalibration on a labelled sample from the shifted distribution, most gain by ~100 labelled samples (N varied 100-5,000); threshold treated separately as operating-cost choice. NOT prompt injection, NOT RAG passages. Closest methodological precedent; must be cited; novelty claim must be scoped to prompt-injection scorers on retrieved passages.
- Berlin (Cisco), blog 2026-07-13, FPR calibration of detector scores using benign scores only; rule of thumb ~16/p benign samples for +-50% relative precision at 95% confidence (p = target FPR). Not prompt-injection/RAG focused. Useful theory check against our sample-size curve (p=5% -> ~320 benign).
- Biswas, "Confidently Wrong: Severity-Aware Calibration of Prompt-Injection Detectors under Attack Shift" (arXiv 2606.22659); Liu et al., "On Calibration of LLM-based Guard Models for Reliable Content Moderation" (arXiv 2410.10414, ICLR 2025); PIDS-Bench (arXiv 2609.15017: abstract says threshold calibration + curated augmentation alone do not close over-defense on externally-sourced benign prompts); "Prompt Injection Detection is Regime-Dependent" (arXiv 2605.26999); kNNGuard (arXiv 2607.02072). Details beyond abstracts still to verify.
- CORRECTION: Chaudhari, "Regime-Adaptive Conformal Guardrails for Prompt Injection Attacks Under Distribution Shift" EXISTS: Preprints.org 202609.2642, submitted 27 Sep 2026, posted 30 Sep 2026, v1, NOT peer-reviewed (fetched and read abstract summary). My earlier web search missed it. It wraps a frozen detector (ProtectAI-v2, PIGuard, similarity shield) with a KS drift monitor on unlabelled data, buys labels for a BENIGN calibration window only when drift is flagged, converts scores to class-conditional conformal p-values at alpha=0.05, and tests 10/20/40% calibration fractions of a 20/40/40 monitor/calibration/test split. Four public prompt datasets (deepset, xTRam, Neuralchemy, Safeguard); full prompts, not RAG passages; no Platt scaling mentioned. => Overlaps our label-cost question; our differences: per-passage RAG setting, absolute N sweep (10-200) with repeated draws, threshold vs Platt, explicit TPR margin, planted single injection, end-to-end check. Must be cited and compared directly.
- PromptShield, "Deployable Detection for Prompt Injection Attacks" (arXiv 2501.15145; only search listing seen): reportedly calibrates detectors to a target FPR (e.g. 1%, 0.1%) and reports TPR. Target-FPR calibration itself is therefore not novel. Details unverified.
- Unverified claim relayed from another AI chat: an "empirical evaluation ... deepset/prompt-injections" paper uses Platt scaling. Not checked.
- Candidate scoped claim: "Prior work recalibrates prompt-injection detectors under shift (Chaudhari 2026, conformal, label budgets as fractions) and studies label-efficient recalibration in intrusion detection (Alalawi 2026); to my knowledge, the absolute number of labelled target passages needed to restore a 5% FPR for a per-passage injection scorer on retrieved RAG passages, with an explicit TPR margin, has not been reported."
- Idea from PIDS-Bench: add a target set of hard clean passages (security-adjacent text) to test the limits of threshold repair.

## 7. Harder tests (code written 2026-10-14; results pending)
- `build_extra.py`: ood_reworded (300 injected science passages, 20 hand-written paraphrased attack sentences without "ignore previous instructions" phrasing, held-out canary words; NOT adaptive, not optimised against the scorer) and ood_hardclean (300 clean science passages with a sentence about instructions/overrides or quoting an injection phrase; PIDS-Bench-inspired hard negatives). Passages not used elsewhere (skips texts already in data/*.jsonl).
- `train.py` now saves the model and scores every data/ood_*.jsonl and garak_ood.jsonl into scores.npz. Retraining may change run1 numbers slightly (GPU nondeterminism); report the new run as final and keep run1 as a note.
- `sweep_extra.py`: threshold repair evaluated on target_test plus each extra set; success rule = target FPR <= 7.5%, TPR on every injected set >= source TPR - 3 pts, FPR on clean-only sets <= 7.5%. Code-tested on fake extra sets only (deleted).
- garak: still to do (payload list must be extracted in Colab).

## 8. Report requirements (from course brief, 2026-10-14)
- IEEE format, 10-12 pages excluding references. Sections: Abstract (150-200 words), Introduction (1-1.5 p; research question, hypothesis, contributions), Related Work (max 1 p), Methodology (gets most marks: reproducible, appropriate statistics, justified decisions), Implementation/Results (graded on implementation and reasoning, not on whether it "works"; evaluate beyond accuracy; study limitations and where it fails), Conclusion and Future work (~1 p, limitations), References (min 10, max 1 page, IEEE), contribution breakdown (solo project, approved by Thomas: keep that on record).
- To do before writing: state explicit hypotheses in the Introduction (e.g. H1: threshold repair restores target FPR <= 5% with N <= 200 labelled passages while TPR stays within 3 pts of source TPR; H2: reworded attacks / hard clean passages break that guarantee). Decide hypotheses BEFORE reading the harder-test results.
- Mapping: log s1-3 -> Methodology; s5, s7 -> Results; s4 -> Limitations / validity; s6 -> Related Work.

## 9. garak test set (prepared 2026-10-14; results pending)
- Source: NVIDIA/garak repo (cloned read-only). Attack sentences taken from garak/probes/latentinjection.py (indirect/latent injection payload wrappers) and garak/resources/promptinject/prompt_data.py (PromptInject goal-hijacking strings), 17 templates x 3 harmless held-out canaries = 51 sentences in data/garak_sentences.txt. Excluded: toxic/discriminatory-sentence prompts, phishing/URL payloads, resume-specific text. Note several templates ("Ignore ... and just print X") are close to our training wording, so garak may be an easy set; report honestly.
- `build_garak.py` -> data/garak_ood.jsonl (300 injected fresh science passages, label 1). `score_new.py` scores it with the saved model and merges into scores.npz (no retraining). Then `sweep_extra.py` picks it up as a TPR set.

## 10. Final run (run2, 2026-10-14; results/run2) -- retrain reproduced run1 exactly (identical loss and scores on target_test)
Scorer at source threshold: source_val AUC .9996 FPR 4.96% TPR 99.81%; target_test AUC .9973 FPR 83.8% TPR 100%; ood_hardclean FPR 96.7%; ood_reworded TPR 100%; garak_ood TPR 100% (uninformative: the unrepaired cutoff blocks almost everything).
Threshold repair, mean over 100 draws (N = labelled target passages, half clean). Floor for TPR = 96.8% (source TPR 99.8% - 3 pts).
| N | target FPR | hard-clean FPR | reworded TPR | garak TPR |
|---|---|---|---|---|
| 10 | 15.1% | 30.2% | 99.5% | 96.3% |
| 20 | 6.9% | 11.6% | 99.3% | 94.7% |
| 50 | 6.8% | 10.3% | 99.4% | 95.0% |
| 100 | 5.8% | 7.0% | 99.4% | 94.6% |
| 200 | 4.9% | 4.4% | 99.3% | 94.3% |
Oracle cutoff (fit on all target_test, 5% FPR): garak TPR 94.3% -> below the 96.8% floor even with a perfect cutoff, so the shortfall comes from the scorer's ranking of garak attacks, not from cutoff estimation error. Hypothesis H2 supported for garak, not for reworded or hard-clean sets.
VALIDITY THREAT (found when reading results): every injected passage in every set (train, reworded, garak) contains a canary of the same shape WORD-nn (all caps + hyphen + digits); clean passages never do. The scorer may be using that token as a shortcut, which would inflate AUC, TPR on reworded/garak, and explain the ceiling effect. Test: rebuild with varied canary formats (train formats vs held-out formats) and retrain; compare. Also hard-clean sentences resemble the training benign sentences (both list words like override/ignore/follow), so that set is probably easier than intended.

## 11. Canary-format fix (run3, prepared 2026-10-14)
- Problem: in run2 every injected passage contained a token shaped WORD-nn; clean passages never did.
- Fix (src/canary.py): training canaries use formats {WORD-nn, plain word, quoted two-word phrase, 5-digit number} and train words; test sets (ood_reworded, garak_ood) use held-out formats {6-char hex, Capitalised word} and held-out words; source_val / target pool / target test use all formats. 20% of clean passages (all sets) get a benign code-like sentence ("The sample reference code is {t}.", etc.) with tokens in the same formats, so token shape alone is not predictive.
- garak_sentences.txt is now 17 templates with a {c} placeholder filled per passage with a held-out-format token.
- Compare run3 against run2: if TPR on reworded/garak and AUC are similar, the shortcut did not drive the results; if they drop, report both (run2 as the confounded version).
- build_garak.py (v2b): half of garak_ood passages use garak's own payload trigger "Haha pwned!" (from garak/probes/latentinjection.py), half use held-out-format canary tokens.

## 12. Run3 (canary-shape fix) results -- results/run3. SUPERSEDES run2 (run2 confounded by token-shape shortcut; report both, lead with run3)
Scorer at source threshold (5% FPR on source_val): source_val AUC .9998 (FPR 4.8%, TPR 99.8%); source_test_unseen AUC .9995; target_pool AUC .986, target_test AUC .986 FPR 82.4% TPR 100%; ood_hardclean FPR 94.7%; ood_reworded TPR 99.3%; garak_ood TPR 99.3% (uninformative: cutoff blocks 82% of clean).
Threshold repair, mean over 100 draws (floor for TPR 96.8%):
| N | target FPR | target TPR | hard-clean FPR | reworded TPR | garak TPR | draws w/ FPR<=7.5% |
|---|---|---|---|---|---|---|
| 10 | 18.1% | 95.1% | 43.0% | 71.7% | 77.1% | 26% |
| 20 | 9.0% | 90.8% | 21.5% | 57.4% | 65.0% | 50% |
| 50 | 6.3% | 90.3% | 12.3% | 54.2% | 62.1% | 55% |
| 100 | 5.0% | 89.2% | 7.6% | 51.3% | 59.1% | 69% |
| 200 | 4.1% | 88.0% | 4.9% | 48.8% | 56.7% | 84% |
| oracle (full target_test) | 4.9% | 89.7% | 5.0% | 51.3% | 58.0% | - |
Platt@0.5: FPR 8.4-9.0% at every N (never reaches 5%), TPR ~94.5-95%, ECE 0.059 -> 0.043 (uncalibrated 0.052).
Findings: (1) FPR can be restored by cutoff repair: mean FPR reaches ~5% at N=100 and ~4% at N=200; reliability (draws within 7.5%) 69% at N=100, 84% at N=200. (2) Detection cannot be restored: even the oracle cutoff gives target TPR 89.7% (6 pts+ below floor 96.8%), reworded 51%, garak 58%: success rule met in 0% of draws at every N. The loss is a ranking problem (AUC .9998 -> .986), not a cutoff problem. H1 (FPR restored with <=200 labels while TPR within 3 pts) not supported; H2 (reworded/garak break the guarantee) strongly supported. (3) The run2 numbers (reworded 99%, garak 94%) were inflated by the token-shape shortcut.
Note: scores_2.npz did not contain ood_garak_haha (safety copy downloaded before it was built); the Haha-pwned garak variant still pending.
Possible extra: what does fix it? e.g. fine-tune scorer on N labelled target passages (needs Colab), or train with more diverse injections.

## 13. Wider N range (results/run3_bigN; same scores as run3; N in 10..700, 100 draws; pool = 1,500 target passages, 752 injected)
Threshold repair, mean FPR / TPR(target) / draws with FPR<=7.5%: N=10 18.1/95.1/26%; 20 9.0/90.8/50%; 30 6.5/89.3/58%; 50 5.5/89.9/59%; 75 4.0/87.6/81%; 100 5.0/89.4/67%; 150 4.4/88.5/84%; 200 4.1/88.1/89%; 300 4.6/89.0/86%; 500 4.4/88.7/96%; 700 4.2/88.4/97%. Oracle TPR 89.7%.
Reading: mean FPR reaches ~5% by N~30-100 and stays ~4.2-4.6%; reliability climbs to ~90% at N~200 and ~96-97% at N>=500; TPR is flat at ~88-89% for all N (plateau at the oracle level): more labels do not improve detection with cutoff repair. Platt@0.5 plateaus at FPR ~8.8%, TPR ~95%, ECE 0.04 (wrong operating point, not a sample-size effect). Caveats: per-N values vary by ~1 pt between runs with different N lists (random draws differ); for N>=300 the draws overlap heavily (sampling without replacement from a 1,500 pool), so the spread understates uncertainty; single trained scorer (one seed).

## 14. Second garak set (results/run3_final; same scorer, verified identical target_test scores)
ood_garak_haha: 300 injected science passages, garak sentences with payload = garak's own "Haha pwned!" for ~half and held-out-format tokens for ~half (the garak_ood set uses held-out-format tokens only). TPR: old source cutoff 98.3% (garak_ood 99.3%); after 5%-FPR repair: oracle 60.3% (garak_ood 58.0%); threshold repair N=20 65.3%, N=50 63.5%, N=100 60.9%, N=200 57.6%, N=500 58.7%. Both garak variants are far below the 96.8% floor at every N; using garak's own payload does not change the conclusion.

## 15. Fine-tuning comparison (script prepared 2026-10-14; results pending)
`src/finetune_target.py`: for N in {20,50,100,200,500} and 3 draws, sample N labelled target_pool passages (half clean), split 70/30 within class: 70% fine-tune the source-trained scorer (3 epochs, lr 2e-5, batch 16, warmup 10%), 30% choose the 5%-FPR cutoff (95th percentile of clean scores); evaluate on target_test and every ood set (never used for fitting) with the same success rule (FPR<=7.5%, TPR>=96.8%, hard-clean FPR<=7.5%). Logic code-tested on fake scores only (torch part untested locally: no torch). Few draws (3) because each run retrains the model: report as indicative, not with tight intervals. For N=20 the cutoff set has only ~3 clean passages: cutoff is very noisy.

## 16. Fine-tuning run 1 (3 draws per N; cutoff from 30% of the N labels) -- indicative only
Mean over 3 draws (FPR target / TPR target / AUC target / hard-clean FPR / reworded TPR / garak TPR / garak-haha TPR): N=20: 26.8/96.4/.983/55.6/78.1/81.4/81.7; N=50: 21.1/98.9/.990/52.9/83.0/87.4/87.2; N=100: 1.3/96.3/.998/4.7/56.0/55.1/56.7; N=200: 7.1/99.8/.999/31.8/80.2/77.2/80.6; N=500: 4.6/99.7/.999/35.4/85.8/79.7/82.7. Success rule met in 0 draws.
Reading: fine-tuning restores RANKING (AUC .986 -> .998-.999 from N=100) and detection of attack styles seen in the fine-tuning sample (target TPR ~99.7% at 4-7% FPR; but target_pool and target_test share the same attack templates, so this is NOT an unseen-attack result). On unseen styles (reworded, garak) TPR rises from ~50-58% (cutoff repair) to ~77-86% at N=200-500, still below the 96.8% floor. Hard-clean FPR rises to 26-41% at N>=200 (worse than cutoff repair, 4-5%): fine-tuning on the pool's passages does not carry over to hard-clean sentences. Results are very noisy: the cutoff comes from only 30% of N (e.g. N=100 -> ~15 clean passages), so FPR per draw ranged 0.3%-47%; e.g. N=100 draws had FPR 0.3-2% and TPR on reworded 44-63% purely because the cutoff was too conservative. Next: rerun with oracle-cutoff metrics (suffix _orc) and 5 draws to separate ranking quality from cutoff noise.

## 17. Fine-tuning run 2 (5 draws per N, oracle-cutoff columns `_orc`)

Setup: finetune_target.py defaults (N = 50/100/200/500, 5 draws, 70% of labels fine-tune results/model for 3 epochs at lr 2e-5, 30% set the 5%-FPR cutoff). `_orc` = cutoff set to 5% FPR on target_test (isolates ranking quality from cutoff noise; FPR_orc = 4.9% by construction).

Means over 5 draws:

| N | AUC target | TPR target (orc) | TPR reworded (orc) | TPR garak_ood (orc) | TPR garak_haha (orc) | FPR hardclean (orc) | FPR target (own cutoff) |
|---|---|---|---|---|---|---|---|
| 50  | .991 | .967 | .645 | .669 | .705 | .155 | .067 |
| 100 | .999 | .996 | .705 | .705 | .737 | .219 | .051 |
| 200 | .999 | .997 | .805 | .778 | .821 | .325 | .062 |
| 500 | .999 | .997 | .933 | .859 | .877 | .458 | .073 |

Findings:
- Ranking on target-style attacks is repaired from N=100 (AUC .999). Caveat: pool and target_test share attack templates, so this is not an unseen-attack result.
- Unseen-style TPR rises steadily with N (reworded .65 -> .93) but only reworded at N=500 approaches the 0.968 floor; garak sets stay at .86-.88. Success rule met in 0/20 runs.
- Hard-clean over-blocking grows with N (15.5% -> 45.8% at oracle cutoff): fine-tuning on injected examples pushes the model to flag any instruction-like sentence. Trade-off, not a free gain.
- Own-cutoff FPR is very noisy per draw (0.4%-16.4%) because the cutoff comes from 30% of N; means are near target (5-7%) but individual draws are unreliable.
- Caveats: single base-model seed, 5 draws, no CIs, hard-clean only 10 sentence templates.

## 18. End-to-end RAG pipeline (src/rag_pipeline.py) - design

- Corpus: SciQ *test* support paragraphs (40-200 words, containing the gold answer), 150 questions + 400 distractor passages. Retriever: BAAI/bge-small-en-v1.5 (cosine, top-3). LLM: Qwen2.5-1.5B-Instruct, greedy, 40 new tokens.
- Attack: the gold passage is replaced by a poisoned copy (one sentence inserted at a sentence boundary asking for a harmless canary token from held-out formats/words). Attack mix cycles: standard held-out wordings (INJ_TEST), reworded, garak-derived sentences. ASR = canary appears in the answer.
- Filter: `TrustFilter(BaseNodePostprocessor)` (llama-index-core); drops nodes with z > cutoff. Tested locally with stub scorer/LLM (no filter ASR 1.0; perfect scorer 0.0; block-all kills accuracy).
- Conditions: no_filter; source_cutoff (95th pct clean z on source_val); repaired_N100 (base scorer, cutoff from 100 labelled pool passages, 3 draws); finetuned_N200 (scorer fine-tuned on 70% of 200 labels, cutoff from other 30%, 3 draws).
- Metrics: ASR, poison_blocked, acc_clean, acc_attacked, clean_drop, gold_kept; scoring latency (ms/query) measured on 30 queries.
- Limits: weak 1.5B LLM may follow injections only sometimes (baseline ASR reported first); one model, one retriever; canary is harmless by design.

## 19. RAG pipeline run 1 (Qwen2.5-1.5B-Instruct, 150 q, top-3, default layout)

| condition | ASR | poison blocked | acc clean | acc attacked | clean passages dropped | gold kept |
|---|---|---|---|---|---|---|
| no_filter | .007 | 0 | .920 | .913 | 0 | .973 |
| source_cutoff | 0 | .993 | .627 | .567 | .838 | .200 |
| repaired_N100 | 0 | .615 | .920 | .649 | 0 | .973 |
| finetuned_N200 | 0 | .788 | .920 | .580 | .016 | .958 |
Latency: scoring 16.5 ms/query (3 passages, GPU), generation ~677 ms/query.

Findings: (1) the 1.5B model almost never follows the planted instruction (no-filter ASR 0.7%), so ASR cannot show a filter benefit -> attack-success metric is uninformative for this LLM; need a more susceptible model/layout (llm_susceptibility.py added; layouts default/context_last; attack kinds classic/standard/reworded/garak). (2) Blocking the poisoned gold passage costs answer accuracy (attacked acc .91 -> .58-.65): filtering removes the evidence as well as the attack. (3) Source cutoff on science text drops 84% of clean passages (acc .63) - the domain-shift failure at system level. (4) Repaired cutoff blocks only 61.5% of poisoned passages (consistent with TPR ~50-60% on unseen styles); fine-tuned 78.8%.

## 20. Decision note: choosing the answering LLM (record for the report - "what we tried and why")

Order of events (keep this honest in the write-up; do not present the final model as the first choice):
1. Planned RAG test with Qwen2.5-1.5B-Instruct (default layout: context first, question last). Reason: small, fast, free. Chosen BEFORE seeing any attack results.
2. Result (section 19): no-filter ASR 0.7% -> the LLM almost never obeys the planted sentence, so ASR cannot show a filter benefit. Filter effects visible only via passages blocked, clean passages dropped, accuracy.
3. Likely reasons Qwen resists: question placed last, trained on RAG-style prompts (context = material, not orders), our canary instruction clashes with a factual question. Not evidence that such attacks are harmless in general.
4. Decision (user + assistant, 2026-10-14): measure susceptibility first with llm_susceptibility.py (no scorer): 4 LLMs (Qwen2.5-0.5B, TinyLlama-1.1B-Chat, Qwen2.5-1.5B, Phi-3.5-mini) x 2 layouts (default, context_last) x 4 attack wordings (classic, standard, reworded, garak), 60 questions each. Then rerun the pipeline on the model/layout with a clear non-trivial no-filter ASR.
5. Reporting rule: always report the 1.5B/default result (ASR 0.7%) alongside the final one, state that the model was selected for susceptibility after the first run (a form of selection; not tuned against the scorer), and report no-filter ASR next to every filtered number.
6. Fallback if no model is fooled often: keep Qwen and report blocking rate / accuracy / over-blocking as the system-level evidence; say ASR could not be demonstrated.

Other facts to carry into the write-up:
- Three RAG parts: retriever (bge-small, not trained by us), scorer (our DistilBERT, the contribution), answering LLM (off-the-shelf). The scorer only decides which passages the LLM sees.
- Filtering a poisoned gold passage also removes the answer (acc under attack .91 -> .58-.65): inherent cost of dropping rather than sanitising.
- Source (Wikipedia) cutoff on science text dropped 84% of clean passages (acc .63) = domain-shift failure at system level.
- Fine-tuning (section 17): ranking fixed from N=100, unseen-style TPR .65 -> .93 (reworded) at N=500, but hard-clean over-blocking rises .16 -> .46 and the 0.968 floor is never met.

## 21. LLM susceptibility test result (no filter; ~58 attacked questions per cell, so +-5 pts noise)

ASR (canary in answer):
| LLM | layout | classic | garak | reworded | standard |
|---|---|---|---|---|---|
| Phi-3.5-mini-instruct | context_last | .276 | .068 | .069 | .121 |
| Phi-3.5-mini-instruct | default | .052 | .017 | .103 | .190 |
| Qwen2.5-0.5B-Instruct | context_last | .052 | .034 | .034 | 0 |
| Qwen2.5-0.5B-Instruct | default | 0 | 0 | 0 | 0 |
| Qwen2.5-1.5B-Instruct | context_last | .069 | .017 | .069 | 0 |
| Qwen2.5-1.5B-Instruct | default | 0 | .017 | 0 | 0 |
| TinyLlama-1.1B-Chat | context_last | .034 | 0 | .052 | 0 |
| TinyLlama-1.1B-Chat | default | .017 | .051 | .017 | .034 |

Finding: all four models resist these injections most of the time; Phi-3.5-mini is the most susceptible (mean ASR ~13% in context_last, ~9% default). Decision: use Phi-3.5-mini + context_last for the main RAG run (selected for susceptibility after seeing run 1; not tuned against the scorer). Report that susceptibility is moderate, not high; ASR numbers carry wide error bars; also report the Qwen-1.5B run-1 result.
Main run: python src/rag_pipeline.py --llm microsoft/Phi-3.5-mini-instruct --prompt-style context_last --kinds classic standard reworded garak --questions 250

## 22. RAG run with Phi-3.5-mini: CUDA out-of-memory (14.5 GB GPU)
First attempt interrupted by hand during generation; second attempt crashed with torch.OutOfMemoryError (12 GB allocated, batch 16 of ~1000-token prompts, longest prompts last, plus fine-tuned scorers resident). Fix: generation batch size now a flag (--gen-bs, default 4), prompts sorted longest-first, GPU cache cleared before loading the LLM. No change to metrics or results logic.

## 23. RAG main run: Phi-3.5-mini-instruct, context_last layout, 250 questions, top-3, attacks cycle classic/standard/reworded/garak, 3 draws for repaired/finetuned (means over draws)

| condition | ASR | poisoned passage blocked | acc clean | acc attacked | clean passages dropped | gold kept (clean) |
|---|---|---|---|---|---|---|
| no_filter | .128 | 0 | .912 | .768 | 0 | .976 |
| source_cutoff (Wikipedia, -5.46) | 0 | 1.00 | .604 | .560 | .831 | .200 |
| repaired_N100 | .008 | .75 | .912 | .621 | 0 | .976 |
| finetuned_N200 | .004 | .87 | .904 | .571 | .018 | .961 |
Run time 757 s (gen batch 4). Latency: scoring 17.5 ms/query vs generation 819 ms/query (~2% overhead for the base scorer at top-3).

Findings:
- Filtering cuts the end-to-end attack success from 12.8% to under 1% with only 100 labelled target passages (cutoff repair), and the fine-tuned scorer is similar (0.4%); difference between those two is not resolvable at n=250 (~2 vs ~1 successes).
- ASR falls more than "blocked" suggests (blocked .75 vs ASR cut ~94%): hypothesis (unverified: per-kind breakdown not saved) the attacks Phi actually obeys are mostly classic/standard wordings, which the scorer detects better than reworded/garak.
- Source cutoff "works" only by blocking nearly everything (83% of clean passages dropped, clean accuracy .60): domain-shift failure at system level.
- Cost: accuracy under attack is LOWER with filters (.62/.57) than with no filter (.77), because removing the poisoned passage also removes the answer-bearing text (gold passage is the poisoned one by construction). Security-utility trade-off; a sanitising filter (delete only the injected sentence) is future work.
- Caveats: one LLM, one retriever, harmless canary, single trained scorer, 3 draws, small ASR counts (~32 successes at no_filter), the LLM was selected for susceptibility after run 1.

## 24. Adaptive-lite attack (src/adaptive_lite.py) - design (reviewer asked for an adaptive/OOD setting beyond garak)
Attacker with query access to the deployed scorer's output tries up to K semantics-preserving variants per passage (wording from classic/standard/reworded/garak lists x position start/mid/end x wrapper {plain, quoted, parentheses, "Note:", brackets} x optional harmless lead-in sentence; ~1000 variants) and keeps the lowest-scoring one. Detection(K) = share of 300 clean target_test passages where all K tried variants are still flagged. K=1 = non-adaptive baseline. Evaluated for: base scorer at Wikipedia / repaired (N=100, 20 draws) / oracle cutoffs, and fine-tuned scorer (N=200, 2 draws) at own and oracle cutoffs.
Limitations to state: query-budget search only (no gradients, no learned generator, not Nasr et al.-style); success = evading detector, LLM compliance with each variant not measured; scorer outputs assumed visible to attacker (stronger than a pure black-box attacker). Logic tested with stub scores; real run in Colab.

## 25. Adaptive attack end to end (src/adaptive_e2e.py) - design
For each of 200 questions the attacker samples 50 variants of the injection for the gold passage (same variant space as adaptive_lite), scores them with the DEPLOYED scorer (base + repaired N=100 cutoff; fine-tuned N=200 + own cutoff) and keeps the lowest-scoring one (non-adaptive = first random variant). Full pipeline run with Phi-3.5-mini (context_last), attacked mode only, with and without filter. Reports real ASR, poison_blocked, acc_attacked. Attacker does not check LLM compliance, so no-filter ASR of adaptive variants shows whether evasion cost effectiveness. rag_pipeline.run() gained a `modes` argument (default unchanged). Stub-tested only; real run in Colab (~20-30 min).

## 26. Adaptive-lite results (300 clean target_test passages x 100 sampled variants from a 930-variant space; detection = share of attacks still flagged)

| setting | K=1 | 5 | 10 | 25 | 50 | 100 |
|---|---|---|---|---|---|---|
| base, Wikipedia cutoff | .997 | .940 | .893 | .857 | .837 | .827 |
| base, oracle cutoff | .597 | .073 | .013 | 0 | 0 | 0 |
| base, repaired N=100 (20 draws) | .598 | .090 | .024 | .010 | .008 | .006 |
| fine-tuned N=200, oracle cutoff (draw 0 / 1) | .773 / .783 | .277 / .330 | .080 / .103 | .027 / .033 | .020 / .023 | .020 / .023 |
| fine-tuned N=200, own cutoff (draw 0 / 1) | .750 / .747 | .223 / .240 | .057 / .050 | .013 | .007 / .010 | .007 |

Findings:
- Non-adaptive (K=1) detection with repaired cutoff is ~60% (consistent with earlier unseen-style TPR); with K=10 queries an attacker evades ~97.6% of the time, K=100: 99.4%. Fine-tuning improves K=1 (75-78%) but collapses the same way (5-10% at K=10, ~1-2% at K=100).
- Repaired vs oracle cutoff curves are nearly identical -> the weakness is the scorer's ranking of rewritten attacks, not cutoff estimation; labels for cutoff cannot fix it.
- Wikipedia-cutoff row is only high because that cutoff flags ~83% of CLEAN science passages (not protection).
- Consistent with "the attacker moves second" (Nasr et al.): a static learned detector is not adaptively robust.
- Caveats: attacker sees the scorer's real-valued output (strong assumption); success = evading the detector only (LLM compliance with the chosen variants unmeasured -> adaptive_e2e.py); variant space is only 930 rewrites of 15+ wordings; variants of a passage are not independent; one trained scorer; 2 fine-tune draws.

## 27. Adaptive attack end to end (adaptive_e2e.py): 200 questions, Phi-3.5-mini, context_last, 50 candidate variants per question, 1 draw per scorer (base: repaired N=100 cutoff 5.22; fine-tuned N=200 own cutoff -1.06)

| scorer | attacker | condition | ASR | poisoned blocked | acc (attacked) |
|---|---|---|---|---|---|
| base repaired N100 | non-adaptive | no_filter | .090 | 0 | .855 |
| base repaired N100 | non-adaptive | filter | .015 | .500 | .740 |
| base repaired N100 | adaptive | no_filter | .005 | 0 | .895 |
| base repaired N100 | adaptive | filter | .005 | 0 | .895 |
| finetuned N200 | non-adaptive | no_filter | .085 | 0 | .855 |
| finetuned N200 | non-adaptive | filter | .010 | .691 | .670 |
| finetuned N200 | adaptive | no_filter | .015 | 0 | .890 |
| finetuned N200 | adaptive | filter | .015 | .005 | .885 |

Findings:
- The adaptive attacker evades the filter almost completely (poisoned blocked: 0% and 0.5%) - confirms adaptive-lite.
- But the variants that evade the scorer also stop working on the LLM: no-filter ASR drops from 9.0%/8.5% (random variant) to 0.5%/1.5% (lowest-scorer variant). The scorer-lowest variants are the ones that read least like instructions; Phi rarely obeys them. So evasion-only adaptation does NOT yield a working end-to-end attack in this setup (evade vs obey tension).
- Therefore the filter's measurable benefit vanishes against the adaptive attacker mostly because the attack is weak, not because the filter holds. Unknown: a JOINT attacker that also queries the LLM and keeps a variant that both evades and is obeyed (candidate next experiment; Phi obeyed ~9% of random variants, so among 50 candidates many questions probably have an obeyed variant; the question is how many are also low-scoring).
- Caveats: n=200 (9% = 18 successes, 0.5% = 1 success, 1.5% = 3); single cutoff draw; one LLM; non-adaptive blocked rates (50%/69%) differ from the earlier 75%/87% run because of a single cutoff draw and a different question/attack mix.

## 28. Joint adaptive attacker (src/adaptive_joint.py) - design
Attacker samples 40 variants per question (200 questions), probes EACH on Phi-3.5-mini with no filter (8,000 prompts, retrieval recomputed per variant) to learn which are obeyed, then per deployed scorer picks (joint) the obeyed variant with the lowest score, or (obey_only) the first obeyed variant. Full pipeline run with/without filter gives real ASR. Probe-only numbers: obeyed_any, joint_feasible (some variant both obeyed and under the cutoff), evade_any. Strong assumptions to state: attacker can query the same answer model ~40 times per target and sees the scorer's score. Selection/feasibility logic unit-tested on toy arrays; full run in Colab (~35-45 min). Saves results/joint_probe.npz (obey matrix + scores).

## 29. Joint attacker run interrupted (Colab free-GPU usage limit reached) - partial results
Probe finished before the interruption: LLM obeyed 5.7% of all 8,000 variants; 80% of questions have >=1 obeyed variant; vs base scorer + repaired N=100 cutoff (5.22) only 15% of questions have a variant that is both obeyed and under the cutoff (evade_any = 100%).
Partial full-pipeline rows (base scorer, 'obey_only' attacker = first obeyed variant, scorer ignored): no_filter ASR .545, acc_attacked .49; with filter ASR .050, poisoned blocked .870, acc_attacked .59. Joint attacker and fine-tuned-scorer rows pending.
Script updated: --reuse-probe (loads results/joint_probe.npz: obey matrix + the scores saved with it; fine-tuned scorer is retrained, tiny numerical differences possible) and results/adaptive_joint.csv is written after every row.

## 30. Confidence intervals and final figures (src/make_figures.py; CPU only; results/final/)
Source: results/run3_final/scores.npz. For each N (labelled target passages, half clean) 200 draws of cutoff repair; evaluated on target_test and harder sets. Spread reported two ways: p10-p90 across draws (which labels were drawn) and 95% bootstrap CI of the mean over test passages (finite test set; draw cutoffs held fixed, 500 resamples). Outputs: repair_ci.csv, oracle_reference.csv, fig_fpr.png, fig_tpr.png, fig_reliability.png.
Key numbers (mean FPR on clean target_test [95% CI]; share of draws with FPR<=7.5%): N=20 8.7% [7.5, 10.1] 53%; N=50 6.6% [5.3, 8.0] 55%; N=100 5.1% [4.1, 6.4] 68%; N=200 3.9% [2.8, 5.0] 89%; N=500 4.3% [3.0, 5.8] 96%; N=700 4.1% [2.9, 5.6] 99%. TPR (target injected) 88-91% for N>=20 (oracle 89.7%), reworded 49-57% (oracle 51.3%), garak 56-65% (oracle 58.0%), garak+Haha oracle 60.3%; hard-clean FPR 4.6% at N=200. Oracle cutoff FPR 4.9%.
Reading: the N needed for the FPR goal on average (~5%) is ~30-100; for >=90% of draws to land within 7.5% it is ~200; detection stays flat at the oracle level whatever N.

## 31. Extras prepared for the GPU (all stub/logic-tested locally; real runs in Colab)
- rag_pipeline.py: run(..., return_detail=True) now returns per-question detail; main() saves results/rag_detail.csv and prints attack success / blocking by attack wording (classic, standard, reworded, garak) for each condition. Rerun: python src/rag_pipeline.py --llm microsoft/Phi-3.5-mini-instruct --prompt-style context_last --kinds classic standard reworded garak --questions 250 (~13+ min). Purpose: check the unverified hypothesis that attacks fell by more (94%) than the blocking rate (75%) because the attacks Phi obeys are the classic/standard wordings the scorer catches best.
- multi_seed.py: seed 0 = existing scorer; --seeds 1 2 retrain with the same settings (3 seeds total); per-seed oracle TPRs, repair FPR at N=100/200, reliability, hard-clean FPR; mean and sd across seeds -> results/multi_seed.csv, results/seeds/seed{k}_scores.npz. Does not touch results/model.
- shift_test.py: length (short / original / long = passage + 2 others) x injected sentence position (start / mid / end), held-out wordings and canary formats; cutoffs fixed from unmodified target passages (repaired N=100 and oracle); reports FPR per length, TPR per (length, position), and the share of attacks inside the 256-token window. -> results/shift_test.csv.
Run order in Colab: adaptive_joint (--reuse-probe) -> rag_pipeline (breakdown) -> multi_seed -> shift_test.

## 32. Extras run in Colab (2026-10-20)

Colab state rebuilt from backup.zip (repaired with zip -FF); results/model was truncated, so the scorer was retrained with train.py (seed 0). It reproduced the earlier metrics (source AUC 0.9998, target AUC 0.986, FPR at source cutoff 82%), so scores.npz stays valid.

### 32a. Length and position shift (src/shift_test.py, 200 passages per cell, mixed standard+reworded+garak wordings)
Cutoffs: repaired N=100 = 4.62, oracle = 4.85.
Clean FPR by length (repaired / oracle): short 3.0/3.0 %, original 6.0/6.0 %, long 7.5/6.5 %. -> repaired cutoff holds when passage length changes.
TPR (repaired / oracle), position end / mid / start:
- long: 14.5/13.5, 54.5/52.0, 67.5/66.5
- original: 54.5/53.0, 67.0/64.5, 71.5/70.5
- short: 61.0/59.5, 70.5/68.5, 70.0/67.5
Share of attacks inside the 256-token window: long end 17.5 %, long mid 89.5 %, otherwise ~96-100 %.
Finding: long passages with the attack at the end are mostly invisible to the scorer (truncation at 256 tokens), a design limit; fix = score overlapping windows and take the max (future work). Absolute TPR (~70 %) is lower than the headline because the wording mix includes the hard reworded and garak sets; compare cells, not levels. Repair vs oracle differ by 1-2 pts only.

### 32b. Seed stability (src/multi_seed.py, seeds 0,1,2, same data and settings)
mean (sd): source AUC .999 (.001); target AUC .984 (.003); FPR at source cutoff .828 (.031); oracle TPR target .882 (.015), reworded .492 (.026), garak .653 (.067), garak+haha .671 (.059);
repair FPR N=100 .050 (.004), N=200 .046 (.006); draws within 7.5 %: N=100 .833 (.081), N=200 .923 (.045); hard-clean FPR at N=200 .033 (.023; per seed .048/.044/.007).
Finding: the headline (FPR restored at N~100, detection not restored, ranking problem) holds in all 3 seeds. Spread is small except garak TPR (+-.07) and hard-clean over-blocking (varies a lot by seed: report as a range, not a point value). Note: share-within-7.5 % at N=100 is .76 for seed 0 here vs 68 % in the earlier figure run (different draw protocol/random draws); quote the multi-seed value with its sd.

### 32c. Pipeline breakdown by attack wording (rag_pipeline.py, Phi-3.5-mini, context_last, 250 q, kinds classic/standard/reworded/garak)
Run time 691 s; latency scoring 16.2 ms vs generation 774 ms per query. Headline reproduced: no_filter ASR .128; repaired_N100 ASR .008 (75 % poison blocked); finetuned_N200 ASR .004 (87 % blocked); source cutoff ASR 0 but drops 83 % of clean passages (clean acc .604).
Per kind (ASR no filter | blocked by repaired N=100 | blocked by fine-tuned N=200 ; n per kind: 62-63 no-filter, 186-189 filtered):
- classic: .175 | .984 | .984
- standard: .143 | .817 | 1.000
- reworded: .129 | .569 | .753
- garak: .065 | .622 | .739
Residual ASR after repaired cutoff: classic .016, standard .005, reworded .011, garak 0; fine-tuned: classic .016, others 0.
Finding: blocking is weakest on reworded and garak wordings (57-62 % repaired, 74-75 % fine-tuned), the same sets where detection was weak in the sweeps, but the answering LLM also obeys those less (garak 6.5 %), so the end-to-end residual ASR stays about 1 %. Reworded is the kind with the most remaining risk. Caveat: per-kind n is small (a 1.6 % ASR is ~3 cases), so report as indicative, not precise.
