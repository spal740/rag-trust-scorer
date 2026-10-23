"""Score extra data sets with the ALREADY-TRAINED model and merge them into results/scores.npz (no retraining).
Usage: python score_new.py data/garak_ood.jsonl [more.jsonl ...]   (model read from results/model)
Adds keys extra_<name>_z / extra_<name>_y to scores.npz, where <name> is the file name without .jsonl.
"""
import os, sys
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from train import load, logits

out = "results"
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(f"{out}/model")
model = AutoModelForSequenceClassification.from_pretrained(f"{out}/model").to(dev)
d = dict(np.load(f"{out}/scores.npz"))
for path in sys.argv[1:]:
    name = os.path.basename(path)[:-6]
    x, y = load(path)
    d[f"extra_{name}_z"] = logits(model, tok, x, dev)
    d[f"extra_{name}_y"] = y
    print("scored", name, len(x))
np.savez(f"{out}/scores.npz", **d)
