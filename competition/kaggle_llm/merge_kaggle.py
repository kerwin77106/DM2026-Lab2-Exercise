"""Turn the fold<f>.npz files downloaded from a Kaggle kernel (data/kaggle/cv_kaggle/<run>/) into the usual
data/kaggle/cv/<run>/ layout (oof.npy [E,n,K], test.npy [5,E,1440,K], summary.json) so blend.py can use the run.
Rows are matched by post_id, so the result is in the same order as common.load().

usage: python merge_kaggle.py <run>
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DATA_DIR, load

run = sys.argv[1]
src = DATA_DIR / "cv_kaggle" / run
train, test, classes = load()
y = train["y"].values
tr_pos = {p: i for i, p in enumerate(train["post_id"])}
te_pos = {p: i for i, p in enumerate(test["post_id"])}
info = json.load(open(src / "run.json"))
E, K = info["epochs"], len(classes)
oof = np.zeros((E, len(train), K), dtype=np.float32)
test_p = np.zeros((5, E, len(test), K), dtype=np.float32)
seen = np.zeros(len(train), dtype=int)
log = []
for f in range(5):
    z = np.load(src / f"fold{f}.npz")
    vi = np.array([tr_pos[p] for p in z["val_ids"]]); ti = np.array([te_pos[p] for p in z["test_ids"]])
    assert len(ti) == len(test) and len(set(ti)) == len(test)
    oof[:, vi] = z["val_p"]; test_p[f][:, ti] = z["test_p"]; seen[vi] += 1
    for e in range(E):
        local = float(f1_score(y[vi], z["val_p"][e].argmax(1), average="macro"))
        assert abs(local - float(z["f1"][e])) < 1e-6, "labels on Kaggle and locally disagree"
        log.append({"fold": f, "epoch": e + 1, "f1": local})
assert (seen == 1).all(), "every training row must be in exactly one validation fold"
out = DATA_DIR / "cv" / run
out.mkdir(parents=True, exist_ok=True)
np.save(out / "oof.npy", oof); np.save(out / "test.npy", test_p)
summary = {**info, "classes": classes, "source": "kaggle kernel", "fold_log": log,
           "oof_f1_per_epoch": [float(f1_score(y, oof[e].argmax(1), average="macro")) for e in range(E)]}
json.dump(summary, open(out / "summary.json", "w"), indent=1)
for e in range(E):
    fs = [r["f1"] for r in log if r["epoch"] == e + 1]
    print(f"SUMMARY {run} ep{e + 1}: fold mean {np.mean(fs):.4f} sd {np.std(fs):.4f} | OOF {summary['oof_f1_per_epoch'][e]:.4f}")
