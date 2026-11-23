"""Suspected mislabels (confident learning, simplified): training rows where the final blend's OUT-OF-FOLD
probability of the given label is very low while the blend is confident about another label.
Writes ../cv_noise/<name>.json = {post_id: blend's label index}. Caveat: the OOF of a row in fold g came from
models trained on the other folds, so the flags depend a little on every fold's labels; validation rows are
never altered by the training script, so this only changes which TRAINING rows are dropped/re-weighted."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, load
from blend import load_runs

train, _, classes = load()
y = train["y"].values
R = load_runs()
TW = [n for n in R if ("twroberta" in n or "bertweet" in n or "emotion" in n) and not n.startswith("N")]
B = (np.mean([R[n][0] for n in TW], 0) + R["TFIDF_lr"][0]) / 2
p_given, p_max, arg = B[np.arange(len(y)), y], B.max(1), B.argmax(1)
out = DATA_DIR / "cv_noise"; out.mkdir(exist_ok=True)
for name, (a, b) in {"S1_p10_c60": (0.10, 0.60), "S2_p20_c50": (0.20, 0.50), "S3_p05_c70": (0.05, 0.70)}.items():
    m = (p_given < a) & (p_max > b)
    json.dump({train.post_id[i]: int(arg[i]) for i in np.where(m)[0]}, open(out / f"{name}.json", "w"))
    print(name, int(m.sum()), "rows; flagged label counts", dict(zip(classes, np.bincount(y[m], minlength=len(classes)))))
