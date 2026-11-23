"""TF-IDF + LR out-of-fold probabilities on the SAME 5 folds (seed 42) as the transformer CV runs,
saved in the same layout (cv/<run>/oof.npy [1,n,K], test.npy [5,1,1440,K]) so blend.py can mix them."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, folds, load
from text_prep import make_prep

train, test, classes = load()
prep = make_prep(("quotes", "elong_caps", "name"))
X, Xte, y, K = train["text"].map(prep).values, test["text"].map(prep).values, train["y"].values, len(classes)
oof = np.zeros((1, len(train), K), dtype=np.float32)
test_p = np.zeros((5, 1, len(test), K), dtype=np.float32)
for f, (tr, va) in enumerate(folds(train)):
    vw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True).fit(X[tr])
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6), min_df=2, sublinear_tf=True).fit(X[tr])
    feats = lambda t: hstack([vw.transform(t), vc.transform(t)]).tocsr()
    clf = LogisticRegression(C=2, class_weight="balanced", max_iter=3000).fit(feats(X[tr]), y[tr])
    oof[0, va] = clf.predict_proba(feats(X[va]))
    test_p[f, 0] = clf.predict_proba(feats(Xte))
out = DATA_DIR / "cv" / "TFIDF_lr"
out.mkdir(parents=True, exist_ok=True)
np.save(out / "oof.npy", oof); np.save(out / "test.npy", test_p)
score = float(f1_score(y, oof[0].argmax(1), average="macro"))
json.dump({"run": "TFIDF_lr", "model": "tfidf word(1,2)+char_wb(3,6), LR C=2 balanced", "epochs": 1,
           "classes": classes, "oof_f1_per_epoch": [score]}, open(out / "summary.json", "w"), indent=1)
print(f"TFIDF_lr OOF macroF1 {score:.4f}")
