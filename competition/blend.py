"""Blend the CV runs in ../cv/ using their out-of-fold (OOF) probabilities.

For each run the probabilities of its good epochs are averaged (epochs whose OOF F1 is within 0.005 of
that run's best). Then:
  1. every single run is scored,
  2. greedy forward selection (with replacement) picks the mix with the best OOF macro F1,
  3. the optimism of that selection is measured honestly: the selection is redone on 4 folds and scored
     on the 5th (nested), so the printed "nested" number is what to trust,
  4. optional per-class bias tuning (multiply probabilities by a weight per class), also nested.
Usage: python blend.py [out_name.csv] [run names to restrict to ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, folds, load

train, test, classes = load()
y, K = train["y"].values, len(classes)
FOLDS = folds(train)
CV = DATA_DIR / "cv"
f1 = lambda p, idx=slice(None): f1_score(y[idx], p[idx].argmax(1), average="macro")


def load_runs(only=None):
    runs = {}
    for d in sorted(CV.iterdir()):
        if not (d / "summary.json").exists() or (only and d.name not in only):
            continue
        oof, te = np.load(d / "oof.npy"), np.load(d / "test.npy")
        per_ep = [f1(oof[e]) for e in range(oof.shape[0])]
        good = [e for e, s in enumerate(per_ep) if s >= max(per_ep) - 0.005]
        runs[d.name] = (oof[good].mean(0), te[:, good].mean((0, 1)), [e + 1 for e in good])
    return runs


def greedy(runs, idx, rounds=12):
    names, chosen, cur = list(runs), [], None
    best = -1
    for _ in range(rounds):
        cand = []
        for n in names:
            p = runs[n][0] if cur is None else (cur * len(chosen) + runs[n][0]) / (len(chosen) + 1)
            cand.append((f1(p, idx), n, p))
        s, n, p = max(cand, key=lambda c: c[0])
        if s <= best + 1e-5:
            break
        best, cur = s, p
        chosen.append(n)
    return chosen, best


def mix(runs, chosen, which):
    return np.mean([runs[n][which] for n in chosen], axis=0)


def tune_bias(p, idx, iters=3):
    w = np.ones(K)
    for _ in range(iters):
        for k in range(K):
            best = (f1(p * w, idx), w[k])
            for v in np.arange(0.6, 1.65, 0.05):
                w2 = w.copy(); w2[k] = v
                s = f1(p * w2, idx)
                if s > best[0] + 1e-6:
                    best = (s, v)
            w[k] = best[1]
    return w


if __name__ == "__main__":
    out_name = sys.argv[1] if len(sys.argv) > 1 else None
    runs = load_runs(set(sys.argv[2:]) or None)
    print("== single runs (OOF macro F1, epochs averaged)")
    for n, (o, _, eps) in sorted(runs.items(), key=lambda kv: -f1(kv[1][0])):
        print(f"  {n:32s} {f1(o):.4f}  epochs {eps}")

    chosen, s = greedy(runs, slice(None))
    print(f"\n== greedy blend on all OOF: {s:.4f}  <- optimistic (selected on the same rows)")
    print("   members:", {n: chosen.count(n) for n in dict.fromkeys(chosen)})

    nested, nested_b = np.zeros((len(y), K)), np.zeros((len(y), K))
    for tr, va in FOLDS:
        c, _ = greedy(runs, tr)
        p = mix(runs, c, 0)
        nested[va] = p[va]
        nested_b[va] = (p * tune_bias(p, tr))[va]
    print(f"== nested greedy blend (honest): {f1(nested):.4f}")
    print(f"== nested greedy + class bias  : {f1(nested_b):.4f}")

    if out_name:
        p_oof, p_te = mix(runs, chosen, 0), mix(runs, chosen, 1)
        if "bias" in out_name:
            w = tune_bias(p_oof, slice(None)); p_te = p_te * w
            print("   class weights:", dict(zip(classes, np.round(w, 2))))
        sub = pd.DataFrame({"id": test["post_id"], "label": np.array(classes)[p_te.argmax(1)]})
        assert len(sub) == 1440 and sub["id"].is_unique
        sub.to_csv(DATA_DIR / out_name, index=False)
        json.dump({"members": chosen, "oof_f1": s}, open(DATA_DIR / (out_name + ".json"), "w"), indent=1)
        print("   wrote", out_name, "pred dist", np.bincount(p_te.argmax(1), minlength=K))
