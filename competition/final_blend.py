"""Final blend that produced the scored submission (Private 0.6643): submission_plusN5_K3_K4_w50.csv.

    base  = ( mean of the 8 fine-tuned transformers  +  TF-IDF ) / 2          OOF 0.6372
    llm   =   mean of the 2 QLoRA LLMs (Qwen2.5-14B K3, Phi-4 K4)             OOF 0.6422
    final =   0.5 * base + 0.5 * llm                                          OOF 0.6490

Every run's probabilities come from data/kaggle/cv/<run>/ (written by tfidf_oof.py, cv_finetune2.py and
kaggle_llm/merge_kaggle.py); blend.load_runs averages each run's good epochs. The weights are fixed in advance
(simple averaging, halves) instead of being tuned on the OOF rows, which would be optimistic.

usage: python final_blend.py            -> prints the OOF scores and writes data/kaggle/submission_final_K3_K4.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blend import load_runs
from common import DATA_DIR, load

TRANSFORMERS = [
    "G3b_twroberta_large",       # cardiffnlp/twitter-roberta-large-2022-154m, seed 42
    "G5_twroberta_large_s7",     # same, seed 7
    "G6_twroberta_large_s123",   # same, seed 123
    "G8_twroberta_ls01",         # same, label smoothing 0.1
    "N5_weight_S2",              # same, 327 suspected mislabels weighted x0.3 in the loss
    "G4_bertweet_large",         # vinai/bertweet-large
    "G7_twroberta_emotion",      # cardiffnlp/twitter-roberta-large-emotion-latest, seed 42
    "G9_twroberta_emotion_s7",   # same, seed 7
]
TFIDF = "TFIDF_lr"
LLMS = ["K3_qwen14b_prompt", "K4_phi4_prompt"]  # trained on Kaggle T4 x2 (kaggle_llm/)
LLM_WEIGHT = 0.5


def blend(runs, which):
    """which = 0: out-of-fold probabilities (3360 x 6), 1: test probabilities (1440 x 6)."""
    base = (np.mean([runs[n][which] for n in TRANSFORMERS], 0) + runs[TFIDF][which]) / 2
    llm = np.mean([runs[n][which] for n in LLMS], 0)
    return base, llm, (1 - LLM_WEIGHT) * base + LLM_WEIGHT * llm


if __name__ == "__main__":
    train, test, classes = load()
    y = train["y"].values
    runs = load_runs(set(TRANSFORMERS + [TFIDF] + LLMS))
    f1 = lambda p: f1_score(y, p.argmax(1), average="macro")
    base, llm, final = blend(runs, 0)
    print(f"OOF macro F1  base {f1(base):.4f}  llm {f1(llm):.4f}  final {f1(final):.4f}")
    pred = blend(runs, 1)[2].argmax(1)
    sub = pd.DataFrame({"id": test["post_id"], "label": np.array(classes)[pred]})
    assert len(sub) == 1440 and sub["id"].is_unique
    out = DATA_DIR / "submission_final_K3_K4.csv"
    sub.to_csv(out, index=False)
    print("wrote", out)
