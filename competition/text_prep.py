"""Text cleaning for the TF-IDF model only (the transformers and LLMs read the raw text).
Final TF-IDF version: make_prep(("quotes", "elong_caps", "name")); each step was kept or dropped after a paired
comparison on 15 splits (Kaggle競賽實驗紀錄 9.3). step_negation was tried and dropped (-0.0012)."""
import re

from common import normalize_quotes

# ---------------------------------------------------------------- preprocessing steps
EMOJI = r"[\U0001F300-\U0001FAFF☀-➿]"
NEGATORS = {"not", "no", "never", "nothing", "nobody", "none", "nor", "cannot"}


def step_baseline(t):           # what the submitted TF-IDF version does
    t = t.lower()
    t = re.sub(r"https?://\S+|www\.\S+", " url ", t)
    t = re.sub(r"@\w+", " user ", t)
    return re.sub(r"\s+", " ", t).strip()


def step_quotes(t):             # he’d -> he'd
    return normalize_quotes(t)


def step_elong_caps(t):         # "soooo" -> "soo"; add a marker for each SHOUTED word (before lowercasing)
    t = re.sub(r"(\w)\1{2,}", r"\1\1", t)
    caps = len(re.findall(r"\b[A-Z]{3,}\b", t))
    return t + " allcapsword" * caps


def step_name(t):               # anonymised [NAME] -> one dedicated token instead of the word "name"
    return re.sub(r"\[name\]", " personname ", t, flags=re.I)


def step_negation(t):           # "not happy at all" -> "not neg_happy neg_at neg_all" (until punctuation, max 3 words)
    out, left = [], 0
    for tok in re.findall(r"[\w']+|[^\w\s]", t):
        if re.fullmatch(r"[^\w\s]", tok):
            left = 0; out.append(tok); continue
        if left > 0:
            out.append("neg_" + tok); left -= 1
        else:
            out.append(tok)
        if tok in NEGATORS or tok.endswith("n't"):
            left = 3
    return " ".join(out)


# word tokenizer that keeps punctuation runs ("!!!", "?!"), emoji and 1-letter words, which the
# default token_pattern (\b\w\w+\b) silently drops
KEEP_TOKENS = r"(?u)[\w']+|[!?]+|\.{2,}|" + EMOJI


def make_prep(steps):
    def prep(t):
        if "quotes" in steps: t = step_quotes(t)
        if "elong_caps" in steps: t = step_elong_caps(t)
        if "name" in steps: t = step_name(t)
        t = step_baseline(t)
        if "negation" in steps: t = step_negation(t)
        return t
    return prep
