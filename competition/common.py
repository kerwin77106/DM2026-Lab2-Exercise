"""Shared pieces for the competition experiments: data loading, CV folds, duplicate handling."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kaggle"  # git-ignored: competition data and run outputs

QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def normalize_quotes(text: str) -> str:
    """Curly quotes -> straight quotes, so he’d and he'd become the same token."""
    return text.translate(QUOTES)


def load():
    posts = pd.DataFrame(json.loads((DATA_DIR / "final_posts.json").read_text(encoding="utf-8")))
    df = posts.merge(pd.read_csv(DATA_DIR / "data_identification.csv"), on="post_id")
    train = df[df["set"] == "train"].merge(pd.read_csv(DATA_DIR / "emotion.csv"), on="post_id").reset_index(drop=True)
    test = df[df["set"] == "test"].reset_index(drop=True)
    classes = sorted(train["label"].unique())
    train["y"] = train["label"].map({c: i for i, c in enumerate(classes)})
    return train, test, classes


def group_key(text: str) -> str:
    """Rows whose text is the same after light normalisation share a group, so a
    duplicated post never sits in both the training and the validation fold."""
    return re.sub(r"\s+", " ", normalize_quotes(text)).strip().lower()


def folds(train, n_splits=5, seed=42):
    """Stratified (keeps class ratios) and grouped (duplicates stay together) K-fold."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    groups = train["text"].map(group_key)
    return list(sgkf.split(train, train["y"], groups))


DEDUP_STRATEGIES = ("keep_all", "drop_conflict", "one_per_text", "unique_pairs")


def dedup(train_part: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Applied to the TRAINING part of a fold only; validation always keeps the original rows.
    keep_all      : use every row as given
    drop_conflict : remove every row of a text that carries more than one label
    one_per_text  : one row per text; conflicting texts take the majority label (ties: first seen)
    unique_pairs  : drop exact (text, label) repeats; a conflicting text keeps each of its labels once
    """
    key = train_part["text"].map(group_key)
    if strategy == "keep_all":
        return train_part
    if strategy == "drop_conflict":
        n_labels = train_part.groupby(key)["label"].transform("nunique")
        return train_part[n_labels == 1]
    if strategy == "one_per_text":
        tmp = train_part.assign(_k=key)
        majority = tmp.groupby("_k")["label"].agg(lambda s: s.value_counts().index[0])
        first = tmp.drop_duplicates("_k").copy()
        first["label"] = first["_k"].map(majority)
        first["y"] = first["label"].map(dict(zip(train_part["label"], train_part["y"])))
        return first.drop(columns="_k")
    if strategy == "unique_pairs":
        return train_part.assign(_k=key).drop_duplicates(["_k", "label"]).drop(columns="_k")
    raise ValueError(strategy)
