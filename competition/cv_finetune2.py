"""5-fold CV fine-tuning, v2 (same folds and outputs as cv_finetune.py, plus memory/optimiser options).

Usage: python cv_finetune2.py <run_name> <model> <epochs> <lr> <batch>
Env:   ACCUM=<int>      gradient accumulation steps (effective batch = batch * ACCUM)
       GRAD_CKPT=1      gradient checkpointing (less GPU memory, ~30% slower)
       LLRD=<float>     layer-wise lr decay, e.g. 0.9 (top layer gets lr, each lower layer x LLRD)
       MAX_LEN=<int>    default 128
       WARMUP=<float>   warmup ratio, default 0.1
       LS=<float>       label smoothing, e.g. 0.1
       OPT=adamw8bit    bitsandbytes 8-bit AdamW (for models that do not fit with fp32 AdamW states)
       CLASS_WEIGHT=1, PREPROC=quotes, DEDUP=<strategy>, SEED=<int>  as in cv_finetune.py
       NOISE_FILE=<json>  suspected-mislabel list {post_id: model's label index} from make_noise_flags.py;
       NOISE_MODE=drop|weight|relabel  what to do with those rows in the TRAINING part of each fold
       NOISE_W=<float>  loss weight of the suspected rows for NOISE_MODE=weight (default 0.3)
       (validation rows are never changed, so scores stay comparable with every other run)
       RDROP=<float>    R-Drop: two forward passes with different dropout, loss += RDROP * symmetric KL (about 2x time)
       AWP=<float>      adversarial weight perturbation lr (e.g. 1.0); AWP_EPS=<float> max relative change per weight
                        (default 1e-3); AWP_START=<int> first epoch (1-based) that uses it (default 2)
       PSEUDO_FILE=<npy> per-fold test-set probabilities (5, n_test, K) from make_pseudo.py; test rows whose top
                        probability >= PSEUDO_T (default 0.7) are added to fold f's training part with the
                        predicted label and loss weight PSEUDO_W (default 1.0). Fold f only uses fold-f models.
       LORA_R=<int>     LoRA fine-tuning (peft) of all linear layers with rank LORA_R (for decoder LLMs such as
                        Qwen2.5-1.5B); LORA_ALPHA (default 2*r), LORA_DROPOUT (default 0.05). The base weights stay
                        frozen, only the adapters and the classification head train (kept in fp32).
       MODEL_DTYPE=bf16 load the base weights in bfloat16 instead of float32 (needed to fit a 1.5B model in 8 GB)
       TRAIN_FRAC=<f>   keep only this random share of each fold's training part (learning-curve check)
       FULL=1           train on all training rows, no validation; test probabilities go to ../cv_full/<run>/
       LOAD_4BIT=1      QLoRA: load the base weights in 4-bit nf4 (bitsandbytes) so a 7B LLM fits in 8 GB; use with LORA_R
       AUG_FILE=<json>  round-trip translated copies {post_id: [text, ...]} from make_backtrans.py; the first AUG_N
                        (default 1) copies of each row in fold f's TRAINING part are added with the row's label and
                        loss weight AUG_W (default 1.0). Validation rows never get copies.
"""
import json, os, re, sys, time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, dedup, folds, load, normalize_quotes

run_name, model_name = sys.argv[1], sys.argv[2]
epochs, lr, batch = int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
CLASS_WEIGHT = os.environ.get("CLASS_WEIGHT") == "1"
PREPROC = os.environ.get("PREPROC", "none")
DEDUP = os.environ.get("DEDUP", "keep_all")
SEED = int(os.environ.get("SEED", "42"))
ACCUM = int(os.environ.get("ACCUM", "1"))
GRAD_CKPT = os.environ.get("GRAD_CKPT") == "1"
LLRD = float(os.environ.get("LLRD", "1.0"))
MAX_LEN = int(os.environ.get("MAX_LEN", "128"))
WARMUP = float(os.environ.get("WARMUP", "0.1"))
OPT = os.environ.get("OPT", "adamw")  # adamw | adamw8bit
LS = float(os.environ.get("LS", "0"))  # label smoothing (soft targets; helps with noisy labels)
dev = os.environ.get("DEV", "cuda")
SMOKE = os.environ.get("SMOKE") == "1"
N_FOLDS_RUN = int(os.environ.get("FOLDS", "5"))  # run only the first N folds (quick diagnostics)
NOISE_FILE = os.environ.get("NOISE_FILE", "")
NOISE_MODE = os.environ.get("NOISE_MODE", "drop")
NOISE_W = float(os.environ.get("NOISE_W", "0.3"))
NOISE = json.load(open(NOISE_FILE)) if NOISE_FILE else {}
RDROP = float(os.environ.get("RDROP", "0"))
AWP = float(os.environ.get("AWP", "0"))
AWP_EPS = float(os.environ.get("AWP_EPS", "1e-3"))
AWP_START = int(os.environ.get("AWP_START", "2"))
PSEUDO_FILE = os.environ.get("PSEUDO_FILE", "")
PSEUDO_T = float(os.environ.get("PSEUDO_T", "0.7"))
PSEUDO_W = float(os.environ.get("PSEUDO_W", "1.0"))
PSEUDO = np.load(PSEUDO_FILE) if PSEUDO_FILE else None
LORA_R = int(os.environ.get("LORA_R", "0"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", str(2 * LORA_R)))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))
MODEL_DTYPE = torch.bfloat16 if os.environ.get("MODEL_DTYPE") == "bf16" else torch.float32
TRAIN_FRAC = float(os.environ.get("TRAIN_FRAC", "1.0"))  # learning-curve check: keep this share of each fold's training part
FULL = os.environ.get("FULL") == "1"  # train once on ALL training rows (no validation); writes test probabilities to ../cv_full/
LOAD_4BIT = os.environ.get("LOAD_4BIT", "") == "1"  # QLoRA: nf4 base weights (needs LORA_R), for 7B-class models in 8 GB
AUG_FILE = os.environ.get("AUG_FILE", "")
AUG_N = int(os.environ.get("AUG_N", "1"))
AUG_W = float(os.environ.get("AUG_W", "1.0"))
AUG = json.load(open(AUG_FILE, encoding="utf-8")) if AUG_FILE else {}

train, test, classes = load()
if SMOKE:
    test = test.head(16)
prep = normalize_quotes if PREPROC == "quotes" else (lambda t: t)
train["x"] = train["text"].map(prep)
test_x = test["text"].map(prep).tolist()
K = len(classes)

out = DATA_DIR / ("cv_full" if FULL else "cv") / run_name
out.mkdir(parents=True, exist_ok=True)
oof = np.zeros((epochs, len(train), K), dtype=np.float32)
test_p = np.zeros((5, epochs, len(test), K), dtype=np.float32)
log = []
t0 = time.time()


def param_groups(model):
    """Layer-wise lr decay: encoder layer i of L gets lr * LLRD**(L-i), embeddings the lowest, head the full lr."""
    if LLRD == 1.0:
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]
    names = [n for n, _ in model.named_parameters()]
    n_layers = 1 + max(int(m.group(1)) for n in names if (m := re.search(r"\.layer\.(\d+)\.", n)))
    groups = {}
    for n, p in model.named_parameters():
        m = re.search(r"\.layer\.(\d+)\.", n)
        if m:
            depth = n_layers - int(m.group(1))
        elif "embeddings" in n:
            depth = n_layers + 1
        else:
            depth = 0  # classifier head, pooler, rel-pos tables at the top
        groups.setdefault(depth, []).append(p)
    return [{"params": ps, "lr": lr * LLRD ** d} for d, ps in groups.items()]


class AdvWeightPerturb:
    """AWP: push every 'weight' tensor a small step along its gradient (each element stays within
    AWP_EPS * |w| of its value), add the loss at that worse point, then put the weights back."""

    def __init__(self, model):
        self.model, self.backup = model, {}

    def attack(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None and "weight" in n:
                self.backup[n] = p.data.clone()
                g, w = torch.norm(p.grad), torch.norm(p.data)
                if g != 0 and not torch.isnan(g):
                    p.data.add_(AWP * p.grad / (g + 1e-6) * (w + 1e-6))
                    lim = AWP_EPS * self.backup[n].abs()
                    p.data = torch.min(torch.max(p.data, self.backup[n] - lim), self.backup[n] + lim)

    def restore(self):
        for n, p in self.model.named_parameters():
            if n in self.backup:
                p.data = self.backup[n]
        self.backup = {}


COLLAPSE_LOSS = 1.65  # ln(6)=1.79 is random guessing; slow-but-healthy runs can still be ~1.3-1.5 after epoch 2


def train_fold(f, tr_x, tr_y, va_x, va_y, seed, tr_w=None):
    """Train one fold; returns per-epoch (val proba, test proba, f1, mean train loss), or None if the
    run collapsed (mean TRAINING loss still > COLLAPSE_LOSS after epoch 2 -- decided without the val fold)."""
    torch.manual_seed(seed); np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=K, dtype=MODEL_DTYPE,  # fp32 by default: some checkpoints are stored in fp16
        ignore_mismatched_sizes=True,  # checkpoints that already have a classifier head with another label count
        **(dict(quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16,
                                                       bnb_4bit_use_double_quant=True),
                device_map={"": 0}) if LOAD_4BIT else {}),
    )
    if model.config.pad_token_id is None:  # decoder LLMs: the head reads the last non-padding token
        model.config.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if GRAD_CKPT:
        model.gradient_checkpointing_enable()
    if LORA_R:
        from peft import LoraConfig, TaskType, get_peft_model
        if GRAD_CKPT:
            model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, r=LORA_R, lora_alpha=LORA_ALPHA,
                                                 lora_dropout=LORA_DROPOUT, target_modules="all-linear"))
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()  # adapters and head in fp32, frozen base stays in MODEL_DTYPE
    if not LOAD_4BIT:  # a 4-bit model is already placed on the GPU by device_map
        model = model.to(dev)
    weight = None
    if CLASS_WEIGHT:
        counts = np.bincount(tr_y, minlength=K)
        weight = torch.tensor(len(tr_y) / (K * counts), dtype=torch.float32, device=dev)

    def batches(texts, labels=None, shuffle=False, bs=batch):
        order = np.random.permutation(len(texts)) if shuffle else np.arange(len(texts))
        for s in range(0, len(texts), bs):
            b = order[s:s + bs]
            enc = tok([texts[i] for i in b], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
            yield enc, (torch.tensor(labels[b], device=dev) if labels is not None else None), b

    @torch.no_grad()
    def proba(texts):
        model.eval(); res = []
        for enc, _, _ in batches(texts, bs=32):
            with torch.autocast(dev, dtype=torch.bfloat16):
                res.append(torch.softmax(model(**enc).logits.float(), -1).cpu().numpy())
        return np.concatenate(res)

    steps = epochs * int(np.ceil(len(tr_x) / (batch * ACCUM)))
    if OPT == "adamw8bit":  # 8-bit optimiser states: ~1/4 of the AdamW memory, lets *-large models fit in 8 GB
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(param_groups(model), lr=lr, weight_decay=0.01)
    else:
        opt = torch.optim.AdamW(param_groups(model), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    res = []
    awp = AdvWeightPerturb(model) if AWP > 0 else None
    try:
        for ep in range(epochs):
            model.train(); losses = []
            for i, (enc, lab, b) in enumerate(batches(tr_x, tr_y, shuffle=True)):
                def fwd():
                    with torch.autocast(dev, dtype=torch.bfloat16):
                        return model(**enc).logits.float()

                def ce(logits):
                    if tr_w is None:
                        return torch.nn.functional.cross_entropy(logits, lab, weight=weight, label_smoothing=LS)
                    w = torch.tensor(tr_w[b], dtype=torch.float32, device=dev)  # per-row weights (suspected mislabels count less)
                    per = torch.nn.functional.cross_entropy(logits, lab, weight=weight, label_smoothing=LS, reduction="none")
                    return (per * w).sum() / w.sum()

                logits = fwd()
                loss = ce(logits)
                if RDROP > 0:  # second pass sees a different dropout mask; penalise disagreement between the two
                    logits2 = fwd()
                    lp1, lp2 = torch.log_softmax(logits, -1), torch.log_softmax(logits2, -1)
                    kl = (torch.nn.functional.kl_div(lp1, lp2, log_target=True, reduction="batchmean")
                          + torch.nn.functional.kl_div(lp2, lp1, log_target=True, reduction="batchmean")) / 2
                    loss = (loss + ce(logits2)) / 2 + RDROP * kl
                (loss / ACCUM).backward(); losses.append(loss.item())
                if awp is not None and ep + 1 >= AWP_START:
                    awp.attack()
                    (ce(fwd()) / ACCUM).backward()
                    awp.restore()
                if (i + 1) % ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); sched.step(); opt.zero_grad()
            opt.zero_grad()
            pv = proba(va_x)
            f1 = f1_score(va_y, pv.argmax(1), average="macro")
            res.append((pv, proba(test_x), f1, float(np.mean(losses))))
            print(f"{run_name} fold{f} seed{seed} ep{ep+1} macroF1={f1:.4f} train_loss={np.mean(losses):.3f} "
                  f"n_train={len(tr_x)} ({time.time()-t0:.0f}s)", flush=True)
            if ep == 1 and np.mean(losses) > COLLAPSE_LOSS:
                print(f"{run_name} fold{f} seed{seed} COLLAPSED (train loss {np.mean(losses):.3f}), retrying", flush=True)
                return None
        return res
    finally:
        del model, opt; torch.cuda.empty_cache()


retries = []
# FULL: one "fold" whose training part is every row; the 64-row "validation" is part of the training data, so its
# F1 is meaningless (printed only as a progress sign) and no OOF is written.
SPLITS = [(np.arange(len(train)), np.arange(64))] if FULL else folds(train)[:N_FOLDS_RUN]
for f, (tr_idx, va_idx) in enumerate(SPLITS):
    if SMOKE:
        va_idx = va_idx[:32]
    tr = dedup(train.iloc[tr_idx], DEDUP)
    tr_w = None
    if NOISE:
        flagged = tr["post_id"].isin(NOISE).values
        if NOISE_MODE == "drop":
            tr = tr[~flagged]
        elif NOISE_MODE == "relabel":
            tr = tr.assign(y=np.where(flagged, tr["post_id"].map(NOISE).fillna(-1).astype(int), tr["y"]))
        elif NOISE_MODE == "weight":
            tr_w = np.where(flagged, NOISE_W, 1.0)
        print(f"fold{f} noise {NOISE_MODE}: {int(flagged.sum())} flagged training rows", flush=True)
    tr_x, tr_y = tr["x"].tolist(), tr["y"].values
    if AUG:  # round-trip translated copies of this fold's training rows only (the validation fold stays untouched)
        ax, ay = [], []
        for pid, lab in zip(tr["post_id"], tr_y):
            for t in AUG.get(pid, [])[:AUG_N]:
                ax.append(prep(t)); ay.append(lab)
        n_real = len(tr_x)
        tr_x = tr_x + ax
        tr_y = np.concatenate([tr_y, np.array(ay, dtype=tr_y.dtype)])
        if AUG_W != 1.0 or tr_w is not None:
            tr_w = np.concatenate([tr_w if tr_w is not None else np.ones(n_real), np.full(len(ax), AUG_W)])
        print(f"fold{f} aug: +{len(ax)} round-trip rows (weight {AUG_W})", flush=True)
    if TRAIN_FRAC < 1.0:
        sub = np.sort(np.random.default_rng(SEED + f).choice(len(tr_x), int(round(TRAIN_FRAC * len(tr_x))), replace=False))
        tr_x, tr_y = [tr_x[i] for i in sub], tr_y[sub]
        tr_w = tr_w[sub] if tr_w is not None else None
        print(f"fold{f} train_frac {TRAIN_FRAC}: {len(tr_x)} training rows", flush=True)
    if PSEUDO is not None:  # confident test predictions of the fold-f models become extra training rows
        keep = np.where(PSEUDO[f].max(1) >= PSEUDO_T)[0]
        keep = keep[keep < len(test_x)]  # SMOKE keeps only the first 16 test rows
        n_real = len(tr_x)
        tr_x = tr_x + [test_x[i] for i in keep]
        tr_y = np.concatenate([tr_y, PSEUDO[f][keep].argmax(1)])
        if PSEUDO_W != 1.0 or tr_w is not None:
            tr_w = np.concatenate([tr_w if tr_w is not None else np.ones(n_real), np.full(len(keep), PSEUDO_W)])
        print(f"fold{f} pseudo: +{len(keep)} test rows (threshold {PSEUDO_T}, weight {PSEUDO_W})", flush=True)
    va_x, va_y = train["x"].values[va_idx].tolist(), train["y"].values[va_idx]
    if SMOKE:
        tr_x, tr_y = tr_x[:48], tr_y[:48]
    for attempt in range(3):
        res = train_fold(f, tr_x, tr_y, va_x, va_y, SEED + f + 1000 * attempt, tr_w)
        if res is not None:
            break
        retries.append(f)
    if res is None:
        raise SystemExit(f"fold {f} collapsed 3 times")
    for ep, (pv, pt, f1, tl) in enumerate(res):
        oof[ep, va_idx] = pv; test_p[f, ep] = pt
        log.append({"fold": f, "epoch": ep + 1, "f1": f1, "train_loss": tl, "n_train": len(tr_x), "attempt": attempt})

if FULL:
    np.save(out / "test.npy", test_p[:1])  # (1, epochs, n_test, K)
    json.dump({"run": run_name, "model": model_name, "epochs": epochs, "lr": lr, "batch": batch, "seed": SEED, "full": True,
               "lora_r": LORA_R, "classes": classes, "minutes": round((time.time() - t0) / 60, 1)},
              open(out / "summary.json", "w"), indent=1)
    raise SystemExit(0)
np.save(out / "oof.npy", oof); np.save(out / "test.npy", test_p)
y = train["y"].values
summary = {"run": run_name, "model": model_name, "epochs": epochs, "lr": lr, "batch": batch, "accum": ACCUM,
           "grad_ckpt": GRAD_CKPT, "llrd": LLRD, "max_len": MAX_LEN, "warmup": WARMUP, "opt": OPT, "label_smoothing": LS,
           "class_weight": CLASS_WEIGHT, "preproc": PREPROC, "dedup": DEDUP, "seed": SEED, "classes": classes,
           "noise_file": NOISE_FILE, "noise_mode": NOISE_MODE if NOISE else "", "noise_w": NOISE_W if NOISE_MODE == "weight" else None,
           "noise_n": len(NOISE), "rdrop": RDROP, "awp": AWP, "awp_eps": AWP_EPS if AWP else None,
           "awp_start": AWP_START if AWP else None, "pseudo_file": PSEUDO_FILE,
           "pseudo_t": PSEUDO_T if PSEUDO_FILE else None, "pseudo_w": PSEUDO_W if PSEUDO_FILE else None,
           "lora_r": LORA_R, "lora_alpha": LORA_ALPHA if LORA_R else None, "model_dtype": str(MODEL_DTYPE), "load_4bit": LOAD_4BIT, "train_frac": TRAIN_FRAC,
           "aug_file": AUG_FILE, "aug_n": AUG_N if AUG_FILE else None, "aug_w": AUG_W if AUG_FILE else None,
           "retried_folds": retries, "minutes": round((time.time() - t0) / 60, 1), "fold_log": log,
           "oof_f1_per_epoch": [float(f1_score(y, oof[e].argmax(1), average="macro")) for e in range(epochs)]}
json.dump(summary, open(out / "summary.json", "w"), indent=1)
for e in range(epochs):
    fs = [r["f1"] for r in log if r["epoch"] == e + 1]
    print(f"SUMMARY {run_name} ep{e+1}: fold mean {np.mean(fs):.4f} sd {np.std(fs):.4f} | OOF {summary['oof_f1_per_epoch'][e]:.4f}")
