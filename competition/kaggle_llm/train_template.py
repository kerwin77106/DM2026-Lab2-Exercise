"""Kaggle GPU kernel: 5-fold QLoRA fine-tuning of a 7B decoder LLM as a 6-class emotion classifier.

Same folds as the local CV runs (the fold of every training post is embedded below by build_kernel.py), so the
out-of-fold probabilities can be blended with the local models. One worker process per GPU (Kaggle gives T4 x2).
Output: /kaggle/working/fold<f>.npz with per-epoch validation and test probabilities, keyed by post_id.

Local smoke test (small model, no quantisation, few rows):
    SMOKE=1 MODEL=Qwen/Qwen2.5-1.5B DATA=<data/kaggle> python train.py
"""
import glob
import json
import os
import subprocess
import sys
import time

RUN = "__RUN__"
MODEL = os.environ.get("MODEL", "__MODEL__")
EPOCHS, LR, BATCH, MAX_LEN, LORA_R, SEED = __EPOCHS__, __LR__, __BATCH__, 128, __LORA_R__, 42
LS = __LS__            # label smoothing
PROMPT = __PROMPT__    # 1: wrap each post in an instruction that ends with "Emotion:", so the last token (which the
                       # classification head reads) is a position where the model is about to name the emotion
ZERO_HEAD = __ZERO_HEAD__  # 1: start the classification head at zero (loss starts at ln 6 instead of ~2.8)
TARGETS = "__TARGETS__"    # LoRA target modules: "all-linear" or a comma list such as "q_proj,k_proj,v_proj,o_proj"
TEMPLATE = ("Classify the emotion expressed in this social media post as one of: anger, disgust, fear, joy, sadness, "
            "surprise.\nPost: {}\nEmotion:")
SMOKE = os.environ.get("SMOKE") == "1"
CLASSES = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
FOLD_OF = {k: int(v) for k, v in (kv.split(":") for kv in "__FOLDS__".split(","))}
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.environ.get("OUT", "./kout")
os.makedirs(OUT, exist_ok=True)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
t0 = time.time()


def find_data():
    if os.environ.get("DATA"):
        return os.environ["DATA"]
    hits = glob.glob("/kaggle/input/**/final_posts.json", recursive=True)
    if not hits:
        raise SystemExit("final_posts.json not found under /kaggle/input")
    return os.path.dirname(hits[0])


def worker(my_folds):
    import numpy as np
    import pandas as pd
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from sklearn.metrics import f1_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup

    d = find_data()
    posts = pd.DataFrame(json.load(open(os.path.join(d, "final_posts.json"), encoding="utf-8")))
    df = posts.merge(pd.read_csv(os.path.join(d, "data_identification.csv")), on="post_id")
    train = df[df["set"] == "train"].merge(pd.read_csv(os.path.join(d, "emotion.csv")), on="post_id").reset_index(drop=True)
    test = df[df["set"] == "test"].reset_index(drop=True)
    assert sorted(train["label"].unique()) == CLASSES and set(train["post_id"]) == set(FOLD_OF)
    train["y"] = train["label"].map({c: i for i, c in enumerate(CLASSES)})
    train["fold"] = train["post_id"].map(FOLD_OF)
    if SMOKE:
        test = test.head(16)
    K, dev = len(CLASSES), "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "right"  # the classification head reads the last non-padding token
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    max_len = MAX_LEN
    if PROMPT:  # cut the post itself to MAX_LEN tokens first, so truncation can never remove the final "Emotion:"
        def wrap(texts):
            ids = tok(texts, truncation=True, max_length=MAX_LEN, add_special_tokens=False)["input_ids"]
            return [TEMPLATE.format(t) for t in tok.batch_decode(ids, skip_special_tokens=True)]
        train["text"], test["text"] = wrap(train["text"].tolist()), wrap(test["text"].tolist())
        max_len = MAX_LEN + 64

    def batches(texts, labels=None, shuffle=False, bs=BATCH):
        order = np.random.permutation(len(texts)) if shuffle else np.argsort([len(t) for t in texts], kind="stable")
        for s in range(0, len(texts), bs):
            b = order[s:s + bs]
            enc = tok([texts[i] for i in b], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(dev)
            yield enc, (torch.tensor(labels[b], device=dev) if labels is not None else None), b

    for f in my_folds:
        torch.manual_seed(SEED + f); np.random.seed(SEED + f)
        tr, va = train[train["fold"] != f], train[train["fold"] == f]
        if SMOKE:
            tr, va = tr.head(48), va.head(32)
        tr_x, tr_y, va_x, va_y, te_x = tr["text"].tolist(), tr["y"].values, va["text"].tolist(), va["y"].values, test["text"].tolist()
        quant = None if SMOKE else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                      bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=K, torch_dtype=torch.float16,
                                                                   quantization_config=quant, device_map={"": 0})
        model.config.pad_token_id = tok.pad_token_id
        if quant is not None:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)  # T4 16 GB: batch 8 ran out of memory without it
        model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, r=LORA_R, lora_alpha=2 * LORA_R,
                                                 lora_dropout=0.05,
                                                 target_modules=TARGETS if TARGETS == "all-linear" else TARGETS.split(",")))
        if ZERO_HEAD:
            for n, p in model.named_parameters():
                if p.requires_grad and "score" in n and "lora" not in n:
                    p.data.zero_()
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()  # adapters and head in fp32, frozen base stays fp16 / 4-bit
        if f == my_folds[0]:
            model.print_trainable_parameters()

        @torch.no_grad()
        def proba(texts):
            model.eval(); res = np.zeros((len(texts), K), dtype=np.float32)
            for enc, _, b in batches(texts, bs=16):
                with torch.autocast(dev, dtype=torch.float16):
                    res[b] = torch.softmax(model(**enc).logits.float(), -1).cpu().numpy()
            return res

        steps = EPOCHS * int(np.ceil(len(tr_x) / BATCH))
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        scaler = torch.amp.GradScaler(dev)
        val_p, test_p, f1s, skipped = [], [], [], 0
        for ep in range(EPOCHS):
            model.train(); losses = []
            for enc, lab, _ in batches(tr_x, tr_y, shuffle=True):
                with torch.autocast(dev, dtype=torch.float16):
                    logits = model(**enc).logits.float()
                loss = torch.nn.functional.cross_entropy(logits, lab, label_smoothing=LS)
                if not torch.isfinite(loss):  # fp16 overflow: drop this batch instead of poisoning the weights
                    skipped += 1; opt.zero_grad(set_to_none=True); sched.step(); continue
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True)
                losses.append(loss.item())
            pv = proba(va_x); val_p.append(pv); test_p.append(proba(te_x))
            f1s.append(float(f1_score(va_y, pv.argmax(1), average="macro")))
            print(f"{RUN} fold{f} ep{ep + 1} macroF1={f1s[-1]:.4f} train_loss={np.mean(losses) if losses else float('nan'):.3f} "
                  f"n_train={len(tr_x)} skipped={skipped} ({time.time() - t0:.0f}s)", flush=True)
        np.savez(os.path.join(OUT, f"fold{f}.npz"), val_ids=va["post_id"].values.astype(str), val_p=np.stack(val_p),
                 test_ids=test["post_id"].values.astype(str), test_p=np.stack(test_p), f1=np.array(f1s))
        del model, opt, scaler; torch.cuda.empty_cache()


if os.environ.get("WORKER_FOLDS"):
    wf = [int(x) for x in os.environ["WORKER_FOLDS"].split(",")]
    if len(wf) == 1:
        worker(wf)
        sys.exit(0)
    # one fresh process per fold: a 14B model left over from the previous fold does not fit next to a new one on a T4
    rc = [subprocess.run([sys.executable, globals().get("__file__") or sys.argv[0]],
                         env={**os.environ, "WORKER_FOLDS": str(f)}).returncode for f in wf]
    sys.exit(max(rc))

# ---- parent: install what the image lacks, fetch the model once, then one worker per GPU ----
if not SMOKE:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes", "peft"], check=False)
import torch
from huggingface_hub import snapshot_download

snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "*.txt"])
n_gpu = max(1, torch.cuda.device_count())
all_folds = [0] if SMOKE else [0, 1, 2, 3, 4]
print(f"{RUN} model={MODEL} gpus={n_gpu} folds={all_folds} epochs={EPOCHS} lr={LR} batch={BATCH} lora_r={LORA_R} ls={LS} prompt={PROMPT} zero_head={ZERO_HEAD} targets={TARGETS}", flush=True)
src = globals().get("__file__") or sys.argv[0]
if n_gpu > 1 and os.path.exists(src):
    procs = [subprocess.Popen([sys.executable, src], env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g),
                                                          "WORKER_FOLDS": ",".join(map(str, all_folds[g::n_gpu]))})
             for g in range(n_gpu)]
    codes = [p.wait() for p in procs]
    print("worker exit codes", codes, flush=True)
else:
    worker(all_folds)
done = sorted(os.path.basename(p) for p in glob.glob(os.path.join(OUT, "fold*.npz")))
json.dump({"run": RUN, "model": MODEL, "epochs": EPOCHS, "lr": LR, "batch": BATCH, "lora_r": LORA_R, "max_len": MAX_LEN,
           "label_smoothing": LS, "prompt": PROMPT, "zero_head": ZERO_HEAD, "targets": TARGETS,
           "seed": SEED, "folds_done": done, "minutes": round((time.time() - t0) / 60, 1)},
          open(os.path.join(OUT, "run.json"), "w"), indent=1)
print("DONE", done, f"{(time.time() - t0) / 60:.1f} min", flush=True)
