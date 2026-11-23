"""Build a Kaggle script kernel from train_template.py (fills in the run settings and the local CV folds).

usage: python build_kernel.py <run> <model> <epochs> <lr> <batch> <lora_r> [ls=0] [prompt=0] [zero_head=0] [targets=all-linear]
       -> data/kaggle/kaggle_build/<run>/train.py + kernel-metadata.json   (push with: kaggle kernels push -p <that folder>)
The Kaggle user name is read from ~/.kaggle/kaggle.json; the key itself is never read here.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import folds, load

run, model, epochs, lr, batch, lora_r = sys.argv[1:7]
ls, prompt, zero_head, targets = (sys.argv[7:11] + ["0", "0", "0", "all-linear"][len(sys.argv) - 7:])
train = load()[0]
# the same 5 folds as the local runs, so Kaggle out-of-fold probabilities can be blended with them
fold_of = {train.post_id[i]: f for f, (_, va) in enumerate(folds(train)) for i in va}
code = (HERE / "train_template.py").read_text(encoding="utf-8")
for k, v in {"__RUN__": run, "__MODEL__": model, "__EPOCHS__": epochs, "__LR__": lr, "__BATCH__": batch, "__LORA_R__": lora_r,
             "__LS__": ls, "__PROMPT__": prompt, "__ZERO_HEAD__": zero_head, "__TARGETS__": targets,
             "__FOLDS__": ",".join(f"{k}:{v}" for k, v in fold_of.items())}.items():
    code = code.replace(k, v)
out = HERE.parent.parent / "data" / "kaggle" / "kaggle_build" / run  # git-ignored
out.mkdir(parents=True, exist_ok=True)
(out / "train.py").write_text(code, encoding="utf-8", newline="\n")
user = json.load(open(Path.home() / ".kaggle" / "kaggle.json"))["username"]
slug = "dm-lab2-" + run.lower().replace("_", "-")
json.dump({"id": f"{user}/{slug}", "title": slug, "code_file": "train.py", "language": "python", "kernel_type": "script",
           "is_private": True, "enable_gpu": True, "enable_internet": True,
           "competition_sources": ["2026-dm-lab-2-private-competition"], "dataset_sources": [], "kernel_sources": [],
           "model_sources": []}, open(out / "kernel-metadata.json", "w"), indent=1)
print(out, slug)
