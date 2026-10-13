"""Chạy toàn bộ thí nghiệm trên Kaggle (GPU T4 x2) thay cho Colab.

Kaggle cho 2 GPU T4 16 GB: mỗi GPU chạy một hàng đợi thí nghiệm riêng (CUDA_VISIBLE_DEVICES),
nên E0–E5 xong trong một phiên. Script này được nhúng vào kernel bởi kaggle/build.py;
STAGE và REF được điền lúc build. Không chứa khoá: dữ liệu gắn qua dataset_sources của Kaggle.
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

STAGE = "__STAGE__"
REF = "__REF__"
REPO_URL = "https://github.com/Catnip-harvest/K4-Track-4-Day10-HoangQuocViet-2A202602563-VisionLanguageModel.git"
REPO = Path("/tmp/repo")
OUT = Path("/kaggle/working/out")      # tải về máy: dự đoán, log, run.json
CKPT = Path("/kaggle/working/ckpt")    # adapter/projector: giữ trên Kaggle, không đưa vào git


def sh(cmd, **kw):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, **kw)


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    sh("nvidia-smi")
    if REPO.exists():
        shutil.rmtree(REPO)
    sh(f"git clone -q {REPO_URL} {REPO} && cd {REPO} && git checkout -q {REF} && git log --oneline -1",
       check=True)
    sh(f"cd {REPO} && pip install -q -r requirements-colab.txt 2>&1 | tail -3", check=True)
    # torchao 0.10 có sẵn trên Kaggle làm peft lỗi khi gắn LoRA vào nn.Linear fp16 của ViT (E3).
    sh("pip uninstall -y -q torchao")
    sh(f"cd {REPO} && python scripts/prepare_data.py --source /kaggle/input --work /tmp/w",
       check=True)
    sh("python -c \"import torch, transformers, peft, trl, bitsandbytes as b; "
       "print(torch.__version__, transformers.__version__, peft.__version__, trl.__version__, b.__version__)\"")
    sh("python -c \"from huggingface_hub import snapshot_download; "
       "snapshot_download('Qwen/Qwen3-VL-4B-Instruct')\"", check=True)


def run(gpu, name, args):
    """Một bước trong hàng đợi; lỗi được ghi lại nhưng không làm sập cả phiên."""
    log = OUT / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1",
           "TOKENIZERS_PARALLELISM": "false"}
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run([sys.executable, "-m", *args], cwd=REPO, env=env,
                            stdout=f, stderr=subprocess.STDOUT).returncode
    status = {"name": name, "gpu": gpu, "rc": rc, "minutes": round((time.time() - t0) / 60, 1)}
    print(json.dumps(status), flush=True)
    with open(OUT / "status.jsonl", "a") as f:
        f.write(json.dumps(status) + "\n")
    return rc == 0


def train(gpu, name, setup_name, *extra):
    ok = run(gpu, f"train_{name}", ["day10.train", "--setup", setup_name, "--out", str(CKPT / name), *extra])
    if ok:
        shutil.copy(CKPT / name / "run.json", OUT / f"{name}_run.json")
        shutil.copy(CKPT / name / "train_log.jsonl", OUT / f"{name}_train_log.jsonl")
    return ok


def infer(gpu, name, ckpt=None, split="test", limit=0):
    args = ["day10.infer", "--split", split, "--out", str(OUT / f"{name}.jsonl")]
    if limit:
        args += ["--limit", str(limit)]
    if ckpt:
        args += ["--adapter", str(CKPT / ckpt / "adapter")]
        if (CKPT / ckpt / "projector.pt").exists():
            args += ["--projector", str(CKPT / ckpt / "projector.pt")]
    return run(gpu, f"infer_{name}", args)


def f1_of(name):
    tmp = OUT / "selection" / f"{name}.json"
    sh(f"cd {REPO} && python scripts/score.py --pred {OUT / (name + '.jsonl')} --name {name} --out {tmp}")
    return json.loads(tmp.read_text())["micro"]["f1"] if tmp.exists() else -1.0


def parallel(*queues):
    threads = [threading.Thread(target=q) for q in queues]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def smoke():
    def gpu0():
        if train(0, "s3", "E3", "--max-steps", "4", "--grad-accum", "2"):
            infer(0, "s3_val", "s3", split="val", limit=16)

    def gpu1():
        if train(1, "s2", "E2", "--max-steps", "3", "--grad-accum", "1"):
            infer(1, "s2_val", "s2", split="val", limit=8)
    parallel(gpu0, gpu1)


def full():
    def gpu0():
        train(0, "e1", "E1") and infer(0, "e1", "e1")
        train(0, "e3", "E3") and infer(0, "e3", "e3")

    def gpu1():
        infer(1, "e0")
        train(1, "e2", "E2") and infer(1, "e2", "e2")
    parallel(gpu0, gpu1)

    # Setup tốt nhất của E1–E3 làm gốc cho E4 và E5 (như GUIDE bước 8–9).
    scores = {s: f1_of(s) for s in ("e1", "e2", "e3")}
    best = max(scores, key=scores.get)
    (OUT / "best.json").write_text(json.dumps({"scores": scores, "best": best}))
    print("best", best, scores, flush=True)

    def gpu0_e4():
        train(0, "e4_data_0.1", best.upper(), "--train-fraction", "0.1") and infer(0, "e4_data_0.1", "e4_data_0.1")

    def gpu1_e5():
        args = ["day10.grpo", "--adapter", str(CKPT / best / "adapter"), "--out", str(CKPT / "e5")]
        if (CKPT / best / "projector.pt").exists():
            args += ["--projector", str(CKPT / best / "projector.pt")]
        if run(1, "grpo_e5", args):
            for f in ("run.json", "reward_log.jsonl", "log_history.json"):
                shutil.copy(CKPT / "e5" / f, OUT / f"e5_{f}")
            proj = ["--projector", str(CKPT / best / "projector.pt")] if (CKPT / best / "projector.pt").exists() else []
            run(1, "infer_e5", ["day10.infer", "--split", "test", "--out", str(OUT / "e5.jsonl"),
                                "--adapter", str(CKPT / "e5" / "adapter"), *proj])
    parallel(gpu0_e4, gpu1_e5)


if __name__ == "__main__":
    setup()
    {"smoke": smoke, "full": full}[STAGE]()
    sh(f"du -sh {OUT} {CKPT}; ls -la {OUT}")
