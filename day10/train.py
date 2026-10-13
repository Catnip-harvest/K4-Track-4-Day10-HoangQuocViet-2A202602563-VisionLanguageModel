"""Bước 6: vòng huấn luyện QLoRA fp16 (AdamW, warmup 5% + cosine, GradScaler, tích luỹ gradient).

Ví dụ: python -m day10.train --setup E1 --out work/e1
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from day10.common import load_base_model, load_image, objects_to_answer, read_split, subsample
from day10.encode import collate, encode_example
from day10.lora_setup import apply_setup, projector_state, SETUPS


class GroundingDataset(Dataset):
    def __init__(self, rows, data_dir, processor):
        self.rows, self.data_dir, self.processor = rows, data_dir, processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        return encode_example(self.processor, load_image(self.data_dir, row),
                              objects_to_answer(row["objects"]))


def cosine_with_warmup(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


def save_checkpoint(model, setup, out):
    model.save_pretrained(out / "adapter")  # chỉ chứa LoRA
    if SETUPS[setup]["train_projector"]:
        torch.save(projector_state(model), out / "projector.pt")  # projector không nằm trong adapter


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True, choices=sorted(SETUPS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--train-fraction", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    model, processor = load_base_model()
    model, info = apply_setup(model, a.setup, rank=a.rank)
    print(json.dumps(info), flush=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()

    rows = subsample(read_split(a.data, "train"), a.train_fraction, seed=a.seed)
    print(f"train images: {len(rows)}", flush=True)
    pad_id = processor.tokenizer.pad_token_id
    loader = DataLoader(GroundingDataset(rows, a.data, processor), batch_size=a.batch_size,
                        shuffle=True, num_workers=2, collate_fn=lambda b: collate(b, pad_id),
                        generator=torch.Generator().manual_seed(a.seed), drop_last=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    warmup = max(1, int(0.05 * a.max_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: cosine_with_warmup(s, a.max_steps, warmup))
    scaler = torch.amp.GradScaler("cuda")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    log = open(out / "train_log.jsonl", "w", encoding="utf-8")
    batches = iter(loader)
    step = 0
    while step < a.max_steps:
        running = 0.0
        for _ in range(a.grad_accum):
            try:
                batch = next(batches)
            except StopIteration:
                batches = iter(loader)
                batch = next(batches)
            batch = {k: v.to("cuda") for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss / a.grad_accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss không hữu hạn ở bước {step}: {loss.item()} (tràn số fp16?)")
            scaler.scale(loss).backward()
            running += loss.item()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        step += 1
        rec = {"step": step, "loss": running, "lr": sched.get_last_lr()[0], "grad_norm": grad_norm,
               "elapsed_min": (time.time() - t0) / 60}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step % 10 == 0 or step == 1:
            print(json.dumps(rec), flush=True)
        if step % 100 == 0 or step == a.max_steps:
            save_checkpoint(model, a.setup, out)

    run = {
        "setup": a.setup, "rank": a.rank, "train_fraction": a.train_fraction,
        "train_images": len(rows), "max_steps": a.max_steps, "lr": a.lr,
        "batch_size": a.batch_size, "grad_accum": a.grad_accum, "seed": a.seed,
        "trainable_params": info["trainable_params"],
        "trainable_projector_params": info["trainable_projector_params"],
        "matched_lora_modules": info["matched_modules"],
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
        "train_minutes": round((time.time() - t0) / 60, 1),
        "gpu": torch.cuda.get_device_name(0),
    }
    (out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps(run), flush=True)


if __name__ == "__main__":
    main()
