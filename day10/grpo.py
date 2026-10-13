"""Bước 9 (E5): GRPO bằng trl.GRPOTrainer với reward IoU, khởi tạo từ adapter SFT tốt nhất.

python -m day10.grpo --adapter work/e2/adapter --projector work/e2/projector.pt --out work/e5
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset

from day10.common import PROMPT_NORM1000, load_image, objects_to_answer, read_split
from day10.infer import load_trained
from day10.reward import compute_reward, gt_boxes_from_objects
from scripts.score import parse_output


class PromptDataset(Dataset):
    def __init__(self, rows, data_dir):
        self.rows, self.data_dir = rows, data_dir

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        return {"prompt": [{"role": "user", "content": PROMPT_NORM1000}],
                "image": load_image(self.data_dir, row),
                "gt": json.dumps(gt_boxes_from_objects(row["objects"]))}


class RewardWithLog:
    """Reward IoU + ghi lại dấu hiệu reward hacking ([] , box khổng lồ, số box) cho từng completion."""
    __name__ = "iou_reward"

    def __init__(self, path):
        self.log = open(path, "w", encoding="utf-8")
        self.calls = 0

    def __call__(self, prompts, completions, gt, **kwargs):
        rewards = []
        for completion, gt_json in zip(completions, gt):
            text = completion[-1]["content"] if isinstance(completion, list) else completion
            if isinstance(text, list):
                text = "".join(part.get("text", "") for part in text)
            boxes = [{"label": b["label"], "bbox": tuple(b["bbox"])} for b in json.loads(gt_json)]
            r = compute_reward(text, boxes)
            pred = parse_output(text)
            giant = sum(1 for p in (pred or [])
                        if (p["bbox"][2] - p["bbox"][0]) * (p["bbox"][3] - p["bbox"][1]) > 250_000)
            self.log.write(json.dumps({"call": self.calls, "reward": r, "valid": pred is not None,
                                       "n_pred": len(pred or []), "n_gt": len(boxes),
                                       "empty": pred == [], "giant_boxes": giant}) + "\n")
            rewards.append(r)
        self.log.flush()
        self.calls += 1
        return rewards


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--projector")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--max-completion-length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    from peft import PeftModel
    from trl import GRPOConfig, GRPOTrainer

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base, processor = load_trained(adapter=None, projector=a.projector)
    model = PeftModel.from_pretrained(base, a.adapter, is_trainable=True)
    processor.tokenizer.padding_side = "left"

    # Chỉ giữ ảnh mà đáp án đầy đủ vừa trong max_completion_length: nếu không, completion đúng
    # cũng bị cắt thành JSON hỏng (reward 0) và GRPO sẽ học cách trả ít box hơn.
    rows = [r for r in read_split(a.data, "train")
            if len(processor.tokenizer(objects_to_answer(r["objects"]))["input_ids"]) + 2
            <= a.max_completion_length]
    random.Random(a.seed).shuffle(rows)
    print(f"GRPO prompts available: {len(rows)}", flush=True)

    args = GRPOConfig(
        output_dir=str(out / "trainer"), max_steps=a.max_steps, learning_rate=a.lr,
        per_device_train_batch_size=a.num_generations, gradient_accumulation_steps=2,
        num_generations=a.num_generations, max_completion_length=a.max_completion_length,
        beta=0.0, use_vllm=False, fp16=True, bf16=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, save_strategy="no", report_to="none", seed=a.seed,
        lr_scheduler_type="constant", warmup_steps=0,
    )
    reward = RewardWithLog(out / "reward_log.jsonl")
    trainer = GRPOTrainer(model=model, reward_funcs=reward, args=args,
                          train_dataset=PromptDataset(rows, a.data), processing_class=processor)
    # TRL đưa tham số LoRA của model lượng tử hoá về bf16; T4 không có bf16 và GradScaler fp16 cần fp32.
    for p in trainer.model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    minutes = (time.time() - t0) / 60
    trainer.model.save_pretrained(out / "adapter")
    (out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1),
                                          encoding="utf-8")
    run = {"setup": "E5-GRPO", "init_adapter": a.adapter, "projector": a.projector,
           "max_steps": a.max_steps, "lr": a.lr, "num_generations": a.num_generations,
           "max_completion_length": a.max_completion_length, "beta": 0.0,
           "prompts_per_step": 2, "trainable_params": trainable,
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
           "train_minutes": round(minutes, 1), "gpu": torch.cuda.get_device_name(0)}
    (out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps(run), flush=True)


if __name__ == "__main__":
    main()
