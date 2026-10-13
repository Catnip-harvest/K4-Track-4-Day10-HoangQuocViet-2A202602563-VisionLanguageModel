"""Bước 3 và 7: suy luận greedy theo batch, ghi {"image_id", "raw"} ra .jsonl.

E0:  python -m day10.infer --split test --out work/e0.jsonl
E2:  python -m day10.infer --adapter work/e2/adapter --projector work/e2/projector.pt --out work/e2.jsonl
"""
import argparse
import json
import time
from pathlib import Path

import torch

from day10.common import load_base_model, load_image, read_split, user_message


def load_trained(adapter=None, projector=None):
    model, processor = load_base_model()
    if projector:
        # Nạp projector đã huấn luyện vào model gốc; thiếu bước này E2/E3 sẽ trông như E1.
        state = torch.load(projector, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"projector có khoá lạ: {unexpected[:3]}")
        print(f"projector loaded: {len(state)} tensors", flush=True)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, processor


@torch.no_grad()
def generate(model, processor, images, max_new_tokens=256):
    processor.tokenizer.padding_side = "left"
    conversations = [[user_message(im)] for im in images]
    inputs = processor.apply_chat_template(
        conversations, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new, skip_special_tokens=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--adapter")
    ap.add_argument("--projector")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    a = ap.parse_args(argv)

    model, processor = load_trained(a.adapter, a.projector)
    rows = read_split(a.data, a.split)
    if a.limit:
        rows = rows[:a.limit]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(a.out, "w", encoding="utf-8") as f:
        for i in range(0, len(rows), a.batch_size):
            chunk = rows[i:i + a.batch_size]
            raws = generate(model, processor, [load_image(a.data, r) for r in chunk],
                            a.max_new_tokens)
            for r, raw in zip(chunk, raws):
                f.write(json.dumps({"image_id": r["image_id"], "raw": raw}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{i + len(chunk)}/{len(rows)} {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"infer_minutes={(time.time() - t0) / 60:.1f}", flush=True)


if __name__ == "__main__":
    main()
