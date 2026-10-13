"""Kiểm tra trước khi train (chạy được trên CPU, không cần tải trọng số):
- số tham số từng khối (ViT, projector, LLM) đếm trên model "meta",
- số module LoRA khớp regex (252 / 348),
- số token ảnh của ảnh 640x360 (220), và giải mã phần labels != -100.

python -m day10.checks --out report/checks.json
"""
import argparse
import json
from pathlib import Path

import torch

from day10.common import MODEL_ID, load_image, objects_to_answer, read_split
from day10.encode import IGNORE, collate, count_image_tokens, encode_example
from day10.lora_setup import matched_modules


def block_param_counts(model):
    groups = {"vit_blocks_and_patch_embed": 0, "visual.merger": 0,
              "visual.deepstack_merger_list": 0, "llm (language_model + lm_head)": 0}
    for n, p in model.named_parameters():
        if "visual.merger" in n:
            groups["visual.merger"] += p.numel()
        elif "visual.deepstack_merger_list" in n:
            groups["visual.deepstack_merger_list"] += p.numel()
        elif n.startswith("model.visual"):
            groups["vit_blocks_and_patch_embed"] += p.numel()
        else:
            groups["llm (language_model + lm_head)"] += p.numel()
    groups["total_unique"] = sum(p.numel() for p in model.parameters())
    groups["tie_word_embeddings"] = bool(getattr(model.config.text_config, "tie_word_embeddings",
                                                 model.config.tie_word_embeddings))
    return groups


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="report/checks.json")
    a = ap.parse_args(argv)

    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
    config = AutoConfig.from_pretrained(MODEL_ID)
    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration(config)
    res = {"params": block_param_counts(model),
           "lora_modules": {s: len(matched_modules(model, s)) for s in ("E1", "E2", "E3")},
           "deepstack_visual_indexes": list(config.vision_config.deepstack_visual_indexes)}

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    row = read_split(a.data, "train")[0]
    answer = objects_to_answer(row["objects"])
    enc = encode_example(processor, load_image(a.data, row), answer)
    supervised = enc["labels"][0][enc["labels"][0] != IGNORE]
    res["example"] = {
        "image_id": row["image_id"], "image_size": [row["width"], row["height"]],
        "image_grid_thw": enc["image_grid_thw"].tolist(),
        "image_tokens": count_image_tokens(processor, enc),
        "image_pad_tokens_in_input_ids": int((enc["input_ids"] == processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")).sum()),
        "seq_len": int(enc["input_ids"].shape[1]),
        "supervised_tokens": int(supervised.numel()),
        "answer": answer,
        "decoded_supervised": processor.tokenizer.decode(supervised),
        "keys": sorted(enc.keys()),
    }
    batch = collate([enc, encode_example(processor, load_image(a.data, read_split(a.data, "train")[1]),
                                         objects_to_answer(read_split(a.data, "train")[1]["objects"]))],
                    processor.tokenizer.pad_token_id)
    res["collate_shapes"] = {k: list(v.shape) for k, v in batch.items()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
