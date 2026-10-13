"""Bước 4: mã hoá một ví dụ theo chat template, mask loss, và gom batch."""
import torch

from day10.common import PROMPT_NORM1000, user_message

IGNORE = -100


def encode_example(processor, image, answer, prompt=PROMPT_NORM1000):
    user = [user_message(image, prompt)]
    full = user + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}]
    kw = dict(tokenize=True, return_dict=True, return_tensors="pt")
    prompt_enc = processor.apply_chat_template(user, add_generation_prompt=True, **kw)
    full_enc = processor.apply_chat_template(full, add_generation_prompt=False, **kw)

    n = prompt_enc["input_ids"].shape[1]
    if not torch.equal(full_enc["input_ids"][0, :n], prompt_enc["input_ids"][0]):
        raise ValueError("prompt không phải tiền tố của chuỗi đầy đủ: kiểm tra chat template")

    labels = full_enc["input_ids"].clone()
    labels[:, :n] = IGNORE  # token ảnh + prompt không tính loss
    out = {k: v for k, v in full_enc.items()}
    out["labels"] = labels
    return out


def collate(batch, pad_token_id):
    """Đệm các tensor theo chuỗi (bên phải); nối pixel_values/image_grid_thw theo chiều 0, không pad."""
    seq_keys = {"input_ids": pad_token_id, "attention_mask": 0, "labels": IGNORE,
                "mm_token_type_ids": 0}
    longest = max(b["input_ids"].shape[1] for b in batch)
    out = {}
    for key, pad in seq_keys.items():
        if key not in batch[0]:
            continue
        rows = []
        for b in batch:
            t = b[key][0]
            rows.append(torch.cat([t, t.new_full((longest - t.shape[0],), pad)]))
        out[key] = torch.stack(rows)
    out["pixel_values"] = torch.cat([b["pixel_values"] for b in batch], dim=0)
    out["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in batch], dim=0)
    return out


def count_image_tokens(processor, enc):
    merge = processor.image_processor.merge_size
    return int(enc["image_grid_thw"].prod(-1).sum().item() // (merge * merge))
