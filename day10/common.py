"""Phần dùng chung: prompt, chuyển nhãn YOLO -> JSON 0-1000, nạp model 4-bit."""
import json
import random
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

PROMPT_NORM1000 = (
    "Locate road damage in this image. "
    "Detect only objects of classes: pothole, crack, manhole. "
    "Output a JSON list of objects with label and bbox_2d coordinates "
    "normalized on a 0-1000 scale: "
    '[{"label": "...", "bbox_2d": [x1, y1, x2, y2]}]. '
    "If no damage is found, output []."
)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def yolo_to_box(label, yolo, width=1000, height=1000):
    """YOLO [xc, yc, w, h] chuẩn hoá [0,1] -> {"label", "bbox_2d": [x1, y1, x2, y2]} số nguyên.

    Mặc định width = height = 1000 nghĩa là thang 0-1000 của Qwen3-VL.
    """
    xc, yc, w, h = yolo
    x1 = int(round(clamp((xc - w / 2) * width, 0, width)))
    y1 = int(round(clamp((yc - h / 2) * height, 0, height)))
    x2 = int(round(clamp((xc + w / 2) * width, 0, width)))
    y2 = int(round(clamp((yc + h / 2) * height, 0, height)))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return {"label": label, "bbox_2d": [x1, y1, x2, y2]}


def objects_to_answer(objects):
    """Câu trả lời đích: JSON gọn, box sắp theo (y1, x1) để thứ tự ổn định cho model học."""
    boxes = [yolo_to_box(o["label"], o["yolo"]) for o in objects]
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))
    return json.dumps(boxes, separators=(",", ":"))


def read_split(data_dir, split):
    rows = []
    with open(Path(data_dir) / "splits" / f"{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def subsample(rows, fraction, seed=42):
    """Tập con cố định (dùng cho E4 data_size)."""
    if fraction >= 1.0:
        return list(rows)
    k = max(1, int(round(len(rows) * fraction)))
    return random.Random(seed).sample(list(rows), k)


def load_image(data_dir, row):
    from PIL import Image
    return Image.open(Path(data_dir) / row["image_path"]).convert("RGB")


def user_message(image, prompt=PROMPT_NORM1000):
    return {"role": "user", "content": [{"type": "image", "image": image},
                                        {"type": "text", "text": prompt}]}


def quantization_config():
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        # ViT (gồm cả merger/deepstack) và lm_head giữ fp16: merger phải là tham số thật
        # thì E2/E3 mới huấn luyện đầy đủ được.
        llm_int8_skip_modules=["visual", "lm_head"],
    )


def load_base_model():
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=quantization_config(),
        dtype=torch.float16, device_map={"": 0})
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def projector_names(name):
    return "visual.merger" in name or "visual.deepstack_merger_list" in name
