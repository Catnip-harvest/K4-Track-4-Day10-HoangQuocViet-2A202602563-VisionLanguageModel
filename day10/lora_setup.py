"""Bước 5: gắn LoRA và đóng băng / mở khối cho E1, E2, E3."""
import re

import torch
from peft import LoraConfig, get_peft_model

from day10.common import projector_names

LLM_TARGETS = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
VIT_TARGETS = r"model\.visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc(1|2))"

SETUPS = {
    "E1": {"vit_lora": False, "train_projector": False},
    "E2": {"vit_lora": False, "train_projector": True},
    "E3": {"vit_lora": True, "train_projector": True},
}


def target_regex(setup):
    if SETUPS[setup]["vit_lora"]:
        return f"({LLM_TARGETS})|({VIT_TARGETS})"
    return LLM_TARGETS


def matched_modules(model, setup):
    pattern = re.compile(target_regex(setup))
    return [n for n, _ in model.named_modules() if pattern.fullmatch(n)]


def apply_setup(model, setup, rank=16, alpha=None):
    alpha = alpha or 2 * rank
    for p in model.parameters():
        p.requires_grad = False
    model.enable_input_require_grads()  # cần cho gradient checkpointing

    matched = matched_modules(model, setup)
    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05,
                        target_modules=target_regex(setup), bias="none")
    peft_model = get_peft_model(model, config)

    if SETUPS[setup]["train_projector"]:
        for name, param in peft_model.named_parameters():
            if projector_names(name):
                param.data = param.data.to(torch.float32)  # fp16 dễ tràn số khi cập nhật trực tiếp
                param.requires_grad = True

    # GradScaler không unscale được gradient fp16: mọi tham số huấn luyện phải là fp32.
    for p in peft_model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.to(torch.float32)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    projector = sum(p.numel() for n, p in peft_model.named_parameters()
                    if p.requires_grad and projector_names(n))
    return peft_model, {"matched_modules": len(matched), "trainable_params": trainable,
                        "trainable_projector_params": projector}


def projector_state(peft_model):
    """Tham số projector đã huấn luyện, khoá theo tên trong model gốc (không có tiền tố peft)."""
    base = peft_model.get_base_model()
    return {n: p.detach().to("cpu", torch.float16).clone()
            for n, p in base.named_parameters() if projector_names(n)}
