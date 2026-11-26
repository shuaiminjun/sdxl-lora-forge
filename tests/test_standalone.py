"""自带训练器的纯计算部分（不需要 torch / GPU，CI 里也跑得动）。"""

from __future__ import annotations

import math

from forge.standalone import bucket_size, estimate_steps, peft_key_to_kohya


def test_bucket_square():
    assert bucket_size(1024, 1024, 1024) == (1024, 1024)


def test_bucket_portrait_keeps_short_side():
    # 832×1216 是 SDXL 常用的竖向尺寸；短边给 832，长边按比例
    w, h = bucket_size(832, 1216, 832)
    assert w == 832
    assert h % 8 == 0
    assert 1200 <= h <= 1232


def test_bucket_landscape_swaps():
    w, h = bucket_size(1216, 832, 832)
    assert h == 832
    assert w >= 1184


def test_bucket_explicit_height_wins():
    assert bucket_size(1024, 1536, 1024, 1536) == (1024, 1536)


def test_bucket_rounds_to_multiple_of_eight():
    w, h = bucket_size(1001, 1499, 1001, 1499)
    assert w % 8 == 0 and h % 8 == 0


def test_estimate_steps_monotonic():
    assert estimate_steps(120) > estimate_steps(60)
    assert estimate_steps(60, epochs=10) == 600
    assert estimate_steps(0) >= 1


def test_kohya_key_unet_query():
    got = peft_key_to_kohya("down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight")
    assert got is not None
    name, side = got
    assert side == "down"
    assert name.startswith("lora_unet_")
    assert name.endswith("to_q.lora_down.weight")
    assert "." not in name.removeprefix("lora_unet_").replace(".lora_down.weight", "")


def test_kohya_key_text_encoder_prefix():
    got = peft_key_to_kohya("text_model.encoder.layers.0.self_attn.q_proj.lora_B.weight", "te1")
    assert got is not None and got[0].startswith("lora_te1_") and got[1] == "up"


def test_kohya_key_ignores_plain_weights():
    assert peft_key_to_kohya("down_blocks.0.resnets.0.conv1.weight") is None


def test_kohya_key_strips_module_prefix():
    got = peft_key_to_kohya("module.foo.to_q.lora_A.weight")
    assert got is not None and got[0].startswith("lora_unet_foo")
