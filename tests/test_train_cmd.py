"""sd-scripts command builder: flags present, VRAM estimate honest."""

from __future__ import annotations

from forge.train_cmd import TrainConfig, build_command, estimate_vram_gb, render


def test_command_has_the_flags_that_actually_matter():
    cfg = TrainConfig()
    argv = build_command(cfg)
    joined = " ".join(argv)
    assert "sdxl_train_network.py" in joined
    assert "--network_module=networks.lora" in argv
    assert "--gradient_checkpointing" in argv
    assert "--no_half_vae" in argv          # fp16 VAE → black images
    assert "--network_train_unet_only" in argv


def test_bucket_flags_toggle():
    cfg = TrainConfig(bucket=False)
    assert "--no_bucket" in build_command(cfg)
    cfg = TrainConfig(bucket=True, min_bucket_reso=640, max_bucket_reso=1024)
    argv = build_command(cfg)
    assert "--min_bucket_reso=640" in argv and "--max_bucket_reso=1024" in argv


def test_validation_catches_common_mistakes():
    assert TrainConfig(resolution=1000).validate()
    assert TrainConfig(network_dim=16, network_alpha=32).validate()
    assert TrainConfig(max_train_steps=100).validate()
    assert TrainConfig(mixed_precision="fp8").validate()
    assert TrainConfig(gradient_checkpointing=False).validate()
    assert TrainConfig().validate() == []


def test_vram_estimate_scales_with_resolution_and_batch():
    base = estimate_vram_gb(TrainConfig(resolution=768))["total_gb"]
    bigger = estimate_vram_gb(TrainConfig(resolution=1024))["total_gb"]
    deeper = estimate_vram_gb(TrainConfig(batch_size=4))["total_gb"]
    assert base < bigger < deeper


def test_dropping_gradient_checkpointing_costs_several_gb():
    """权重是固定成本，所以看"多花多少"比看倍数更有决策意义。"""
    on = estimate_vram_gb(TrainConfig(gradient_checkpointing=True))["total_gb"]
    off = estimate_vram_gb(TrainConfig(gradient_checkpointing=False))["total_gb"]
    assert off - on >= 5.0


def test_render_says_whether_it_fits():
    tight = render(TrainConfig(resolution=1024, batch_size=4), vram_limit_gb=20.0)
    assert "WILL NOT FIT" in tight
    ok = render(TrainConfig(resolution=768), vram_limit_gb=20.0)
    assert "OK" in ok and "accelerate launch" in ok


def test_render_lists_problems():
    out = render(TrainConfig(resolution=1000, max_train_steps=50), vram_limit_gb=20.0)
    assert "PROBLEMS" in out and "multiple of 64" in out
