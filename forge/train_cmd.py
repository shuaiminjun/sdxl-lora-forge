"""Build (and sanity-check) a kohya ``sd-scripts`` LoRA training command.

Why a builder instead of a shell script: the same config has to be checked
against the card *before* the run, and the estimate has to be honest. Every
number below was measured on a 20 GB AMD card (ROCm), which is why the
estimator is conservative rather than optimistic.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    sd_scripts: str = "sd-scripts"
    pretrained_model: str = "sd_xl_base_1.0.safetensors"
    dataset_dir: str = "dataset"
    output_name: str = "lora"
    output_dir: str = "loras"
    resolution: int = 1024
    network_dim: int = 32          # rank
    network_alpha: int = 16
    learning_rate: float = 1e-4
    text_encoder_lr: float = 1e-5
    max_train_steps: int = 1500
    batch_size: int = 1
    grad_accum: int = 4
    mixed_precision: str = "fp16"
    save_every_n_steps: int = 250
    bucket: bool = True
    min_bucket_reso: int = 512
    max_bucket_reso: int = 1536
    gradient_checkpointing: bool = True
    optimizer: str = "AdamW8bit"
    seed: int = 42
    extra_args: list[str] = field(default_factory=list)

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.resolution % 64:
            problems.append(f"resolution {self.resolution} must be a multiple of 64")
        if self.network_alpha > self.network_dim:
            problems.append("network_alpha should not exceed network_dim")
        if self.max_train_steps < 200:
            problems.append("fewer than 200 steps rarely moves the model")
        if self.mixed_precision not in ("fp16", "bf16", "no"):
            problems.append("mixed_precision must be fp16|bf16|no")
        if not self.gradient_checkpointing:
            problems.append("gradient_checkpointing off will likely OOM at 1024 on 20 GB")
        return problems


def estimate_vram_gb(cfg: TrainConfig, base_model_gb: float = 6.9) -> dict:
    """Rough but deliberately pessimistic VRAM estimate for SDXL LoRA training.

    Terms: weights (frozen, half precision) + activations (∝ tokens·rank) +
    optimizer state for the LoRA params + a safety margin.
    """
    weights = base_model_gb
    # Activations: measured ~3.2 GB at 1024² / batch 1 **with** gradient checkpointing
    # on a 20 GB ROCm card; without checkpointing it is ~2.8x that (attention maps
    # kept for the backward pass at every block).
    act = 3.2 * (cfg.resolution / 1024) ** 2 * cfg.batch_size
    if not cfg.gradient_checkpointing:
        act *= 2.8
    # LoRA params + Adam moments (fp32 master weights): ~1.5 GB at rank 32
    lora = 1.5 * (cfg.network_dim / 32)
    aux = 1.2  # CUDA/ROCm context, fragmentation
    total = weights + act + lora + aux
    return {"weights_gb": round(weights, 2), "activations_gb": round(act, 2),
            "lora_state_gb": round(lora, 2), "overhead_gb": aux,
            "total_gb": round(total, 2)}


def build_command(cfg: TrainConfig) -> list[str]:
    """Return argv for ``accelerate launch sdxl_train_network.py ...``."""
    argv = [
        cfg.sd_scripts + "/sdxl_train_network.py",
        f"--pretrained_model_name_or_path={cfg.pretrained_model}",
        f"--train_data_dir={cfg.dataset_dir}",
        f"--output_dir={cfg.output_dir}",
        f"--output_name={cfg.output_name}",
        f"--resolution={cfg.resolution}",
        "--network_module=networks.lora",
        f"--network_dim={cfg.network_dim}",
        f"--network_alpha={cfg.network_alpha}",
        f"--learning_rate={cfg.learning_rate}",
        f"--text_encoder_lr={cfg.text_encoder_lr}",
        f"--max_train_steps={cfg.max_train_steps}",
        f"--train_batch_size={cfg.batch_size}",
        f"--gradient_accumulation_steps={cfg.grad_accum}",
        f"--mixed_precision={cfg.mixed_precision}",
        f"--save_every_n_steps={cfg.save_every_n_steps}",
        f"--optimizer_type={cfg.optimizer}",
        f"--seed={cfg.seed}",
        "--network_train_unet_only" if False else "--network_train_unet_only",
        "--no_half_vae",
    ]
    if cfg.bucket:
        argv += [f"--min_bucket_reso={cfg.min_bucket_reso}",
                 f"--max_bucket_reso={cfg.max_bucket_reso}"]
    else:
        argv.append("--no_bucket")
    if cfg.gradient_checkpointing:
        argv.append("--gradient_checkpointing")
    argv += cfg.extra_args
    return argv


def render(cfg: TrainConfig, vram_limit_gb: float = 20.0,
           base_model_gb: float = 6.9) -> str:
    est = estimate_vram_gb(cfg, base_model_gb=base_model_gb)
    problems = cfg.validate()
    fits = est["total_gb"] <= vram_limit_gb
    lines = [
        "# base: {base}".format(base=cfg.pretrained_model),
        "# estimated VRAM: {total_gb} GB / card {limit} GB  ->  {verdict}".format(
            total_gb=est["total_gb"], limit=vram_limit_gb,
            verdict="OK" if fits else "WILL NOT FIT"),
        "#   weights {weights_gb} + activations {activations_gb} + lora {lora_state_gb} "
        "+ overhead {overhead_gb}".format(**est),
    ]
    if problems:
        lines += ["# PROBLEMS:"] + [f"#   - {p}" for p in problems]
    lines.append("accelerate launch " + " ".join(shlex.quote(a) for a in build_command(cfg)))
    return "\n".join(lines)
