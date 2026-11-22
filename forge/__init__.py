"""sdxl-lora-forge: dataset QC, caption synthesis, sd-scripts command builder."""

from .caption import CaptionPlan, audit_captions, build_caption, validate_trigger, write_sidecars
from .dataset import DatasetReport, ImageInfo, read_size, scan
from .train_cmd import TrainConfig, build_command, estimate_vram_gb, render

__all__ = [
    "CaptionPlan", "build_caption", "write_sidecars", "audit_captions", "validate_trigger",
    "scan", "read_size", "DatasetReport", "ImageInfo",
    "TrainConfig", "build_command", "estimate_vram_gb", "render",
]
__version__ = "0.1.0"
