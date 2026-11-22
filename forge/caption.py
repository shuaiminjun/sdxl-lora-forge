"""Caption synthesis for LoRA datasets.

The one rule that matters: **the trigger token must be the first token of every
caption**, and it must be a token the tokenizer does not already split into
common words (otherwise the model learns nothing new). Everything after that is
a short, factual description — not prose.

This module is deliberately dumb: a template plus optional per-image notes.
VLM captioning is a plug-in (``--notes-from``), not a dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

MAX_TOKENS_APPROX = 60  # SDXL CLIP text encoder truncates at 77 tokens


@dataclass
class CaptionPlan:
    trigger: str
    base: str = ""
    per_image: dict[str, str] | None = None
    max_chars: int = 220


def validate_trigger(trigger: str) -> str:
    """A trigger should be one lowercase token, no spaces, no punctuation."""
    t = trigger.strip()
    if not t:
        raise ValueError("trigger must not be empty")
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,31}", t):
        raise ValueError(
            "trigger must be lowercase letters/digits/underscore, 3..32 chars, "
            f"start with a letter (got {trigger!r})")
    return t


def sanitize(text: str) -> str:
    """Collapse whitespace, drop characters CLIP handles badly, cap length."""
    t = re.sub(r"\s+", " ", text).strip()
    t = t.replace("(", " ").replace(")", " ")  # parentheses are weight syntax
    t = re.sub(r"[<>{}|\\]", "", t)
    return t


def build_caption(plan: CaptionPlan, image_name: str) -> str:
    trigger = validate_trigger(plan.trigger)
    parts = [trigger]
    if plan.base:
        parts.append(sanitize(plan.base))
    extra = ""
    if plan.per_image:
        extra = sanitize(plan.per_image.get(Path(image_name).stem, ""))
    if extra:
        parts.append(extra)
    caption = ", ".join(p for p in parts if p)
    if len(caption) > plan.max_chars:
        caption = caption[: plan.max_chars].rsplit(" ", 1)[0]
    return caption


def write_sidecars(root: str | Path, plan: CaptionPlan, *, overwrite: bool = False,
                   dry_run: bool = False) -> dict[str, int]:
    """Write ``<image>.txt`` next to each image. Returns counts."""
    root = Path(root)
    written = skipped = 0
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        sidecar = p.with_suffix(".txt")
        if sidecar.exists() and not overwrite:
            skipped += 1
            continue
        caption = build_caption(plan, p.name)
        if not dry_run:
            sidecar.write_text(caption, encoding="utf-8")
        written += 1
    return {"written": written, "skipped": skipped}


def audit_captions(root: str | Path, trigger: str) -> dict:
    """Find captions that will hurt: missing trigger, duplicated, too long."""
    root = Path(root)
    total = 0
    missing_trigger: list[str] = []
    too_long: list[str] = []
    duplicates: dict[str, list[str]] = {}
    for p in sorted(root.rglob("*.txt")):
        text = p.read_text(encoding="utf-8").strip()
        total += 1
        if not text.lower().startswith(trigger):
            missing_trigger.append(p.name)
        if len(text) > 220:
            too_long.append(p.name)
        duplicates.setdefault(text, []).append(p.name)
    dup_groups = {k: v for k, v in duplicates.items() if len(v) > 1}
    return {
        "captions": total,
        "missing_trigger": missing_trigger[:20],
        "too_long": too_long[:20],
        "duplicate_groups": len(dup_groups),
    }
