"""Dataset QC: what is actually in the folder, and is it trainable.

Why this exists
---------------
Most failed LoRA runs are not a hyper-parameter problem — the dataset was wrong:
duplicated frames, one subject at 3 aspect ratios, images so small that the
latent is mush, or 90% of the captions missing. Catching that in two seconds is
worth more than any learning-rate sweep.

Everything here works on image *headers* (PNG / JPEG / WebP), so it runs without
Pillow. That keeps the checker usable in CI and on a machine without the imaging
stack installed.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class ImageInfo:
    path: Path
    width: int = 0
    height: int = 0
    fmt: str = ""
    bytes: int = 0
    sha1: str = ""
    error: str = ""

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect(self) -> float:
        return (self.width / self.height) if self.height else 0.0


@dataclass
class DatasetReport:
    root: Path
    images: list[ImageInfo] = field(default_factory=list)
    duplicates: dict[str, list[Path]] = field(default_factory=dict)
    missing_captions: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    blocking: list[str] = field(default_factory=list)

    @property
    def usable(self) -> list[ImageInfo]:
        return [i for i in self.images if not i.error]

    @property
    def ok(self) -> bool:
        return not self.blocking

    def summary(self) -> dict:
        good = self.usable
        sizes = sorted(i.area for i in good)
        aspects = sorted(round(i.aspect, 3) for i in good)
        return {
            "images": len(self.images),
            "usable": len(good),
            "broken": len(self.images) - len(good),
            "duplicates": sum(len(v) - 1 for v in self.duplicates.values()),
            "missing_captions": len(self.missing_captions),
            "min_side_px": min((min(i.width, i.height) for i in good), default=0),
            "median_area": sizes[len(sizes) // 2] if sizes else 0,
            "aspect_spread": round(aspects[-1] - aspects[0], 3) if aspects else 0.0,
            "ok": self.ok,
            "blocking": self.blocking,
            "warnings": self.warnings,
        }


def read_size(path: Path) -> tuple[int, int, str]:
    """Return (width, height, format) from the file header. Raises ValueError."""
    with path.open("rb") as f:
        head = f.read(32)
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            f.seek(16)
            w, h = struct.unpack(">II", f.read(8))
            return w, h, "png"
        if head[:2] == b"\xff\xd8":
            f.seek(2)
            while True:
                marker = f.read(2)
                if len(marker) < 2:
                    break
                if marker[0] != 0xFF:
                    continue
                kind = marker[1]
                if kind in (0xD8, 0xD9):
                    continue
                seg_len_raw = f.read(2)
                if len(seg_len_raw) < 2:
                    break
                seg_len = struct.unpack(">H", seg_len_raw)[0]
                if kind in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                            0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    data = f.read(5)
                    if len(data) < 5:
                        break
                    h, w = struct.unpack(">HH", data[1:5])
                    return w, h, "jpeg"
                f.seek(seg_len - 2, 1)
            raise ValueError("jpeg: no SOF marker found")
        if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
            # VP8X (extended) carries canvas size at offset 24
            chunk = head[12:16]
            if chunk == b"VP8X":
                f.seek(24)
                b = f.read(6)
                w = 1 + (b[0] | (b[1] << 8) | (b[2] << 16))
                h = 1 + (b[3] | (b[4] << 8) | (b[5] << 16))
                return w, h, "webp"
            if chunk == b"VP8 ":
                f.seek(26)
                b = f.read(4)
                w = struct.unpack("<H", b[0:2])[0] & 0x3FFF
                h = struct.unpack("<H", b[2:4])[0] & 0x3FFF
                return w, h, "webp"
            raise ValueError("webp: unsupported chunk")
    raise ValueError("unrecognised image header")


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scan(root: str | Path, *, min_side: int = 512, max_aspect_spread: float = 1.2,
         caption_ext: str = ".txt", hash_files: bool = True) -> DatasetReport:
    """Inspect a training folder and decide whether it can be trained.

    Blocking problems stop a run; warnings do not.
    """
    root = Path(root)
    rep = DatasetReport(root=root)
    if not root.is_dir():
        rep.blocking.append(f"{root} is not a directory")
        return rep

    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMAGE_SUFFIXES or not p.is_file():
            continue
        info = ImageInfo(path=p, bytes=p.stat().st_size)
        try:
            info.width, info.height, info.fmt = read_size(p)
        except Exception as exc:  # noqa: BLE001
            info.error = f"{type(exc).__name__}: {exc}"
        if hash_files and not info.error:
            info.sha1 = _sha1(p)
            rep.duplicates.setdefault(info.sha1, []).append(p)
        if p.with_suffix(caption_ext).exists() is False:
            rep.missing_captions.append(p)
        rep.images.append(info)

    dupes = {k: v for k, v in rep.duplicates.items() if len(v) > 1}
    rep.duplicates = dupes

    good = rep.usable
    if not good:
        rep.blocking.append("no readable images")
        return rep
    small = [i for i in good if min(i.width, i.height) < min_side]
    if len(small) > len(good) * 0.2:
        rep.blocking.append(
            f"{len(small)}/{len(good)} images have a side < {min_side}px — latent will be mush")
    elif small:
        rep.warnings.append(f"{len(small)} images below {min_side}px on the short side")

    aspects = sorted(i.aspect for i in good)
    spread = aspects[-1] - aspects[0] if aspects else 0.0
    if spread > max_aspect_spread:
        rep.warnings.append(
            f"aspect spread {spread:.2f} is wide ({aspects[0]:.2f}..{aspects[-1]:.2f}); "
            "bucketing will crop inconsistently")

    if dupes:
        rep.warnings.append(f"{sum(len(v)-1 for v in dupes.values())} duplicate images (same sha1)")
    if rep.missing_captions:
        ratio = len(rep.missing_captions) / len(rep.images)
        msg = f"{len(rep.missing_captions)}/{len(rep.images)} images have no {caption_ext} sidecar"
        (rep.blocking if ratio > 0.5 else rep.warnings).append(msg)
    if len(good) < 10:
        rep.warnings.append(f"only {len(good)} images — expect the LoRA to memorise rather than generalise")
    return rep
