"""Dataset QC tests — files are synthesised, so no fixtures on disk."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

from forge.dataset import read_size, scan


# --------------------------------------------------------------- helpers
def write_png(path: Path, w: int, h: int, color: int = 0) -> None:
    """Smallest possible valid PNG (no Pillow)."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)  # 8-bit grayscale
    raw = b"".join(b"\x00" + bytes([color]) * w for _ in range(h))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b""))
    path.write_bytes(png)


def write_jpeg_header(path: Path, w: int, h: int) -> None:
    """Only the header — enough for read_size, not a decodable image."""
    sof = b"\xff\xc0" + struct.pack(">H", 17) + b"\x08" + struct.pack(">HH", h, w) + b"\x03" + b"\x00" * 9
    path.write_bytes(b"\xff\xd8" + sof + b"\xff\xd9")


# ----------------------------------------------------------------- tests
def test_read_size_png(tmp_path):
    p = tmp_path / "a.png"
    write_png(p, 640, 512)
    assert read_size(p) == (640, 512, "png")


def test_read_size_jpeg(tmp_path):
    p = tmp_path / "a.jpg"
    write_jpeg_header(p, 800, 600)
    assert read_size(p) == (800, 600, "jpeg")


def test_read_size_rejects_garbage(tmp_path):
    p = tmp_path / "x.png"
    p.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        read_size(p)


def test_scan_ok_dataset(tmp_path):
    for i in range(12):
        write_png(tmp_path / f"img{i:02d}.png", 1024, 1024, color=i * 3)
        (tmp_path / f"img{i:02d}.txt").write_text("tok, a photo", encoding="utf-8")
    rep = scan(tmp_path)
    s = rep.summary()
    assert s["usable"] == 12
    assert s["missing_captions"] == 0
    assert s["ok"] is True


def test_scan_flags_duplicates(tmp_path):
    for i in range(6):
        write_png(tmp_path / f"c{i}.png", 1024, 1024, color=7)  # identical bytes
        (tmp_path / f"c{i}.txt").write_text("tok", encoding="utf-8")
    rep = scan(tmp_path)
    assert rep.summary()["duplicates"] == 5
    assert any("duplicate" in w for w in rep.summary()["warnings"])


def test_scan_blocks_when_images_too_small(tmp_path):
    for i in range(10):
        write_png(tmp_path / f"s{i}.png", 256, 256, color=i)
        (tmp_path / f"s{i}.txt").write_text("tok", encoding="utf-8")
    rep = scan(tmp_path)
    assert rep.summary()["ok"] is False
    assert any("side <" in b for b in rep.summary()["blocking"])


def test_scan_blocks_when_captions_missing(tmp_path):
    for i in range(10):
        write_png(tmp_path / f"m{i}.png", 1024, 1024, color=i)
    (tmp_path / "m0.txt").write_text("tok", encoding="utf-8")
    rep = scan(tmp_path)
    assert rep.summary()["ok"] is False
    assert any("no .txt sidecar" in b for b in rep.summary()["blocking"])


def test_scan_warns_on_few_images(tmp_path):
    for i in range(4):
        write_png(tmp_path / f"f{i}.png", 1024, 1024, color=i)
        (tmp_path / f"f{i}.txt").write_text("tok", encoding="utf-8")
    rep = scan(tmp_path, min_side=64)
    assert rep.summary()["ok"] is True
    assert any("only 4 images" in w for w in rep.summary()["warnings"])


def test_scan_handles_broken_file(tmp_path):
    write_png(tmp_path / "good.png", 1024, 1024)
    (tmp_path / "good.txt").write_text("tok", encoding="utf-8")
    (tmp_path / "bad.png").write_bytes(b"\x89PNG\r\n\x1a\nnope")
    rep = scan(tmp_path, min_side=64)
    assert rep.summary()["broken"] == 1
    assert rep.summary()["usable"] == 1


def test_scan_missing_dir(tmp_path):
    rep = scan(tmp_path / "nope")
    assert rep.summary()["ok"] is False
