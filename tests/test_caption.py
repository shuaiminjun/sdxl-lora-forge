"""Caption rules: trigger first, sanitised, capped, idempotent."""

from __future__ import annotations

import pytest

from forge.caption import CaptionPlan, audit_captions, build_caption, validate_trigger, write_sidecars


def test_trigger_must_be_lowercase_single_token():
    assert validate_trigger("hsmwhale") == "hsmwhale"
    for bad in ("", "HsM", "two words", "with-dash", "x", "a" * 40):
        with pytest.raises(ValueError):
            validate_trigger(bad)


def test_caption_starts_with_trigger_then_base():
    plan = CaptionPlan(trigger="hsmwhale", base="a chibi anime girl")
    cap = build_caption(plan, "001.png")
    assert cap == "hsmwhale, a chibi anime girl"


def test_per_image_note_is_appended():
    plan = CaptionPlan(trigger="tok", base="a photo", per_image={"002": "at the beach"})
    assert build_caption(plan, "002.png").endswith("at the beach")


def test_parentheses_are_stripped_because_they_are_weight_syntax():
    plan = CaptionPlan(trigger="tok", base="(masterpiece), a (cat)")
    cap = build_caption(plan, "x.png")
    assert "(" not in cap and ")" not in cap


def test_long_caption_is_truncated_on_a_word_boundary():
    plan = CaptionPlan(trigger="tok", base="word " * 100, max_chars=60)
    cap = build_caption(plan, "x.png")
    assert len(cap) <= 60
    assert not cap.endswith(" ")
    assert cap.startswith("tok, word")


def test_write_sidecars_does_not_overwrite_by_default(tmp_path):
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "a.txt").write_text("hand written", encoding="utf-8")
    plan = CaptionPlan(trigger="tok", base="a photo")
    result = write_sidecars(tmp_path, plan)
    assert result == {"written": 0, "skipped": 1}
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "hand written"

    result = write_sidecars(tmp_path, plan, overwrite=True)
    assert result["written"] == 1
    assert (tmp_path / "a.txt").read_text(encoding="utf-8").startswith("tok")


def test_dry_run_writes_nothing(tmp_path):
    (tmp_path / "a.png").write_bytes(b"x")
    result = write_sidecars(tmp_path, CaptionPlan(trigger="tok"), dry_run=True)
    assert result["written"] == 1
    assert not (tmp_path / "a.txt").exists()


def test_audit_finds_missing_trigger_and_duplicates(tmp_path):
    (tmp_path / "a.txt").write_text("other, a cat", encoding="utf-8")
    (tmp_path / "b.txt").write_text("tok, same", encoding="utf-8")
    (tmp_path / "c.txt").write_text("tok, same", encoding="utf-8")
    rep = audit_captions(tmp_path, "tok")
    assert rep["captions"] == 3
    assert rep["missing_trigger"] == ["a.txt"]
    assert rep["duplicate_groups"] == 1
