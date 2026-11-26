"""CLI: ``python -m forge {check,caption,train}``."""

from __future__ import annotations

import argparse
import json
import sys

from .caption import CaptionPlan, audit_captions, write_sidecars
from .dataset import scan
from .train_cmd import TrainConfig, estimate_vram_gb, render


def cmd_check(args) -> int:
    rep = scan(args.dir, min_side=args.min_side)
    summary = rep.summary()
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"目录          {rep.root}")
        print(f"图片          {summary['images']} 张（可用 {summary['usable']}，坏 {summary['broken']}）")
        print(f"重复          {summary['duplicates']} 张")
        print(f"缺 caption    {summary['missing_captions']} 张")
        print(f"短边最小      {summary['min_side_px']} px")
        print(f"宽高比跨度    {summary['aspect_spread']}")
        print(f"结论          {'可以训' if summary['ok'] else '不能训'}")
        for w in summary["warnings"]:
            print(f"  [警告] {w}")
        for b in summary["blocking"]:
            print(f"  [阻断] {b}")
    return 0 if summary["ok"] else 2


def cmd_caption(args) -> int:
    plan = CaptionPlan(trigger=args.trigger, base=args.base or "")
    result = write_sidecars(args.dir, plan, overwrite=args.overwrite, dry_run=args.dry_run)
    print(f"写入 {result['written']} 条 caption，跳过 {result['skipped']} 条")
    if args.audit:
        print(json.dumps(audit_captions(args.dir, args.trigger), ensure_ascii=False, indent=2))
    return 0


# 常用底座预设：名字 -> (权重文件名, 权重体积 GB)
# 体积用于显存估算——动画底座和写实底座的文件大小差 0.5~1 GB，估错就会误判 OOM。
BASE_PRESETS = {
    "sdxl-base": ("sd_xl_base_1.0.safetensors", 6.9),
    "animagine-4": ("animagine-xl-4.0-opt.safetensors", 6.6),
    "animagine-3.1": ("animagine-xl-3.1.safetensors", 6.6),
    "anything-xl": ("AnythingXL_xl.safetensors", 6.5),
}


def resolve_base(spec: str) -> tuple[str, float]:
    """把 --base 解析成 (权重文件, 体积 GB)。

    既接受预设名（animagine-4），也接受任意 .safetensors 路径；体积未知时按 6.9 GB 估。
    """
    if spec in BASE_PRESETS:
        return BASE_PRESETS[spec]
    return (spec, 6.9)


def cmd_train(args) -> int:
    base_file, base_gb = resolve_base(args.base_model)
    cfg = TrainConfig(
        sd_scripts=args.sd_scripts,
        pretrained_model=base_file,
        dataset_dir=args.dir,
        output_name=args.name,
        output_dir=args.out_dir,
        resolution=args.res,
        network_dim=args.rank,
        network_alpha=max(1, args.rank // 2),
        learning_rate=args.lr,
        max_train_steps=args.steps,
        batch_size=args.batch,
    )
    est = estimate_vram_gb(cfg, base_model_gb=base_gb)
    print(render(cfg, vram_limit_gb=args.vram, base_model_gb=base_gb), flush=True)
    if est["total_gb"] > args.vram:
        print(f"\n[退出码 3] 估算 {est['total_gb']} GB 超过 {args.vram} GB，建议降分辨率或开 gradient checkpointing",
              file=sys.stderr)
        return 3
    return 0


def cmd_train_local(args) -> int:
    from forge import standalone

    return standalone.main(["--data", args.data, "--out", args.out, "--base", args.base,
                            "--res", str(args.res), "--height", str(args.height),
                            "--steps", str(args.steps), "--epochs", str(args.epochs),
                            "--rank", str(args.rank), "--lr", str(args.lr),
                            "--dropout", str(args.dropout), "--tag", args.tag,
                            "--log-every", str(args.log_every)]
                           + (["--train-te"] if args.train_te else [])
                           + (["--dry-run"] if args.dry_run else []))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="forge", description="small-dataset LoRA tooling")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="dataset QC")
    c.add_argument("--dir", required=True)
    c.add_argument("--min-side", type=int, default=512)
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_check)

    p = sub.add_parser("caption", help="write .txt sidecars")
    p.add_argument("--dir", required=True)
    p.add_argument("--trigger", required=True)
    p.add_argument("--base", default="")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--audit", action="store_true")
    p.set_defaults(func=cmd_caption)

    t = sub.add_parser("train", help="print the sd-scripts command + VRAM estimate")
    t.add_argument("--dir", required=True)
    t.add_argument("--name", required=True)
    t.add_argument("--out-dir", default="loras")
    t.add_argument("--base-model", "--base", dest="base_model", default="sdxl-base",
                   help="preset (sdxl-base | animagine-4 | animagine-3.1 | anything-xl) or a .safetensors path")
    t.add_argument("--sd-scripts", default="sd-scripts")
    t.add_argument("--res", type=int, default=1024)
    t.add_argument("--rank", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--steps", type=int, default=1500)
    t.add_argument("--batch", type=int, default=1)
    t.add_argument("--vram", type=float, default=20.0)
    t.set_defaults(func=cmd_train)

    s = sub.add_parser("train-local",
                       help="train here, without sd-scripts (diffusers + peft)")
    s.add_argument("--data", required=True, help="dataset dir: images + matching .txt")
    s.add_argument("--out", required=True, help="output .safetensors")
    s.add_argument("--base", required=True, help="base checkpoint .safetensors")
    s.add_argument("--res", type=int, default=1024, help="width (also the side of a square)")
    s.add_argument("--height", type=int, default=0, help="height for portrait/landscape; 0 = square")
    s.add_argument("--steps", type=int, default=0, help="0 = estimate from image count")
    s.add_argument("--epochs", type=float, default=13.0)
    s.add_argument("--rank", type=int, default=32)
    s.add_argument("--lr", type=float, default=3e-5)
    s.add_argument("--dropout", type=float, default=0.0)
    s.add_argument("--tag", default="mychar")
    s.add_argument("--train-te", action="store_true",
                   help="also train the text encoders (slower, more self-contained trigger)")
    s.add_argument("--log-every", type=int, default=20)
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_train_local)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
