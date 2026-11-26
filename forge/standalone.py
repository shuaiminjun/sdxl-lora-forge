"""不依赖 sd-scripts 的自带训练器：diffusers + peft 直接训 SDXL LoRA。

为什么仓库里要放第二套实现：`train` 子命令只负责"检查 + 拼 kohya 命令"，
它假定你装了 sd-scripts、accelerate、bitsandbytes 那一整套。实际用下来，
在 ROCm + 20 GB 的卡上装那一套很折腾，而**训练一个 60~120 张的小数据集，
diffusers 加 peft 就够**：不用 accelerate、不用 bitsandbytes、不用装 kohya。

导出的权重是 kohya 命名（``lora_unet_*`` / ``lora_te1_*`` / ``lora_te2_*``），
ComfyUI / A1111 / forge 都能直接加载，所以"自带训练器"和"交给 sd-scripts"
两条路的产物可以互换。

用法见 README 的"自带训练器"一节。想跑 CUDA 以外的平台（ROCm / MPS）也可以，
它只用 diffusers 的标准组件，没有平台特定代码。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

# 纯计算部分（无 torch 依赖），供 CI 与 --dry-run 使用


def bucket_size(width: int, height: int, res: int, height_res: int | None = None) -> tuple[int, int]:
    """把 (宽, 高) 归一到"短边 = res、长边按比例"且 8 的倍数。

    为什么要归一到 8 的倍数：VAE 下采样 8 倍，非 8 倍数的尺寸会在 latent 里
    引入半像素偏移，训练时表现为"整体轻微错位"。

    传入 ``height_res`` 时按"宽 = res、高 = height_res"精确取（非方形数据集）。
    """
    if height_res:
        w = max(8, int(round(res / 8)) * 8)
        h = max(8, int(round(height_res / 8)) * 8)
        return w, h
    aspect = height / width
    if aspect >= 1:
        return max(8, int(round(res / 8)) * 8), max(8, int(round(res * aspect / 8)) * 8)
    return max(8, int(round(res / aspect / 8)) * 8), max(8, int(round(res / 8)) * 8)


def estimate_steps(image_count: int, epochs: float = 10, batch: int = 1) -> int:
    """按图片数与轮数估步数。

    经验：小数据集（60~120 张）跑 **10 轮左右**就能锁住外形；
    60 张 × 10 轮 = 600 步偏少（实测外形不稳），所以默认给 1200 步；
    120 张 × 13 轮 = 1600 步更稳。
    """
    return max(1, int(round(image_count * epochs / max(1, batch))))


def kohya_prefix(kind: str) -> str:
    """peft 子模块 → kohya 键名前缀。"""
    return {"unet": "lora_unet_", "te1": "lora_te1_", "te2": "lora_te2_"}.get(kind, "lora_unet_")


def peft_key_to_kohya(key: str, kind: str = "unet") -> tuple[str, str] | None:
    """``...to_q.lora_A.weight`` → ``('lora_unet_...to_q.lora_down.weight', 'down')``。

    返回 None 表示这个 key 不是 LoRA 权重（比如原始 UNet 参数），调用方跳过即可。
    """
    k = key.removeprefix("module.")   # DistributedDataParallel 包装前缀
    for suf, new, side in ((".lora_A.", ".lora_down.weight", "down"),
                           (".lora_B.", ".lora_up.weight", "up")):
        if suf in k and k.endswith(".weight"):
            path = k.split(suf)[0].replace(".", "_")
            return f"{kohya_prefix(kind)}{path}{new}", side
    return None


# ---------------------------------------------------------------- 数据集


def load_items(data_dir: Path) -> list[dict]:
    """读 ``<name>.png`` + 同名 ``.txt``，返回 [{image, caption}]。

    没有 .txt 的图会被跳过并打印提示：没有 caption 的图在 LoRA 训练里
    只会让模型记住"这张图的样子"，学不到"这个词对应什么"。
    """
    items, skipped = [], 0
    for img in sorted(data_dir.iterdir()):
        if img.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        cap = img.with_suffix(".txt")
        if not cap.is_file():
            skipped += 1
            continue
        text = cap.read_text(encoding="utf-8").strip()
        if not text:
            skipped += 1
            continue
        items.append({"image": str(img), "caption": text})
    if skipped:
        print(f"[warn] 跳过 {skipped} 张没有 caption 的图", file=sys.stderr)
    return items


# ---------------------------------------------------------------- 训练


def _lora_config(rank: int, alpha: int, dropout: float, text_side: bool):
    from peft import LoraConfig

    targets = (["q_proj", "k_proj", "v_proj", "out_proj"] if text_side
               else ["to_q", "to_k", "to_v", "to_out.0"])
    return LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                      init_lora_weights="gaussian", target_modules=targets)


def train(args) -> int:
    import torch
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline
    from safetensors.torch import save_file
    from torchvision import transforms

    device, dtype = "cuda" if torch.cuda.is_available() else "cpu", torch.float16
    data_dir, out_path = Path(args.data), Path(args.out)
    items = load_items(data_dir)
    if not items:
        print("数据集里没有带 caption 的图", file=sys.stderr)
        return 2

    width, height = bucket_size(
        args.res, args.height or args.res, args.res, args.height)
    print(f"[data] {len(items)} 张 → 训练尺寸 {width}×{height}")

    pipe = StableDiffusionXLPipeline.from_single_file(
        args.base, torch_dtype=dtype, use_safetensors=True).to(device)
    unet, vae, te1, te2 = pipe.unet, pipe.vae, pipe.text_encoder, pipe.text_encoder_2
    tokenizers = (pipe.tokenizer, pipe.tokenizer_2)
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    te1.requires_grad_(False)
    te2.requires_grad_(False)
    unet.add_adapter(_lora_config(args.rank, args.rank // 2, args.dropout, text_side=False))
    if args.train_te:
        te1.add_adapter(_lora_config(args.rank, args.rank // 2, args.dropout, text_side=True))
        te2.add_adapter(_lora_config(args.rank, args.rank // 2, args.dropout, text_side=True))

    params = [p for p in unet.parameters() if p.requires_grad]
    if args.train_te:
        params += [p for te in (te1, te2) for p in te.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)

    tf = transforms.Compose([
        transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((height, width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def encode_image(path: str):
        from PIL import Image
        px = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device=device, dtype=dtype)
        return vae.encode(px).latent_dist.sample() * vae.config.scaling_factor

    def encode_prompt(text: str):
        out = []
        for te, tok in ((te1, tokenizers[0]), (te2, tokenizers[1])):
            ids = tok(text, padding="max_length", max_length=tok.model_max_length,
                      truncation=True, return_tensors="pt").input_ids.to(device)
            out.append(te(ids, output_hidden_states=True).hidden_states[-2])
        emb = torch.cat(out, dim=-1)
        return emb

    print("[cache] 预编码图片 latent 与文本 embedding ...")
    t0 = time.time()
    cached = [{"latent": encode_image(it["image"]), "emb": encode_prompt(it["caption"])}
              for it in items]
    print(f"[cache] 完成，用时 {time.time() - t0:.0f}s")

    noise_sched = DDPMScheduler.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
    add_time_ids = torch.tensor([[height, width, 0, 0, height, width]], device=device, dtype=dtype)

    steps = args.steps or estimate_steps(len(items), args.epochs)
    print(f"[train] {steps} 步，rank {args.rank}，lr {args.lr}")
    t0, running = time.time(), 0.0
    unet.train()
    for step in range(1, steps + 1):
        batch = cached[(step - 1) % len(cached)]
        noise = torch.randn_like(batch["latent"])
        t = torch.randint(0, 1000, (1,), device=device).long()
        noisy = noise_sched.add_noise(batch["latent"], noise, t)
        pred = unet(noisy, t, encoder_hidden_states=batch["emb"],
                    added_cond_kwargs={"text_embeds": batch["emb"].mean(1, keepdim=True).repeat(
                        1, 1280), "time_ids": add_time_ids}).sample
        loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running = running * 0.9 + loss.item() * 0.1
        if step % args.log_every == 0 or step == steps:
            per = (time.time() - t0) / args.log_every
            print(f"  step {step}/{steps} loss {running:.4f} {per:.2f}s/step", flush=True)
            t0 = time.time()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    for obj, kind in ([(unet, "unet")]
                      + ([(te1, "te1"), (te2, "te2")] if args.train_te else [])):
        for key, val in obj.state_dict().items():
            mapped = peft_key_to_kohya(key, kind)
            if not mapped:
                continue
            name, _side = mapped
            state[name] = val.contiguous()
            state[name.rsplit(".", 2)[0] + ".alpha"] = torch.tensor(float(args.rank // 2))
    save_file(state, str(out_path), metadata={"tag": args.tag, "rank": str(args.rank)})
    print(f"[done] {out_path}（{len(state)} 个张量）用时 {(time.time() - t0):.0f}s")
    (out_path.with_suffix(".json")).write_text(
        json.dumps({"tag": args.tag, "base": str(args.base), "steps": steps,
                    "rank": args.rank, "lr": args.lr, "size": [width, height]},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="forge train-local",
                                 description="不依赖 sd-scripts 的 SDXL LoRA 训练器")
    ap.add_argument("--data", required=True, help="数据集目录（图 + 同名 .txt）")
    ap.add_argument("--out", required=True, help="输出 .safetensors")
    ap.add_argument("--base", required=True, help="底模 .safetensors")
    ap.add_argument("--res", type=int, default=1024, help="宽（也用于方图的边长）")
    ap.add_argument("--height", type=int, default=0, help="非方图时的高；0 = 方图")
    ap.add_argument("--steps", type=int, default=0, help="0 = 按图片数与 epoch 自动估")
    ap.add_argument("--epochs", type=float, default=13.0)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--tag", default="mychar", help="触发词，只写进元数据备查")
    ap.add_argument("--train-te", action="store_true", help="同时训文本编码器（更慢，触发词更独立）")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不加载模型")
    a = ap.parse_args(argv)

    width, height = bucket_size(a.res, a.height or a.res, a.res, a.height or None)
    if a.dry_run:
        items = load_items(Path(a.data))
        steps = a.steps or estimate_steps(len(items), a.epochs)
        print(f"图片 {len(items)} 张 | 尺寸 {width}×{height} | rank {a.rank} | {steps} 步")
        return 0
    return train(a)


if __name__ == "__main__":
    sys.exit(main())
