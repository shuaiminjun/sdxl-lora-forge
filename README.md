[![ci](https://github.com/shuaiminjun/sdxl-lora-forge/actions/workflows/ci.yml/badge.svg)](https://github.com/shuaiminjun/sdxl-lora-forge/actions/workflows/ci.yml) [![python](https://img.shields.io/badge/python-3.10%2B-blue)]() [![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

# sdxl-lora-forge

> **Status** — early-stage, actively developed. Feedback, corrections and "you got this wrong" are all welcome: **shuaiminjun541@gmail.com**
>
> Built with heavy AI assistance. Nothing here is a claim of state-of-the-art results: the numbers in this README are what I measured on one 20 GB card, and the failures are documented alongside them.


**小数据集 LoRA 的"先检查、再训练"工具链**：数据集质检、caption 合成、sd-scripts 命令构建、显存可行性估算。零依赖（图像头解析自己写的，不需要 Pillow）。

```bash
python -m forge check --dir data/lora/whale           # 这个文件夹能不能训
python -m forge caption --dir data/lora/whale --trigger hsmwhale --base "a chibi anime girl"
python -m forge train --dir data/lora/whale --name whale_v4 --res 768 --rank 32 --steps 900
```

最后一条不会真的训练，它**打印命令 + 显存估算**，装不下就直接以退出码 3 结束：

```
# estimated VRAM: 12.8 GB / card 20.0 GB  ->  OK
#   weights 6.9 + activations 3.2 + lora 1.5 + overhead 1.2
accelerate launch sd-scripts/sdxl_train_network.py --pretrained_model_name_or_path=... 
```

---

## 为什么要先"检查"再"训练"

我训废了 4 版才得到一个可用角色 LoRA，每次 30 分钟，复盘下来**三次里有两次是数据或 caption 的问题，不是超参**：

| 版本 | 现象 | 真实原因 |
|---|---|---|
| v1 | 出图是"蓝色鲸鱼团块" | caption 里姿态写在类别前面，触发词被模型当成"鲸鱼"这个物体 |
| v2 | 干脆画真鲸鱼 | 我把类别名词从 caption 里去掉了（错误归因） |
| v3 | 三个不同发色的女孩 | 340 步太少，外形没锁住 |
| **v4** | 外形锁住，发色漂白 | SDXL 的 "chibi 动画"先验压过 caption 里的 navy-teal → **推理时把发色写进提示词开头** |

所以工具的第一职责是**在花掉 30 分钟之前就告诉你哪里错了**。

---

## 三个命令

### `check` — 数据集质检

```bash
python -m forge check --dir data/lora/whale
```

```
图片          24 张（可用 24，坏 0）
重复          0 张
缺 caption    0 张
短边最小      768 px
宽高比跨度    0.0
结论          可以训
```

判定规则（阻断 / 警告分开，避免"什么都报错"）：

| 检查 | 阈值 | 级别 | 为什么 |
|---|---|---|---|
| 短边 < 512 px | 超过 20% 的图 | **阻断** | latent 会糊成一团 |
| caption 缺失 | 超过 50% 的图 | **阻断** | 模型会只学 token，不学内容 |
| 宽高比跨度 > 1.2 | — | 警告 | bucketing 会裁得不一致 |
| 重复图（sha1 相同） | 任意 | 警告 | 重复即加权，会让训练偏向 |
| 图 < 10 张 | — | 警告 | 会过拟合记忆而非泛化 |

**实现细节**：`read_size()` 直接解析 PNG IHDR / JPEG SOF / WebP VP8X 头，不解码像素——所以 24 张 1024² 图的质检是毫秒级，且在没有 Pillow 的机器上也能跑。

### `caption` — caption 合成

规则只有一条硬的：**触发词必须是第一个 token，且后面紧跟类别名词**。

```bash
python -m forge caption --dir data/lora/whale --trigger hsmwhale --base "a chibi anime girl"
# hsmwhale, a chibi anime girl
```

- 触发词校验：小写、字母开头、3~32 字符、无空格（否则 tokenizer 会把它拆成已有词，等于没学）
- 自动去掉 `(` `)`（那是权重语法）、`<>{}|\`（CLIP 处理不好）
- 超长按**单词边界**截断到 220 字符（SDXL 是 77 token 上限）
- 默认**不覆盖**已有 `.txt`（手写的 caption 比模板值钱），`--overwrite` 才覆盖
- `--audit` 检查：缺触发词、重复 caption、过长

### `train` — 命令 + 显存估算

估算公式（在 20 GB ROCm 卡上实测标定，故意偏保守）：

```
总显存 = 底模权重 + 激活 + LoRA 状态 + 开销
激活   = 3.2 GB × (分辨率/1024)² × batch   （开 gradient checkpointing）
        ×2.8  （关掉时）
```

实测结果：

| 分辨率 | batch | 估算 | 判定 |
|---|---|---|---|
| 768 | 1 | 11.4 GB | OK |
| 768 | 4 | 16.8 GB | OK |
| 1024 | 2 | 16.0 GB | OK |
| **1024** | **4** | **22.4 GB** | **装不下**（20 GB 卡） |

命令里强制带的几个 flag，每个都对应一次真实事故：

| flag | 不加会怎样 |
|---|---|
| `--no_half_vae` | fp16 VAE 出**纯黑图** |
| `--gradient_checkpointing` | 1024² 直接 OOM |
| `--network_train_unet_only` | 文本编码器被训坏，触发词泛化到所有图 |
| `--network_module=networks.lora` | 忘了就是全量微调，显存直接翻倍 |

---

## 测试（25 passed）

```bash
pip install pytest && pytest -q
```

- `check`：PNG/JPEG 头解析、坏文件、重复检测、小图阻断、缺 caption 阻断、目录不存在
- `caption`：触发词校验（含 6 种非法）、括号剥离、按词截断、不覆盖、dry-run、审计
- `train`：关键 flag 存在、bucket 开关、5 类配置错误、估算随分辨率/batch 单调、关 checkpointing 至少多花 5 GB、装不下时 `WILL NOT FIT`

所有测试**不下载模型、不需要 GPU**。

---

## 目录

```
forge/
  dataset.py    # 图像头解析 + 质检（阻断/警告分级）
  caption.py    # 触发词校验、caption 合成、审计
  train_cmd.py  # sd-scripts 命令构建 + 显存估算
  cli.py        # check / caption / train
tests/
```

---

## 边界（如实说）

- **不做训练本身**：训练交给 kohya `sd-scripts`，这个仓库只负责"训练前别犯错"和"命令别写错"。
- **估算是保守近似**，不是精确实测；它的用途是**在开跑前排除明显不可行的配置**。
- **不做自动调参**。这套工具的价值是让失败变快、变清楚，不是替你找最优超参。

## License

MIT

## 换底座（同一个 LoRA 工具链，换 base 就行）

底座是**参数**，不是代码。`train` 的 `--base` / `--base-model` 接受预设名或任意 `.safetensors` 路径：

| 预设 | 权重 | 体积 |
|---|---|---|
| `sdxl-base`（默认） | `sd_xl_base_1.0.safetensors` | 6.9 GB |
| `animagine-4` | `animagine-xl-4.0-opt.safetensors` | 6.6 GB |
| `animagine-3.1` | `animagine-xl-3.1.safetensors` | 6.6 GB |
| `anything-xl` | `AnythingXL_xl.safetensors` | 6.5 GB |

```bash
# 写实底座
python -m forge train --dir ./data --name mysdxl --base sdxl-base
# 动画底座（同一个数据集，显存估算会按 6.6 GB 重算）
python -m forge train --dir ./data --name myanime --base animagine-4
# 自己的底座
python -m forge train --dir ./data --name myown --base ./models/my-anime-xl.safetensors
```

**为什么体积要参与估算**：显存估算是 `权重 + 激活 + LoRA 状态 + 余量`，换底座只影响第一项。
文件大小差 0.3~0.5 GB 就足以让"刚好卡 20 GB"的判断反过来，所以预设里带着体积，
`render()` 也会把实际用的底座名打在第一行。

**LoRA 能不能跨底座用**：结构相同的 SDXL 系底座（SDXL base / AnimagineXL / AnythingXL）
之间 LoRA 可以直接互相加载，但**效果会变**——在 A 底训的角色放到 B 底，线条/配色会带 B 的
先验（实测：SDXL base 训的动画角色放到动画底座上，画质明显变好，发色也更稳）。
