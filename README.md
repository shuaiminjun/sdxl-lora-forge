[![ci](https://github.com/shuaiminjun/sdxl-lora-forge/actions/workflows/ci.yml/badge.svg)](https://github.com/shuaiminjun/sdxl-lora-forge/actions/workflows/ci.yml) [![python](https://img.shields.io/badge/python-3.10%2B-blue)]() [![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

# sdxl-lora-forge

> **Status** — early-stage, actively developed. Feedback, corrections and "you got this wrong" are all welcome: **shuaiminjun541@gmail.com**
>
> Built with heavy AI assistance: the code was largely written by AI, and what I did was decide what to build, run it, measure it, and keep the parts that held up. Nothing here claims state-of-the-art — the numbers come from one 20 GB card, and the versions that failed are written down next to the ones that worked. Corrections are genuinely welcome.

**给 SDXL 系模型训一个小数据集角色 LoRA 的工具箱。** 四件事：

1. `check` —— 这个数据集能不能训（先花 3 秒，别烧 30 分钟）
2. `caption` —— 批量写 caption，触发词放对位置
3. `train` —— 生成 kohya `sd-scripts` 命令 + 估显存
4. `train-local` —— **不装 sd-scripts 也能训**（diffusers + peft，直接导出 kohya 格式权重）

零运行时依赖（图像头解析是自己写的，不需要 Pillow）；只有在真正训练时才需要 torch。

```bash
python -m forge check  --dir data/my-char
python -m forge caption --dir data/my-char --trigger mychar --base "a girl with silver hair and a red scarf"
python -m forge train  --dir data/my-char --name mychar_v1 --res 768 --rank 32 --steps 900
python -m forge train-local --data data/my-char --out mychar_v1.safetensors \
       --base ./models/illustrious-xl-v1.0.safetensors --res 1024 --height 1536 --steps 1600 --rank 32 --tag mychar
```

`train` 不真的训练，它打印命令和显存估算，装不下就以退出码 3 结束：

```
# estimated VRAM: 12.8 GB / card 20.0 GB  ->  OK
#   weights 6.9 + activations 3.2 + lora 1.5 + overhead 1.2
accelerate launch sd-scripts/sdxl_train_network.py --pretrained_model_name_or_path=...
```

---

## 什么时候该用哪个

| 你的情况 | 用 |
|---|---|
| 已经装了 kohya `sd-scripts`，想要一条能跑的命令 | `train` |
| 没装、也不想装 accelerate / bitsandbytes | **`train-local`** |
| 不确定数据集能不能训 | 先 `check` |
| 一堆图没写 caption | `caption` |

两条路产出的 `.safetensors` 都是 kohya 命名（`lora_unet_*` / `lora_te1_*` / `lora_te2_*`），
ComfyUI、A1111、forge 都能直接加载，可以互换。

---

## 为什么"先检查再训练"

我训废了 4 版才拿到一个能用的角色 LoRA，每次 30 分钟。复盘下来**三次里有两次是数据或 caption 的错，不是超参**：

| 版本 | 出图现象 | 真实原因 |
|---|---|---|
| v1 | 一个"蓝色鲸鱼团块" | caption 里姿态写在类别前面，触发词被当成"鲸鱼"这个物体 |
| v2 | 干脆画真鲸鱼 | 我把类别名词删了（错误归因） |
| v3 | 三个不同发色的女孩 | 340 步太少，外形没锁住 |
| **v4** | 外形锁住，但发色漂白 | 底座先验压过 caption → 推理时把发色写进提示词开头 |

所以第一步永远是 `check`：**在花掉 30 分钟之前把明显的问题挑出来。**

---

## 四个命令

### `check` —— 数据集体检

```bash
python -m forge check --dir data/my-char
```

```
图片          24 张（可用 24，坏 0）
重复          0 张
缺 caption    0 张
短边最小      768 px
宽高比跨度    0.0
结论          可以训
```

阻断与警告分开，避免"什么都报错"：

| 检查 | 阈值 | 级别 |
|---|---|---|
| 短边 < 512 px | 超过 20% 的图 | 阻断 |
| caption 缺失 | 超过 50% 的图 | 阻断 |
| 宽高比跨度 > 1.2 | — | 警告 |
| 重复图（sha1 相同） | 任意 | 警告 |
| 图 < 10 张 | — | 警告 |

### `caption` —— 批量写 caption

硬规则只有一条：**触发词第一个，后面紧跟类别名词。**

```bash
python -m forge caption --dir data/my-char --trigger mychar --base "a girl with silver hair and a red scarf"
# → mychar, a girl with silver hair and a red scarf
```

- 触发词校验：小写、字母开头、3~32 字符、不能有空格（有空格会被 tokenizer 拆成已有词，等于没学）
- 自动去掉 `(` `)` `<>{}|\` 这类 tokenizer 处理不好的字符
- 超长按**单词边界**截到 220 字符（SDXL 上限 77 token）
- 默认不覆盖已有 `.txt`（手写的比模板值钱），`--overwrite` 才覆盖
- `--audit` 检查缺触发词、重复 caption、过长

### `train` —— kohya 命令 + 显存估算

```
总显存 = 底模权重 + 激活 + LoRA 状态 + 开销
激活   = 3.2 GB × (分辨率/1024)² × batch     （开 gradient checkpointing）
       × 2.8                                 （关掉时）
```

实测标定（20 GB 卡）：

| 分辨率 | batch | 估算 | 判定 |
|---|---|---|---|
| 768 | 1 | 11.4 GB | OK |
| 768 | 4 | 16.8 GB | OK |
| 1024 | 2 | 16.0 GB | OK |
| **1024** | **4** | **22.4 GB** | **装不下** |

命令里强制带的几个 flag，每个都对应一次真实事故：

| flag | 不加会怎样 |
|---|---|
| `--no_half_vae` | fp16 VAE 出**纯黑图** |
| `--gradient_checkpointing` | 1024² 直接 OOM |
| `--network_train_unet_only` | 文本编码器被训坏，触发词泛化到所有图 |
| `--network_module=networks.lora` | 忘写就变全量微调，显存翻倍 |

### `train-local` —— 不依赖 sd-scripts 的训练器

```bash
# 先看计划，不加载模型（几秒）
python -m forge train-local --data data/my-char --out my.safetensors \
       --base ./models/sd_xl_base_1.0.safetensors --res 1024 --height 1536 --dry-run

# 真训
python -m forge train-local --data data/my-char --out my.safetensors \
       --base ./models/illustrious-xl-v1.0.safetensors \
       --res 1024 --height 1536 --steps 1600 --rank 32 --lr 3e-5 --tag mychar
```

| 参数 | 说明 |
|---|---|
| `--res` / `--height` | 宽 / 高。`--height 0` 是方图；`--height 1536` 是竖图（角色立绘推荐） |
| `--steps` | 0 = 按图片数自动估（120 张 ≈ 1600 步） |
| `--rank` | 32 对角色够用；风格 LoRA 24~32 |
| `--lr` | 3e-5 稳，1e-4 容易过拟合 |
| `--train-te` | 同时训文本编码器：更慢，但触发词更独立 |
| `--dry-run` | 只打印尺寸和步数 |

需要的包（只在训练时）：

```bash
pip install torch diffusers transformers peft safetensors torchvision pillow
```

实测速度（20 GB AMD 卡 / ROCm）：**约 1.1 秒一步**，1024×1536 的 1600 步 ≈ 30 分钟。
`train-local` 训完会在旁边写一个同名 `.json`，记下触发词、底座、步数、尺寸，方便以后对账。

---

## 该知道的坑（都在这张表里）

| 现象 | 原因 |
|---|---|
| 出图是纯黑 | fp16 VAE 溢出（NaN），不是显存不够。加 `--no_half_vae`（kohya）或换 bf16 |
| 角色每次都不一样 | 步数不够，或 caption 没把外形写全 |
| 触发词没效果 | 触发词有空格被拆词了，或者没紧跟类别名词 |
| 眼睛颜色乱变 | caption 里漏写瞳色 |
| 换底模后角色变了 | LoRA 能跨同结构底座加载，但会带新底座的配色/线条先验 |
| 训练到一半 `Can't get available size` | ROCm 上显存被别的进程占着，直接 abort。训练前把出图进程关掉 |

更细的实测结论（数据量、非方图、强度取值、什么时候该放弃训练）见
[`docs/WHAT_WORKS.md`](docs/WHAT_WORKS.md)。

---

## 底座是参数，不是代码

`--base` 接受预设名或任意 `.safetensors` 路径：

| 预设 | 权重 | 体积 |
|---|---|---|
| `sdxl-base`（默认） | `sd_xl_base_1.0.safetensors` | 6.9 GB |
| `animagine-4` | `animagine-xl-4.0-opt.safetensors` | 6.6 GB |
| `animagine-3.1` | `animagine-xl-3.1.safetensors` | 6.6 GB |
| `anything-xl` | `AnythingXL_xl.safetensors` | 6.5 GB |

显存估算里底模体积参与计算，所以换底座会重算；差 0.3~0.5 GB 就足以让"刚好卡 20 GB"的判断反过来。

---

## 测试

```bash
pip install pytest && pytest -q      # 37 passed
```

全部测试**不下载模型、不需要 GPU**：尺寸归桶、步数估算、kohya 键名映射、
PNG/JPEG 头解析、触发词校验、显存估算单调性。

---

## 边界（如实说）

- `train-local` 是**够用**的实现，不是最快：它把图片 latent 预编码后循环采样，没有 dataloader 与分布式。
- 估算是**保守近似**，用途是开跑前排除明显不可行的配置，不是精确预测。
- 不做自动调参。这套工具的价值是让失败变快、变清楚。

## License

MIT
