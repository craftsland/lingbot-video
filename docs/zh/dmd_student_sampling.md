# 少步(DMD Student)采样

DMD 蒸馏的 student 权重用 8 步生成完整视频（而非 40 步）。蒸馏模型必须用**训练时的
步进几何**采样；标准 `FlowUniPCMultistepScheduler` 不是这套几何，会让 student 出片
质量下降。请改用 `DMDStudentScheduler`。

## 模型权重

`MODEL_DIR` 应指向下载到本地的完整模型目录，包含 `model_index.json`、`transformer/`、
`text_encoder/`、`processor/`、`vae/`、`scheduler/`；其中 transformer 必须是蒸馏 student。

可直接使用 `scripts/single-gpu/run_moe_dmd_t2v.sh` 和
`scripts/single-gpu/run_moe_dmd_ti2v.sh`，先设置 `MODEL_DIR`。
两个脚本复用现有推理入口，固定 8 步配方，默认使用包内的 5 秒结构化 prompt 和配套首帧。

## 配方

| 参数 | 值 | 原因 |
|---|---|---|
| Scheduler | `DMDStudentScheduler` | warped sigma 0.5 以上 DDIM(eta=1)，以下 Euler ODE —— 训练时的几何 |
| `num_inference_steps` | 8 | 蒸馏步数 |
| `guidance_scale` | **1.0** | teacher CFG 已蒸入 student，再加引导等于双重引导 |
| `shift` | 3.0 | 必须与蒸馏时的 sigma warp 一致 |
| 分辨率/帧数 | 480×832,121 帧 @ 24 fps | 训练分布 |

## 用法

```python
import torch
from lingbot_video import (
    DMDStudentScheduler,
    LingBotVideoPipeline,             # t2v
    LingBotVideoImageToVideoPipeline, # ti2v
)

pipe = LingBotVideoPipeline.from_pretrained(model_root, torch_dtype=torch.bfloat16).to("cuda")
pipe.scheduler = DMDStudentScheduler()  # 直接替换

frames = pipe(
    prompt=structured_caption,  # 结构化 JSON caption（见 Prompt 准备）
    num_inference_steps=8,
    guidance_scale=1.0,
    shift=3.0,
    height=480, width=832, num_frames=121,
    generator=torch.Generator("cpu").manual_seed(42),
    output_type="np",
).frames
```

t2v 和 i2v（首帧条件）都支持:i2v pipeline 的每步条件帧回贴机制与本 scheduler
直接组合，无需改动。

也可以走 CLI —— 与普通推理完全一致，只需加上 `--scheduler dmd_student`。
两个引擎都支持：`--backend diffusers` 与 `--backend sglang`（sglang 引擎走同一条
pipeline 去噪循环，只是内核换成 sglang 加速版，采样几何完全一致）。ti2v 场景改用
`--mode ti2v --image <首帧图>`：

```bash
python scripts/inference.py \
  --backend diffusers \
  --model_dir "$MODEL_DIR" \
  --mode t2v \
  --prompt_json prompt.json \
  --output outputs/student.mp4 \
  --height 480 --width 832 --num_frames 121 --fps 24 \
  --scheduler dmd_student \
  --steps 8 --guidance_scale 1 --shift 3 \
  --seed 42
```

refiner（若启用）仍用自己的 `FlowUniPCMultistepScheduler`;该开关只改 base 模型采样器。

## 已验证的对齐性

本 scheduler 对训练时采样几何的复现是**逐位(bit-exact)**级别的,t2v 与
ti2v（首帧条件）双双通过验证：注入相同噪声后，文本嵌入、全部 8 步模型前向、每一步
scheduler 转移、终点 latent 误差全为零。复现逐位一致需要两个设置:

```bash
export DIFFUSERS_ATTN_BACKEND=_flash_3   # flash-attn-3 内核
# 同时在推理进程中关闭 matmul 和 cuDNN TF32：
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False
# torch.set_float32_matmul_precision("highest")
```

默认 attention 后端不属于逐位复现设置。不同 attention 内核或 TF32 设置可能通过
MoE 路由及数值差异改变生成结果。

当前 CLI 的 `--no-allow_tf32` 不会复位进程中已经开启的 TF32，运行日志也只记录 matmul
状态；严格复现不能仅依赖该开关，应记录 matmul 和 cuDNN 两项状态。例如，环境变量
`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` 会在进程启动时开启 matmul TF32。
quickstart 示例用于正常生成视频，不承诺逐位复现。

## 注意

- 随机步会消耗传给 pipeline 的 `generator`，需要复现请固定 seed。
- prompt 必须是结构化 JSON caption（模型母语）;纯散文会导致首帧降质。
  参见 [Prompt Preparation](prompt_preparation.md)。
- teacher / 未蒸馏权重仍用 `FlowUniPCMultistepScheduler`(40 步，guidance 3.0)。
- 除非权重蒸馏时用了不同值，否则不要改动 `high_noise_threshold` / `ddim_eta`
  的默认值(0.5 / 1.0)。
