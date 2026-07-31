# DiT 推理

这页详细介绍 diffusers backend 推理：

```bash
python scripts/inference.py --backend diffusers ...
```

runner 支持 `--mode t2i`、`--mode t2v` 和 `--mode ti2v`。

**⚠️ 注意：DiT 推理不接受原始自然语言 prompt，必须先用 [Rewriter](prompt_preparation.md) 生成结构化 JSON Prompt，再通过 `--prompt_json` 传入。**

## 配置模型路径

```bash
export MODEL_DIR="<path_to_lingbot-video-model>"
```

如果 `prompt.json` 包含 `duration`，runner 会根据 `duration` 和 `fps` 自动计算
帧数。视频任务的 `num_frames` 必须是 `1` 或 `4n+1`；`5s`、`24 fps` 对应
`121` 帧。

使用 auto negative 时追加 `--negative_prompt_json negative.json`。如果不传，
runner 会使用当前模式对应的内置默认 negative prompt。

下面所有示例里 `--output`（及 `--refiner_output`）指定的目录都会由 runner 自动创建。

默认情况下，runner 会在主进程输出简洁的模型加载状态日志，并显示去噪阶段的 tqdm
进度条；Hugging Face 内部的逐权重加载进度条会被关闭。如果需要安静的批处理日志，
可以加 `--quiet_progress`。

## 示例资源

仓库在 `assets/cases/` 下为每种模式准备了现成的结构化 JSON Prompt（`prompt.json`），
TI2V 还附带对应首帧图 `first_frame.png`，可以直接使用。

完整清单见 `assets/cases/manifest.json`。

## 只跑 T2V Base

```bash
python scripts/inference.py \
  --backend diffusers \
  --model_dir "$MODEL_DIR" \
  --mode t2v \
  --prompt_json "assets/cases/t2v/example_1/prompt.json" \
  --output "outputs/t2v_base.mp4" \
  --height 480 \
  --width 832 \
  --num_frames 121 \
  --fps 24 \
  --steps 40 \
  --guidance_scale 3 \
  --shift 3 \
  --transformer_dtype bf16 \
  --text_encoder_dtype bf16 \
  --vae_dtype fp32
```

## Base + Refiner

传入 `--run_refiner` 后，runner 会从同一个模型根目录加载并运行 `refiner/` DiT。
如果不传 `--run_refiner`，runner 不会加载也不会运行 refiner。

```bash
python scripts/inference.py \
  --backend diffusers \
  --model_dir "$MODEL_DIR" \
  --run_refiner \
  --mode t2v \
  --prompt_json "assets/cases/t2v/example_1/prompt.json" \
  --output "outputs/t2v_base.mp4" \
  --refiner_output "outputs/t2v_refined.mp4" \
  --height 480 \
  --width 832 \
  --refiner_height 1088 \
  --refiner_width 1920 \
  --num_frames 121 \
  --fps 24 \
  --steps 40 \
  --refiner_steps 8 \
  --guidance_scale 3 \
  --refiner_guidance_scale 3 \
  --shift 3 \
  --refiner_shift 3 \
  --refiner_t_thresh 0.85 \
  --refiner_sigma_tail_steps 2 \
  --transformer_dtype bf16 \
  --text_encoder_dtype bf16 \
  --vae_dtype fp32 \
  --refiner_vae_dtype fp32 \
  --refiner_vae_tiling \
  --refiner_vae_tile_height 384 \
  --refiner_vae_tile_width 640 \
  --refiner_vae_tile_stride_height 288 \
  --refiner_vae_tile_stride_width 480 \
  --release_base_before_refiner \
  --reuse_condition_features
```

refiner 只支持视频模式。开启 `--reuse_condition_features` 时，refiner 会复用
base 的 condition feature。
`--release_base_before_refiner` 会在 base 输出保存后释放 base pipeline；只有
长期服务需要保留 base pipeline 供后续请求复用时，才使用
`--no-release_base_before_refiner`。

### 视频 VAE Decode Tiling

视频 VAE decode 可以通过 tiling 降低峰值显存：Base VAE 使用 `--vae_tiling`，
Refiner VAE 使用 `--refiner_vae_tiling`。1080p Refiner 默认使用 `384x640` tile
和 `288x480` stride。Tiling 会增加少量 decode 时间；显存充足时可以分别使用
`--no-vae_tiling` 或 `--no-refiner_vae_tiling` 关闭。

## 多卡 FSDP 推理

当 base DiT 和 refiner DiT 需要同时常驻 GPU、但每张卡复制一份完整 DiT
显存压力较大时，使用 `--enable_fsdp_inference`。这个开关会用 PyTorch
composable FSDP 切分所有已加载的 DiT transformer：

- 只跑 base 时，切分 base `transformer/` DiT。
- 跑 base + refiner 时，在 base 生成开始前同时切分 base `transformer/` DiT 和
  `refiner/` DiT。
- VLM/text encoder、VAE、scheduler 和 prompt feature 不会被这个开关切分；
  如需切分 Qwen3-VL，需要独立添加 `--enable_vlm_fsdp_inference`。

DiT 和 VLM 两个开关可以分别启用，也可以组合使用。VLM FSDP 会切分 36 个语言层、
24 个视觉 block 和 encoder root，并且要求使用多于一个进程的 distributed 启动。
它可以降低 GPU 峰值显存，但不会提升稳定单请求速度；实测数据见
[推理性能与显存报告](performance_benchmark.md)。

### VLM FSDP 快速用法

VLM FSDP 必须通过 `torchrun` 使用多进程启动。在原有推理命令中添加对应开关：

| 需要切分的组件 | 参数 |
| --- | --- |
| 只切分 Qwen3-VL | `--enable_vlm_fsdp_inference` |
| 只切分 DiT | `--enable_fsdp_inference` |
| 同时切分 DiT 和 Qwen3-VL | 同时添加两个参数 |

8 GPU、CP8、同时切分 DiT 和 Qwen3-VL 时，在下方完整推理示例中保留这四个参数：

```text
--context_parallel_degree 8
--context_parallel_ulysses_anything
--enable_fsdp_inference
--enable_vlm_fsdp_inference
```

启动后检查日志中的 `vlm_fsdp_inference=VLMFSDPInferenceInfo(enabled=True, ...)`；
出现该字段表示 VLM FSDP 已生效。

FSDP 推理可以和 context parallel、CFG parallel 一起使用。预置脚本和示例默认
使用 sequential CFG 降低峰值显存。使用所有 GPU 做 context parallel 时，令
`--context_parallel_degree` 等于 GPU 数；只有明确希望使用 batched CFG 时才添加
`--batch_cfg`。如果只想做 FSDP 显存切分、不做 CP/CFG 并行，则用 `torchrun`
启动多进程，同时保持 `--cfg_parallel_degree 1 --context_parallel_degree 1`。

FSDP 推理降低的是 DiT wrap 之后的 GPU 显存占用。初始化阶段每个 rank 仍会先在
host memory 上构建一份 transformer，然后再做 FSDP 切分；大 MoE checkpoint 需要
确保机器有足够系统内存支撑启动进程数。

示例：多 GPU base + refiner 推理，并同时切分两个 DiT：

```bash
torchrun --standalone --nproc_per_node 8 scripts/inference.py \
  --backend diffusers \
  --model_dir "$MODEL_DIR" \
  --run_refiner \
  --mode t2v \
  --prompt_json "assets/cases/t2v/example_1/prompt.json" \
  --output "outputs/t2v_base.mp4" \
  --refiner_output "outputs/t2v_refined.mp4" \
  --height 480 \
  --width 832 \
  --refiner_height 1088 \
  --refiner_width 1920 \
  --fps 24 \
  --steps 40 \
  --refiner_steps 8 \
  --guidance_scale 3 \
  --refiner_guidance_scale 3 \
  --shift 3 \
  --refiner_shift 3 \
  --cfg_parallel_degree 1 \
  --context_parallel_degree 8 \
  --enable_fsdp_inference \
  --enable_vlm_fsdp_inference \
  --transformer_dtype bf16 \
  --text_encoder_dtype bf16 \
  --vae_dtype fp32 \
  --refiner_vae_dtype fp32 \
  --refiner_vae_tiling \
  --refiner_vae_tile_height 384 \
  --refiner_vae_tile_width 640 \
  --refiner_vae_tile_stride_height 288 \
  --refiner_vae_tile_stride_width 480 \
  --release_base_before_refiner \
  --reuse_condition_features
```

运行日志会分别打印 `dit_fsdp_inference=...` 和 `vlm_fsdp_inference=...`。
成功开启后，对应显示 `FSDPInferenceInfo(enabled=True, ...)` 和
`VLMFSDPInferenceInfo(enabled=True, ...)`。

## TI2V

rewriter 和 DiT 推理应该使用同一张首帧：rewriter 里传
`--first-frame "<first_frame.png>"`，DiT 里传 `--image "<first_frame.png>"`。

```bash
python scripts/inference.py \
  --backend diffusers \
  --model_dir "$MODEL_DIR" \
  --mode ti2v \
  --image "assets/cases/ti2v/example_4/first_frame.png" \
  --prompt_json "assets/cases/ti2v/example_4/prompt.json" \
  --output "outputs/ti2v.mp4" \
  --height 480 \
  --width 832 \
  --num_frames 121 \
  --fps 24 \
  --steps 40 \
  --guidance_scale 3 \
  --shift 3
```

当前 runner 没有为 TI2V 启用 CFG parallel。

> **多卡配置**：FSDP 显存切分和 context parallel 的用法与 T2V 完全一致，参考上文「多卡 FSDP 推理」段落，把 `--mode` 换成 `ti2v` 并补上 `--image` 即可（CFG parallel 除外，TI2V 不支持）。

## T2I

T2I 内部使用 `num_frames=1`，输出图片：

**⚠️ 注意：refiner 没有在图像上训练，T2I 不支持 refiner（`--run_refiner` 仅对视频模式有效）。**

> **多卡配置**：FSDP 显存切分和 context parallel 的用法与 T2V 完全一致，参考上文「多卡 FSDP 推理」段落，把 `--mode` 换成 `t2i`；因为 T2I 不支持 refiner，去掉所有 `--run_refiner` 及 `refiner_*` 相关参数。

```bash
python scripts/inference.py \
  --backend diffusers \
  --model_dir "$MODEL_DIR" \
  --mode t2i \
  --prompt_json "assets/cases/t2i/example_6/prompt.json" \
  --output "outputs/image.png" \
  --height 480 \
  --width 832 \
  --steps 40 \
  --guidance_scale 3 \
  --shift 3
```
