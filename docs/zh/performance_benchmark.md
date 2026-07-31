# LingBot-Video 推理速度与显存

以下结果均为 5 秒、24 FPS、121 帧、480 x 832 的 Base-only T2V 单请求端到端测试。

## 完整性能矩阵

| 模型 | 拓扑 | 执行路径 | 完整运行时间 | 稳定单次推理时间 | NVML 每卡峰值 |
| --- | --- | --- | ---: | ---: | ---: |
| Dense | Single | Direct Diffusers | 223.37 s | 157.57 s | 27.10 GiB |
| Dense | Single | Adapter Exact | 224.97 s | 157.36 s | 27.10 GiB |
| Dense | CP8 | Direct Diffusers | 89.53 s | 19.89 s | 29.28 GiB |
| Dense | CP8 | Adapter Exact | 89.95 s | 19.88 s | 29.28 GiB |
| Dense | CP8 + DiT-FSDP8 | Direct Diffusers | 90.30 s | 20.44 s | 27.33 GiB |
| Dense | CP8 + DiT-FSDP8 | Adapter Exact | 89.15 s | 20.43 s | 27.33 GiB |
| Dense | CP8 + DiT-FSDP8 + VLM-FSDP8 | Direct Diffusers | — | 20.497 s | 22.993 GiB |
| Dense | CP8 + DiT-FSDP8 + VLM-FSDP8 | Adapter Exact | — | 20.465 s | 22.993 GiB |
| MoE | Single | Direct Diffusers | 460.91 s | 381.95 s | 90.15 GiB |
| MoE | Single | Adapter Exact | 460.29 s | 381.71 s | 90.15 GiB |
| MoE | Single | FP8 expert | 395.81 s | 317.04 s | 110.37 GiB |
| MoE | CP8 | Direct Diffusers | 130.27 s | 46.76 s | 83.75 GiB |
| MoE | CP8 | Adapter Exact | 131.60 s | 46.71 s | 83.75 GiB |
| MoE | CP8 | FP8 expert | 125.71 s | 36.72 s | 110.80 GiB |
| MoE | CP8 + DiT-FSDP8 | Direct Diffusers | 142.93 s | 54.78 s | 37.61 GiB |
| MoE | CP8 + DiT-FSDP8 | Adapter Exact | 141.79 s | 54.85 s | 37.61 GiB |
| MoE | CP8 + DiT-FSDP8 | FP8 expert | 163.73 s | 73.24 s | 65.43 GiB |
| MoE | CP8 + DiT-FSDP8 + VLM-FSDP8 | Direct Diffusers | — | 54.912 s | 31.208 GiB |
| MoE | CP8 + DiT-FSDP8 + VLM-FSDP8 | Adapter Exact | — | 54.879 s | 31.208 GiB |

完整运行时间包含进程启动、模型加载和第一条完整输出；稳定单次推理时间表示同一常驻
进程完成预热后的单次完整请求。两个时间列均报告中位数。`—` 表示 VLM-FSDP
candidate 没有重复对应的完整运行测试，或者该统计项没有进入冻结汇总，不表示运行
失败。VLM-FSDP 的稳定单次推理时间同样来自 3 次预热后的 7 次完整请求。

## 速度解读

- Dense 的 Exact 速度优先配置是 8 GPU CP8，稳定单次推理时间约 `19.9 s`。
- Dense 显存优先配置 `CP8 + DiT-FSDP8 + VLM-FSDP8` 的稳定单次推理时间约为
  `20.5 s`。
- MoE 的 BF16 Exact 速度优先配置是 8 GPU CP8，稳定单次推理时间约 `46.7 s`。
- MoE CP8 FP8 expert 为本表最快配置，稳定单次推理时间为 `36.72 s`，但会使用更多
  GPU 显存，并且不属于 Exact 配置。
- MoE 显存优先配置 `CP8 + DiT-FSDP8 + VLM-FSDP8` 的稳定单次推理时间约为
  `54.9 s`。
- Direct Diffusers 与 Adapter Exact 复用相同的 pipeline execution loop，因此本表
  中两条 Exact 路径的单请求性能接近。这里没有测试 SGLang server scheduler、连续
  批处理、多请求并发或在线服务吞吐。

## 显存解读

- Dense CP8 每卡峰值为 `29.28 GiB`。DiT-FSDP8 将其降至约 `27.33 GiB`，继续
  开启 VLM-FSDP8 后降至 `22.993 GiB`。
- MoE CP8 每卡峰值为 `83.75 GiB`。DiT-FSDP8 将其降至约 `37.61 GiB`，继续
  开启 VLM-FSDP8 后降至 `31.208 GiB`。
- MoE CP8 FP8 expert 的每卡峰值为 `110.80 GiB`。FP8 cache 和 BF16 原权重同时
  驻留，因此它是速度配置，不是显存配置。
- VLM FSDP 会增加 host memory 压力，因为每个 rank 当前仍先在 CPU 上构建完整
  VLM，再执行 FSDP。实测 aggregate host RSS 最高约为 Dense `91.8 GiB`、MoE
  `145.2 GiB`；使用该配置前需要确认系统内存充足。

## 测试配置

- 模型：LingBot-Video Dense、LingBot-Video MoE。
- 输入：`assets/cases/t2v/example_3/prompt.json` 中的完整 structured caption。
- 输出：5 秒、24 FPS、121 帧、480 x 832、Base-only，不含 Refiner。
- 采样：40 steps、guidance 3、shift 3、seed 10、Sequential CFG。
- 精度：DiT/VLM BF16，VAE FP32。
- Attention backend：`_native_flash`，即 FlashAttention-3。
- VAE tiling：关闭。
- 多 GPU 测试运行在单个 8-GPU 节点；每组正式测试前确认 GPU 没有其他计算进程。

`Exact` 表示 BF16 原始 DiT/VLM 权重、FP32 VAE、`_native_flash` attention、
`grouped_mm` MoE expert backend，并且不启用 FP8、近似 cache、跳步或动态稀疏优化。
`Direct Diffusers` 使用 `--backend diffusers`；`Adapter Exact` 使用
`--backend sglang`，但仍复用相同 pipeline execution loop；只有标记为 FP8 expert
的行使用 SGLang fused FP8 expert kernel。

## 测量说明

- 完整运行时间从创建全新推理子进程开始，到第一条完整视频写入、并行进程同步清理
  并正常退出为止。它包含 import、distributed/NCCL、CP/FSDP、模型加载、首次 kernel
  初始化、condition、40-step denoising、VAE decode 和 MP4 导出。
- 完整运行时间每组至少测试两个独立进程；两次结果相差超过 5% 时补第 3 次并报告
  中位数。测试前 checkpoint 很可能已进入系统 page cache，因此它不是严格的磁盘
  冷启动时间。
- 稳定单次推理时间在同一常驻进程中连续执行 10 次完整请求，前 3 次预热，后 7 次
  用于统计中位数。每次请求都重新执行完整 condition、denoising、VAE decode 和导出。
- NVML peak 由被测进程外部每 100 ms 采样一次，每个配置使用独立 memory run。
- 基础矩阵包含 31 次完整运行、15 组稳定推理和 15 组显存测试；VLM-FSDP 额外增加
  4 组稳定推理和 4 组显存测试，没有重跑完整运行时间。
- 基础 15 组 T2V 矩阵测自 public-export revision `e6c1f9518`，4 组 VLM-FSDP
  candidate 在后续测试中补充。最终 encoder-only `lm_head` 清理没有重跑全部
  40-step 性能矩阵，因此本文数字应作为对应冻结 revision 的实测参考。

## 推荐配置

| 目标 | 推荐配置 | 稳定单次推理时间 | NVML 每卡峰值 |
| --- | --- | ---: | ---: |
| Dense 速度优先 | 8 GPU CP8 Exact | 约 19.9 s | 29.28 GiB |
| Dense 显存优先 | CP8 + DiT-FSDP8 + VLM-FSDP8 Exact | 约 20.5 s | 22.993 GiB |
| MoE BF16 速度优先 | 8 GPU CP8 Exact | 约 46.7 s | 83.75 GiB |
| MoE 极致速度，允许 FP8 | 8 GPU CP8 + FP8 expert | 36.72 s | 110.80 GiB |
| MoE 显存优先 | CP8 + DiT-FSDP8 + VLM-FSDP8 Exact | 约 54.9 s | 31.208 GiB |
