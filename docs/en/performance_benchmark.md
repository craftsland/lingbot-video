# LingBot-Video Inference Speed and Memory

All results below are end-to-end, single-request measurements for five-second, 24 FPS,
121-frame, 480 x 832 base-only T2V generation.

## Complete Performance Matrix

| Model | Topology | Execution path | Full run time | Steady single-run time | Peak NVML/GPU |
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

Full run time includes process startup, model loading, and the first complete output. Steady
single-run time measures one complete request in the same resident process after warm-up.
Both time columns report medians. `—` means that the VLM-FSDP candidate did not repeat the
corresponding full-run measurement or that the statistic was not included in the frozen
summary. It does not indicate a failed run. VLM-FSDP steady single-run values also use seven
complete requests after three warm-ups.

## Speed

- The speed-oriented Dense Exact configuration is 8-GPU CP8, with about `19.9 s` steady
  single-run time.
- The memory-oriented Dense `CP8 + DiT-FSDP8 + VLM-FSDP8` configuration has about
  `20.5 s` steady single-run time.
- The speed-oriented BF16 Exact MoE configuration is 8-GPU CP8, with about `46.7 s`
  steady single-run time.
- MoE CP8 FP8 expert is the fastest configuration in the table at `36.72 s`, but it uses
  more GPU memory and is not an Exact configuration.
- The memory-oriented MoE `CP8 + DiT-FSDP8 + VLM-FSDP8` configuration has about
  `54.9 s` steady single-run time.
- Direct Diffusers and Adapter Exact reuse the same pipeline execution loop, so their
  single-request performance is similar in this table. This benchmark does not cover the
  SGLang server scheduler, continuous batching, concurrent requests, or online throughput.

## GPU Memory

- Dense CP8 peaks at `29.28 GiB/GPU`. DiT-FSDP8 reduces it to about `27.33 GiB/GPU`,
  and adding VLM-FSDP8 reduces it further to `22.993 GiB/GPU`.
- MoE CP8 peaks at `83.75 GiB/GPU`. DiT-FSDP8 reduces it to about `37.61 GiB/GPU`,
  and adding VLM-FSDP8 reduces it further to `31.208 GiB/GPU`.
- MoE CP8 FP8 expert peaks at `110.80 GiB/GPU`. The FP8 cache and original BF16 weights
  coexist, so this is a speed configuration rather than a memory configuration.
- VLM FSDP increases host-memory pressure because every rank currently constructs a full
  VLM on CPU before applying FSDP. Measured aggregate host RSS reaches approximately
  `91.8 GiB` for Dense and `145.2 GiB` for MoE; verify that sufficient system memory is
  available before using this topology.

## Test Configuration

- Models: LingBot-Video Dense and LingBot-Video MoE.
- Input: the complete structured caption in
  `assets/cases/t2v/example_3/prompt.json`.
- Output: five seconds, 24 FPS, 121 frames, 480 x 832, base model only, no refiner.
- Sampling: 40 steps, guidance 3, shift 3, seed 10, sequential CFG.
- Precision: BF16 DiT/VLM and FP32 VAE.
- Attention backend: `_native_flash`, which selects FlashAttention-3.
- VAE tiling: disabled.
- Multi-GPU measurements use one 8-GPU node. The GPUs were checked for unrelated compute
  processes before every formal run.

`Exact` means original BF16 DiT/VLM weights, FP32 VAE, `_native_flash` attention, the
`grouped_mm` MoE expert backend, and no FP8, approximate cache, step skipping, or dynamic
sparsity. `Direct Diffusers` uses `--backend diffusers`. `Adapter Exact` uses
`--backend sglang` but reuses the same pipeline execution loop. Only rows labeled FP8
expert use the SGLang fused FP8 expert kernel.

## Measurement Notes

- Full run time starts when a new inference subprocess is created and ends after the
  first complete video is written, distributed workers finish synchronized cleanup, and
  the process exits normally. It includes imports, distributed/NCCL and CP/FSDP setup,
  model loading, first-use kernel initialization, conditioning, 40-step denoising, VAE
  decoding, and MP4 export.
- Each full-run configuration uses at least two independent processes. A third run is added
  when the first two differ by more than 5%, and the median is reported. The checkpoint was
  likely present in the system page cache, so this is not strict disk-cold startup time.
- Steady single-run time is measured from ten complete requests in one resident process.
  The first three are warm-ups; the remaining seven provide the median. Every request repeats full
  conditioning, denoising, VAE decoding, and export.
- An external process samples NVML device memory every 100 ms. Each configuration uses a
  separate memory run.
- The base matrix contains 31 full runs, 15 steady-inference groups, and 15 memory runs.
  VLM-FSDP adds four steady-inference and four memory groups; full-run measurements were
  not repeated.
- The base 15-row T2V matrix was measured at public-export revision `e6c1f9518`; four
  VLM-FSDP candidates were added in a later study. The final encoder-only `lm_head` cleanup
  did not rerun the full 40-step performance matrix, so these numbers should be treated as
  measured references for their corresponding frozen revisions.

## Recommended Configurations

| Goal | Recommended configuration | Steady single-run time | Peak NVML/GPU |
| --- | --- | ---: | ---: |
| Dense speed | 8-GPU CP8 Exact | about 19.9 s | 29.28 GiB |
| Dense memory | CP8 + DiT-FSDP8 + VLM-FSDP8 Exact | about 20.5 s | 22.993 GiB |
| MoE BF16 speed | 8-GPU CP8 Exact | about 46.7 s | 83.75 GiB |
| Maximum MoE speed with FP8 | 8-GPU CP8 + FP8 expert | 36.72 s | 110.80 GiB |
| MoE memory | CP8 + DiT-FSDP8 + VLM-FSDP8 Exact | about 54.9 s | 31.208 GiB |
