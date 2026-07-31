# SGLang Diffusion

Use this guide for SGLang Diffusion inference:

```bash
python scripts/inference.py --backend sglang ...
```

SGLang Diffusion is mainly intended for high-throughput serving and large-batch
parallel generation. For a single prompt, direct diffusers can be competitive or
faster because distributed launch, communication, and fused-kernel setup overhead
may dominate. Use SGLang when you want to keep multiple requests or multiple
samples in flight, run CP/FSDP layouts, or use the optional MoE fused kernels.

If the optional SGLang package is not installed, `--backend sglang`
automatically falls back to direct diffusers and prints a warning. Install the
optional requirements to enable the SGLang runtime:

```bash
python -m pip install --no-deps -r requirements-sglang.txt
```

Use `--no-deps` here so the PyTorch/CUDA stack from `requirements.txt` stays
unchanged.

## GPU Layout

Parallel inference has two independent degrees:

- `--cfg_parallel_degree` splits the two classifier-free-guidance branches
  (conditional / unconditional). It is either `1` or `2`.
- `--context_parallel_degree` shards the sequence (context parallel) across GPUs.

For context parallel you can add `--context_parallel_ulysses_anything`, which
uses Ulysses all-to-all communication and supports arbitrary (non-divisible)
sequence splits. The examples below enable it by default; you normally do not
need to change it.

The number of processes must equal the product of the two degrees:

```
nproc_per_node = cfg_parallel_degree × context_parallel_degree
```

Pick the layout from the number of GPUs you have:

| GPUs | Layout | Flags |
| ---: | --- | --- |
| 1 | single GPU | omit `torchrun`; `--cfg_parallel_degree 1 --context_parallel_degree 1` |
| 2 | CFG×CP | `--cfg_parallel_degree 2 --context_parallel_degree 1` |
| 4 | CFG×CP | `--cfg_parallel_degree 2 --context_parallel_degree 2` |
| N (even) | CFG×CP | `--cfg_parallel_degree 2 --context_parallel_degree N/2` |
| N | CP-only | `--cfg_parallel_degree 1 --context_parallel_degree N` |

The examples below use shell variables so they work for any GPU count. Set them
once to match your machine:

```bash
export CFG=2                 # 1 or 2
export CP=4                  # context-parallel degree
export NPROC=$((CFG * CP))   # processes to launch
```

For a single GPU, set `CFG=1 CP=1` and run `python scripts/inference.py ...`
directly, without `torchrun`.

## FSDP Inference Sharding

Add `--enable_fsdp_inference` when you want to keep the base DiT and refiner
DiT loaded together while sharding their transformer weights across the launched
processes. This flag is backend-agnostic: with `--backend sglang`, SGLang owns
the native execution path while PyTorch composable FSDP shards the DiT modules.
Add `--enable_vlm_fsdp_inference` when Qwen3-VL should also be sharded. The VLM
flag is independent of the DiT flag and uses the same distributed process mesh.

FSDP can be combined with the SGLang layouts above:

- CP-only: `--cfg_parallel_degree 1 --context_parallel_degree <GPU_COUNT>`.
  The provided scripts use sequential CFG by default; add `--batch_cfg` and
  `--refiner_batch_cfg` only when you explicitly want batched CFG.
- CFG×CP: keep `--batch_cfg` and `--refiner_batch_cfg` off, then set
  `--cfg_parallel_degree 2 --context_parallel_degree <GPU_COUNT/2>`.
- FSDP-only memory sharding: launch with `torchrun --nproc_per_node <GPU_COUNT>`
  and keep both parallel degrees at `1`.

When `--run_refiner` and `--enable_fsdp_inference` are both present, the runner
loads the base DiT and refiner DiT first, then shards both before base sampling
starts. The runtime logs include separate base and refiner
`dit_fsdp_inference=FSDPInferenceInfo(...)` and
`vlm_fsdp_inference=VLMFSDPInferenceInfo(...)` fields.

This is GPU memory sharding, not a meta-init loader. During initialization, each
rank still constructs the transformer on host memory before FSDP sharding, so
large MoE checkpoints require enough system RAM for the chosen process count.
VLM FSDP has the same host-memory limitation. Its measured trade-offs are
included in the [inference performance benchmark](performance_benchmark.md).

## Dense Run

The Dense model has no experts, so the `LINGBOT_MOE_*` variables do not apply.
Configure only the attention backend and the GPU layout:

```bash
export DIFFUSERS_ATTN_BACKEND=_native_flash

torchrun --standalone --nproc_per_node $NPROC scripts/inference.py \
  --backend sglang \
  --model_dir "$MODEL_DIR" \
  --mode t2v \
  --prompt_json "assets/cases/t2v/example_1/prompt.json" \
  --output "outputs/t2v_base.mp4" \
  --height 480 \
  --width 832 \
  --fps 24 \
  --steps 40 \
  --guidance_scale 3 \
  --shift 3 \
  --cfg_parallel_degree $CFG \
  --context_parallel_degree $CP \
  --context_parallel_ulysses_anything \
  --enable_fsdp_inference \
  --transformer_dtype bf16 \
  --text_encoder_dtype bf16 \
  --vae_dtype fp32
```

The file passed to `--prompt_json` can be a dict or a non-empty list. Each
sample can either contain a `caption` field whose value is the structured prompt
or be the structured prompt object itself. If it contains `duration`, frame
count is derived from `duration` and `fps`.

## MoE Run — Grouped Experts

The MoE model adds expert kernels. Grouped MoE is the default quality-oriented
expert backend:

```bash
export DIFFUSERS_ATTN_BACKEND=_native_flash
export LINGBOT_MOE_PAD_BACKEND=vectorized
export LINGBOT_MOE_EXPERT_BACKEND=grouped_mm

torchrun --standalone --nproc_per_node $NPROC scripts/inference.py \
  --backend sglang \
  --model_dir "$MODEL_DIR" \
  --run_refiner \
  --mode t2v \
  --prompt_json "assets/cases/t2v/example_1/prompt.json" \
  --output "outputs/t2v_base.mp4" \
  --refiner_output "outputs/t2v_refined.mp4" \
  --height 480 \
  --width 832 \
  --fps 24 \
  --steps 40 \
  --refiner_steps 8 \
  --guidance_scale 3 \
  --refiner_guidance_scale 3 \
  --shift 3 \
  --refiner_shift 3 \
  --refiner_t_thresh 0.85 \
  --refiner_sigma_tail_steps 2 \
  --cfg_parallel_degree $CFG \
  --context_parallel_degree $CP \
  --context_parallel_ulysses_anything \
  --enable_fsdp_inference \
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

For 1080p refiner output, refiner VAE tiling is enabled by default with
`384x640` tiles and `288x480` strides. Use `--no-refiner_vae_tiling` only when
decode memory is already comfortable and you prefer faster VAE decode.
The provided refiner scripts also release the base pipeline before refiner
execution to reduce peak memory.

## MoE Run — Speed-First FP8

Use this path when visual throughput matters most. It switches the MoE expert
kernel to the FP8 SGLang Triton path. Export the FP8 expert backend, then run
the same `torchrun ... scripts/inference.py` command as the grouped MoE run
above:

```bash
export DIFFUSERS_ATTN_BACKEND=_native_flash
export LINGBOT_MOE_PAD_BACKEND=vectorized
export LINGBOT_MOE_EXPERT_BACKEND=sglang_triton_fp8
```

The GPU layout is the same as the grouped MoE run above (see "GPU Layout").

FP8 MoE may change generated details relative to grouped MoE. Use it when speed
matters more than strict numerical reproducibility.

## Acceleration Choices (MoE)

| Need | Setting |
| --- | --- |
| Default grouped experts | `LINGBOT_MOE_EXPERT_BACKEND=grouped_mm` |
| Fast visual screening | `LINGBOT_MOE_EXPERT_BACKEND=sglang_triton_fp8` |
| DiT memory sharding | add `--enable_fsdp_inference` |
| VLM memory sharding | add `--enable_vlm_fsdp_inference` |

For the GPU layout (CFG×CP / CP-only), see the "GPU Layout" section above.

For 8-GPU CP + FSDP smoke tests, use the public scripts below. They default to
direct diffusers; set `BACKEND=sglang` externally to exercise SGLang Diffusion.

```bash
MODEL_DIR="<path_to_lingbot-video-dense>" ./scripts/multi-gpus/run_dense_t2i_fsdp_cp8.sh
MODEL_DIR="<path_to_lingbot-video-dense>" ./scripts/multi-gpus/run_dense_t2v_fsdp_cp8.sh
MODEL_DIR="<path_to_lingbot-video-dense>" ./scripts/multi-gpus/run_dense_ti2v_fsdp_cp8.sh

MODEL_DIR="<path_to_lingbot-video-moe>" ./scripts/multi-gpus/run_moe_t2i_fsdp_cp8.sh
MODEL_DIR="<path_to_lingbot-video-moe>" ./scripts/multi-gpus/run_moe_t2v_refiner_fsdp_cp8.sh
MODEL_DIR="<path_to_lingbot-video-moe>" ./scripts/multi-gpus/run_moe_ti2v_refiner_fsdp_cp8.sh
```

Activate your environment before running these scripts, or set
`PYTHON_BIN="<path_to_python>"`.
