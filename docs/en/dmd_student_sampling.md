# Few-Step (DMD Student) Sampling

DMD-distilled student checkpoints generate a full video in 8 steps instead of 40.
A distilled student must be sampled with the exact step geometry it was trained
on; the standard `FlowUniPCMultistepScheduler` is **not** that geometry and
degrades student output. Use `DMDStudentScheduler` instead.

## Model weights

Download the DMD student checkpoint
as a complete local model directory containing `model_index.json`,
`transformer/`, `text_encoder/`, `processor/`, `vae/`, and `scheduler/`.
The transformer must be the distilled student, not the ordinary MoE checkpoint.

For the bundled 5-second examples, use
`scripts/single-gpu/run_moe_dmd_t2v.sh` and
`scripts/single-gpu/run_moe_dmd_ti2v.sh`. Set `MODEL_DIR` first. These scripts
reuse the existing inference runner, set the trained 8-step recipe, and use
structured prompts and a matching first frame from `assets/cases/`.

## The recipe

| Setting | Value | Why |
|---|---|---|
| Scheduler | `DMDStudentScheduler` | DDIM(eta=1) above warped sigma 0.5, Euler ODE below — the trained geometry |
| `num_inference_steps` | 8 | The distillation step count |
| `guidance_scale` | **1.0** | Teacher CFG is distilled into the student; extra guidance double-applies it |
| `shift` | 3.0 | Must match the distillation-time sigma warp |
| Resolution / frames | 480×832, 121 frames @ 24 fps | Training distribution |

## Usage

```python
import torch
from lingbot_video import (
    DMDStudentScheduler,
    LingBotVideoPipeline,             # t2v
    LingBotVideoImageToVideoPipeline, # ti2v
)

pipe = LingBotVideoPipeline.from_pretrained(model_root, torch_dtype=torch.bfloat16).to("cuda")
pipe.scheduler = DMDStudentScheduler()  # drop-in replacement

frames = pipe(
    prompt=structured_caption,  # structured JSON caption (see Prompt Preparation)
    num_inference_steps=8,
    guidance_scale=1.0,
    shift=3.0,
    height=480, width=832, num_frames=121,
    generator=torch.Generator("cpu").manual_seed(42),
    output_type="np",
).frames
```

Text-to-video and image-to-video (first-frame conditioned) both work: the i2v
pipeline's per-step condition-frame restore composes with this scheduler
unchanged.

Or via the CLI — identical to a normal run plus `--scheduler dmd_student`.
Both engines are supported: `--backend diffusers` and `--backend sglang`
(the sglang engine denoises through the same pipeline loop with
sglang-accelerated kernels, so the sampler geometry is identical). For ti2v,
use `--mode ti2v --image <first_frame>` instead of `--mode t2v`:

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

The refiner (if enabled) keeps its own `FlowUniPCMultistepScheduler`; the flag
only changes the base model's sampler.

## Verified parity

This sampler reproduces the trained sampling geometry **bit-exact**, for both
t2v and ti2v (first-frame conditioned): with identical injected noise, the
text embeddings, all 8 model forwards, every scheduler transition, and the
final latent were verified to match with zero error. Two settings are required
to reproduce that exact correspondence:

```bash
export DIFFUSERS_ATTN_BACKEND=_flash_3   # flash-attn-3 kernels
# Also disable both matmul and cuDNN TF32 in the inference process:
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False
# torch.set_float32_matmul_precision("highest")
```

The default attention backend is not a bitwise-reproduction setting. Changes
in attention kernels and TF32 settings can change the generated video through
MoE routing and numerical differences.

In the current CLI, `--no-allow_tf32` does not reset TF32 if the process already
has it enabled, and the runtime log reports only the matmul flag. Do not rely
on that flag alone for strict reproduction. Record both matmul and cuDNN state;
for example, `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` can enable matmul TF32 at process
startup. The quickstart examples are intended for normal video generation.

## Notes

- The stochastic steps consume the `generator` you pass to the pipeline — seed
  it for reproducible runs.
- Prompts must be structured JSON captions (the model's native format); plain
  prose degrades the first frame. See [Prompt Preparation](prompt_preparation.md).
- Teacher / non-distilled checkpoints should keep `FlowUniPCMultistepScheduler`
  (40 steps, guidance 3.0).
- Do not tune `high_noise_threshold` or `ddim_eta` away from the defaults
  (0.5 / 1.0) unless the checkpoint was distilled with matching values.
