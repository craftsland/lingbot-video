#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -d /usr/local/cuda/compat ]]; then
  export LD_LIBRARY_PATH="/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}"
fi
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/rewriter:${PYTHONPATH:-}"
export DIFFUSERS_ATTN_BACKEND="${DIFFUSERS_ATTN_BACKEND:-_native_flash}"
export LINGBOT_MOE_PAD_BACKEND="${LINGBOT_MOE_PAD_BACKEND:-vectorized}"
export LINGBOT_MOE_EXPERT_BACKEND="${LINGBOT_MOE_EXPERT_BACKEND:-grouped_mm}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_DIR="${MODEL_DIR:-}"
PROMPT_JSON="${PROMPT_JSON:-assets/cases/t2v/example_1/prompt.json}"
OUT_DIR="${OUT_DIR:-outputs/moe_dmd_t2v_single_gpu_$(date +%Y%m%d_%H%M%S)}"

BACKEND="${BACKEND:-diffusers}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
SEED="${SEED:-42}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 2
fi

if [[ -z "$MODEL_DIR" ]]; then
  echo "Set MODEL_DIR to a complete LingBot-Video DMD student model directory." >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

"$PYTHON_BIN" scripts/inference.py \
  --backend "$BACKEND" \
  --model_dir "$MODEL_DIR" \
  --mode t2v \
  --prompt_json "$PROMPT_JSON" \
  --output "$OUT_DIR/t2v.mp4" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --scheduler dmd_student \
  --steps 8 \
  --guidance_scale 1 \
  --shift 3 \
  --duration 5 \
  --num_frames 121 \
  --seed "$SEED" \
  --fps 24 \
  --transformer_dtype bf16 \
  --text_encoder_dtype bf16 \
  --vae_dtype fp32

echo "Saved: $OUT_DIR/t2v.mp4"
