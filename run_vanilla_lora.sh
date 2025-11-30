#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_vanilla_lora.sh [dataset]
# dataset ∈ {wikitext, imdb, sst2, squad, squad_v2}

DATASET="${1:-wikitext}"
RUN_NAME="${RUN_NAME:-${DATASET}_$(date +%Y%m%d_%H%M%S)}"
MODEL_NAME="${MODEL_NAME:-EleutherAI/gpt-neo-125m}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-2e-5}"
RANK="${RANK:-4}"
ALPHA="${ALPHA:-16}"
MAX_LENGTH="${MAX_LENGTH:-128}"
TRAIN_LIMIT="${TRAIN_LIMIT:-0}"     # 0 = full dataset
GPU_ID="${GPU_ID:-0}"
SAVE_BASE="${SAVE_BASE:-1}"         # save frozen base projections once
RUN_EVALS="${RUN_EVALS:-1}"

if [[ "${DATASET}" == squad* ]]; then
  MAX_LENGTH="${MAX_LENGTH:-256}"
fi

CMD=(python src/main.py
  --dataset "${DATASET}"
  --model-name "${MODEL_NAME}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --rank "${RANK}"
  --alpha "${ALPHA}"
  --max-length "${MAX_LENGTH}"
  --run-name "${RUN_NAME}"
)

if [[ "${TRAIN_LIMIT}" != "0" ]]; then
  CMD+=(--train-limit "${TRAIN_LIMIT}")
fi

if [[ "${SAVE_BASE}" == "1" ]]; then
  CMD+=(--save-base)
fi

if [[ "${RUN_EVALS}" == "1" ]]; then
  CMD+=(--run-evals)
fi

echo "Running on GPU ${GPU_ID} with run name ${RUN_NAME}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}"
