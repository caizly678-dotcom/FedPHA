#!/usr/bin/env bash
set -euo pipefail

# Examples:
#   bash scripts/SPF_align_few_shot.sh
#   DATASET=dtd SEED=2 bash scripts/SPF_align_few_shot.sh

TRAINER="${TRAINER:-GL_SVDMSE}"
DATASET="${DATASET:-caltech101}"
SHOTS="${SHOTS:-16}"
BACKBONE="${BACKBONE:-ViT-B/16}"
USERS="${USERS:-10}"
SEED="${SEED:-1}"
GAMMA="${GAMMA:-0.05}"
ENERGY="${ENERGY:-0.80}"
MIN_RANK="${MIN_RANK:-1}"
MAX_RANK="${MAX_RANK:-8}"
SHARED_LAMBDA="${SHARED_LAMBDA:-0.1}"
WARMUP_ROUNDS="${WARMUP_ROUNDS:-5}"
ALIGN_TAU="${ALIGN_TAU:-0.07}"
SINKHORN_ITERS="${SINKHORN_ITERS:-5}"
CONF_POWER="${CONF_POWER:-1.0}"

python federated_main.py \
  --trainer "${TRAINER}" \
  --dataset "${DATASET}" \
  --num_shots "${SHOTS}" \
  --backbone "${BACKBONE}" \
  --num_users "${USERS}" \
  --seed "${SEED}" \
  --use_spf \
  --spf_use_alignment \
  --spf_detach_private \
  --spf_gamma_init "${GAMMA}" \
  --spf_energy "${ENERGY}" \
  --spf_min_rank "${MIN_RANK}" \
  --spf_max_rank "${MAX_RANK}" \
  --spf_shared_lambda "${SHARED_LAMBDA}" \
  --spf_warmup_rounds "${WARMUP_ROUNDS}" \
  --spf_align_tau "${ALIGN_TAU}" \
  --spf_sinkhorn_iters "${SINKHORN_ITERS}" \
  --spf_conf_power "${CONF_POWER}" \
  "$@"
