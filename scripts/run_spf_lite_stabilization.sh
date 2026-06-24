#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash scripts/run_spf_lite_stabilization.sh dtd <variant> <seed>" >&2
  echo "Variants: weak_fusion_baseline | shared_init_only | fixed_basis_only | shared_init_fixed_basis" >&2
  exit 1
fi

DATASET="$1"
VARIANT="$2"
SEED="$3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

case "${VARIANT}" in
  weak_fusion_baseline)
    SPF_SHARED_INIT=()
    SPF_FIXED_ROUND_BASIS=()
    ;;
  shared_init_only)
    SPF_SHARED_INIT=(--spf_shared_init)
    SPF_FIXED_ROUND_BASIS=()
    ;;
  fixed_basis_only)
    SPF_SHARED_INIT=()
    SPF_FIXED_ROUND_BASIS=(--spf_fixed_round_basis)
    ;;
  shared_init_fixed_basis)
    SPF_SHARED_INIT=(--spf_shared_init)
    SPF_FIXED_ROUND_BASIS=(--spf_fixed_round_basis)
    ;;
  *)
    echo "Unknown variant: ${VARIANT}" >&2
    exit 1
    ;;
esac

python federated_main.py \
  --trainer GL_SVDMSE \
  --dataset "${DATASET}" \
  --backbone ViT-B/16 \
  --num_shots 16 \
  --num_users 10 \
  --seed "${SEED}" \
  --root /workspace/FedPHA/DATA \
  --use_spf \
  --spf_variant "${VARIANT}" \
  --spf_gamma_init 0.01 \
  --spf_energy 0.80 \
  --spf_max_rank 8 \
  "${SPF_SHARED_INIT[@]}" \
  "${SPF_FIXED_ROUND_BASIS[@]}"
