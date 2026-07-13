#!/bin/bash
# bash scripts/SPF_FedPHA_few_shot.sh

TRAINER="GL_SVDMSE"
DATASET="dtd"
SHOTS=16
BACKBONE="ViT-B/16"
USERS=10
SEED=1

python federated_main.py \
  --trainer ${TRAINER} \
  --dataset ${DATASET} \
  --root ./DATA \
  --num_shots ${SHOTS} \
  --backbone ${BACKBONE} \
  --num_users ${USERS} \
  --seed ${SEED} \
  --use_spf \
  --spf_gamma_init 0.05 \
  --spf_energy 0.80 \
  --spf_max_rank 8 \
  --spf_shared_lambda 0.1
