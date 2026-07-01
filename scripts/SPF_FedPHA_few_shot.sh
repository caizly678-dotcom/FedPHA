# bash scripts/SPF_FedPHA_few_shot.sh

TRAINER="GL_SVDMSE"
DATASET="caltech101"
SHOTS=2
BACKBONE="rn50"
USERS=10
SEED=1

python federated_main.py \
  --trainer ${TRAINER} \
  --dataset ${DATASET} \
  --num_shots ${SHOTS} \
  --backbone ${BACKBONE} \
  --num_users ${USERS} \
  --seed ${SEED} \
  --use_spf \
  --spf_energy 0.90 \
  --spf_max_rank 8 \
  --spf_shared_lambda 0.1 \
  --spf_late_alpha 0.5 \
  --spf_fused_ce_lambda 1.0 \
  --spf_global_ce_lambda 1.0 \
  --spf_local_ce_lambda 1.0
