#!/bin/bash

# 2. 定义数据集数组（去除了重复的 oxford_flowers，共6个）
DATASETS=("Office31" "OfficeHome" "caltech101" "food101" "oxford_flowers" "oxford_pets")

# 3. 定义spf_gamma_init数组
GAMMA_INITS=("0.01" "0.02" "0.03" "0.04" "0.06" "0.07" "0.08" "0.09")

echo "🎉 开始批量运行所有联邦学习实验..."

for GAMMA in "${GAMMA_INITS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        
        echo "======================================================================"
        echo "🚀 正在运行 | 数据集: ${DATASET} | spf_gamma_init: ${GAMMA}"
        echo "💡 日志、配置和CSV将完全按照 federated_main.py 的内部路径规则自动保存"
        echo "======================================================================"

        # 直接执行原始命令，不干扰源代码的日志保存机制
        CUDA_VISIBLE_DEVICES=2 python federated_main.py \
            --trainer GL_SVDMSE \
            --dataset "$DATASET" \
            --num_shots 16 \
            --backbone ViT-B/16 \
            --num_users 10 \
            --seed 42 \
            --root /workspace/FedPHA/DATA \
            --use_spf \
            --spf_gamma_init "$GAMMA" \
            --spf_energy 0.90 \
            --spf_max_rank 8 \
            --spf_shared_lambda 0.1

        echo "✅ 当前任务完成 | 数据集: ${DATASET} | spf_gamma_init: ${GAMMA}"
        echo -e "\n"
        
    done
done

echo "🏁 所有实验已全部运行完毕！"