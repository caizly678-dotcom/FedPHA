#!/bin/bash

# 1. 定义 spf_gamma_init 的值数组
DATASETS=("Office31" "OfficeHome" "caltech101" "food101" "oxford_flowers" "oxford_pets")


echo "🎉 开始针对不同 spf_gamma_init 的批量实验..."

# 2. 遍历 Gamma 值
for DATASET in "${DATASETS[@]}"; do

    
    echo "======================================================================"
    echo "🚀 正在运行 | spf_gamma_init: ${GAMMA}"
    echo "💡 使用种子: 42 | 数据集: dtd"
    echo "======================================================================"

    # 执行实验
    # 采用你提供的原始命令结构，保持参数设置的一致性
    CUDA_VISIBLE_DEVICES=3 python federated_main.py \
        --trainer GL_SVDMSE \
        --dataset "${DATASET}" \
        --num_shots 16 \
        --backbone ViT-B/16 \
        --num_users 10 \
        --seed 42 \
        --root /workspace/FedPHA/DATA \
        --use_spf \
        --spf_gamma_init 0.01 \
        --spf_energy 0.80 \
        --spf_max_rank 8 \
        --spf_shared_lambda 0.1 \
        --spf_dynamic_gamma \
        --spf_gamma_verbose 

    echo -e "\n"
        
done

echo "🏁 所有 gamma 参数相关的实验已全部运行完毕！"