import re
import numpy as np
import pandas as pd
from pathlib import Path

# ================= 配置区域 =================
# 基于你的工作区，设定 output 为根搜索目录
BASE_DIR = Path('output_A0')
DATASETS = [
    'caltech101', 'dtd', 'food101',
    'Office31', 'OfficeHome', 'oxford_flowers', 'oxford_pets'
]
SEEDS = [1, 2, 3]

# 针对 utils/fed_utils.py 中的打印格式进行精确正则匹配
# 匹配目标: "--Global test acc: 75.5342..."
GLOBAL_ACC_REGEX = re.compile(r"--Global\s+test\s+acc:\s+([0-9\.]+)")
# ============================================

def extract_global_acc_from_log(file_path):
    """读取日志，专门提取每一轮的 Global test acc"""
    acc_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = GLOBAL_ACC_REGEX.search(line.strip())
                if match:
                    # 提取数值
                    acc_list.append(float(match.group(1)))
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
    return acc_list

def main():
    results = []

    for dataset in DATASETS:
        dataset_path = BASE_DIR / dataset
        if not dataset_path.exists():
            print(f"⚠️ 提示: 未找到数据集目录 {dataset_path}")
            continue

        # 存储当前数据集 3 个 seed 的“最后10轮平均 Global Acc”
        seed_last_10_global_avgs = []
        
        for seed in SEEDS:
            # 兼容 FedPHA 嵌套路径格式
            pattern = f"**/seed_{seed}/**/log.txt"
            log_files = list(dataset_path.rglob(pattern))
            
            if not log_files:
                print(f"⚠️ 提示: 未找到 {dataset} 目录下 seed_{seed} 的 log.txt 文件。")
                continue
            
            log_file = log_files[0]
            acc_list = extract_global_acc_from_log(log_file)
            
            if acc_list:
                # 获取该 seed 下最后 10 轮的 Global Acc
                last_10 = acc_list[-10:] if len(acc_list) >= 10 else acc_list
                # 计算该 seed 最后 10 轮的均值
                last_10_avg = np.mean(last_10)
                seed_last_10_global_avgs.append(last_10_avg)
            else:
                print(f"⚠️ 警告: {log_file} 中未找到 '--Global test acc:' 数据。")

        # 汇总 3 个 seed 的结果，计算最终的均值和方差
        if seed_last_10_global_avgs:
            dataset_res = {
                'Dataset': dataset,
                # 3个种子的 最后10轮Global均值 的 平均值
                'Global_Last10_Mean': round(np.mean(seed_last_10_global_avgs), 4),
                # 3个种子的 最后10轮Global均值 的 方差 (ddof=1)
                'Global_Last10_Var': round(np.var(seed_last_10_global_avgs, ddof=1) if len(seed_last_10_global_avgs) > 1 else 0.0, 6)
            }
            results.append(dataset_res)

    # 打印终端可视化表格并导出 CSV
    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*70)
        print(" "*15 + "FedPHA 全局测试准确率 (Global Test Acc) 汇总" + " "*15)
        print("="*70)
        print(df.to_markdown(index=False))
        print("="*70)
        
        output_csv = 'global_acc_summary.csv'
        df.to_csv(output_csv, index=False)
        print(f"\n✅ 汇总表格已成功保存至: {output_csv}")
    else:
        print("\n❌ 没有提取到任何有效数据，请检查日志。")

if __name__ == '__main__':
    main()