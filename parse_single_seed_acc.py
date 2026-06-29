import re
import numpy as np
import pandas as pd
from pathlib import Path

# ================= 配置区域 =================
BASE_DIR = Path('output_A1')
DATASETS = [
    'caltech101', 'dtd', 'food101', 'oxford_flowers', 
    'oxford_pets', 'Office31', 'OfficeHome'
]
SEEDS = [1, 2, 3]

# 核心修改点：精准匹配你的单行整合输出格式！
# 自动提取 GM(micro), PM(macro) [即 legacy PM], WorstClient, ClientStd 和 RoundTime
SUMMARY_REGEX = re.compile(
    r"Round \d+\s*\|\s*GM\(micro\)=([0-9\.]+)\s*\|\s*PM\(micro\)=([0-9\.]+)\s*\|\s*PM\(macro\)=([0-9\.]+)\s*\|\s*WorstClient=([0-9\.]+)\s*\|\s*ClientStd=([0-9\.]+).*?RoundTime=([0-9\.]+)s"
)
# ============================================

def parse_log_for_metrics(file_path):
    gms, pms, worsts, stds, times = [], [], [], [], []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = SUMMARY_REGEX.search(line)
                if match:
                    gms.append(float(match.group(1)))       # 提取 GM(micro)
                    pms.append(float(match.group(3)))       # 提取 PM(macro) 作为 legacy PM
                    worsts.append(float(match.group(4)))    # 提取 WorstClient
                    stds.append(float(match.group(5)))      # 提取 ClientStd
                    times.append(float(match.group(6)))     # 提取 单轮耗时
    except Exception as e:
        print(f"读取文件出错: {e}")
        return None

    if not gms:
        return None

    res = {}
    
    # ================= 1. 全局模型 (GM) 指标 =================
    res['Best GM'] = np.max(gms)
    res['Final GM'] = gms[-1]
    last10_gm = gms[-10:] if len(gms) >= 10 else gms
    res['Last10 GM Mean'] = np.mean(last10_gm)
    res['Last10 GM Std'] = np.std(last10_gm, ddof=1) if len(last10_gm) > 1 else 0.0

    # ================= 2. 个性化模型 (PM) 指标 =================
    # (完全对应你的 legacy_mean_personalized_acc)
    res['Best PM'] = np.max(pms)
    res['Final PM'] = pms[-1]
    
    # 【新增】计算最后十轮平均 legacy_mean_personalized_acc
    last10_pm = pms[-10:] if len(pms) >= 10 else pms
    res['Last10 PM Mean'] = np.mean(last10_pm)
    
    res['Worst PM'] = worsts[-1] # 最后一轮的最差客户端

    # ================= 3. 异构公平性与效率 =================
    res['Client Std'] = stds[-1] # 最后一轮的客户端方差
    res['Time/Round'] = np.mean(times)
    
    return res

def main():
    results = []
    print("\n" + "="*145)
    print(" "*45 + "FedPHA: 全维度综合评估指标 (3 个 Seed 平均)" + " "*45)
    print("="*145)

    for dataset in DATASETS:
        dataset_path = BASE_DIR / dataset
        if not dataset_path.exists():
            continue

        seed_metrics = []
        for seed in SEEDS:
            pattern = f"**/seed_{seed}/**/log.txt"
            log_files = list(dataset_path.rglob(pattern))
            if not log_files:
                continue
            
            # 解析搜索到的第一个 log.txt
            log_file = log_files[0]
            metrics = parse_log_for_metrics(log_file)
            if metrics:
                seed_metrics.append(metrics)
        
        # 跨 3 个 seed 求平均值
        if seed_metrics:
            row = {'Dataset': dataset}
            keys = seed_metrics[0].keys()
            for k in keys:
                vals = [m[k] for m in seed_metrics if not pd.isna(m[k])]
                if vals:
                    avg_val = np.mean(vals)
                    # 方差列保留4位小数以防全变0，其他保留2位
                    row[k] = f"{avg_val:.4f}" if "Std" in k else f"{avg_val:.2f}"
                else:
                    row[k] = "-"
            results.append(row)

    # 打印排版并导出 CSV
    if results:
        # 新增了 'Last10 PM Mean' 这一列
        columns_order = [
            'Dataset', 'Best GM', 'Final GM', 'Last10 GM Mean', 'Last10 GM Std', 
            'Best PM', 'Final PM', 'Last10 PM Mean', 'Worst PM', 'Client Std', 'Time/Round'
        ]
        df = pd.DataFrame(results, columns=columns_order)
        print(df.to_markdown(index=False, stralign="center", numalign="center"))
        print("="*145)
        
        output_csv = 'fedpha_comprehensive_3seeds.csv'
        df.to_csv(output_csv, index=False)
        print(f"\n✅ 大满贯表格已成功保存至: {output_csv}")
    else:
        print("\n❌ 未能提取到有效数据，请检查日志目录。")

if __name__ == '__main__':
    main()