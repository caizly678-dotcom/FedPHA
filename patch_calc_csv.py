import os
import glob
import pandas as pd
import numpy as np

def analyze_csv(file_path):
    try:
        # 读取 CSV 文件
        df = pd.read_csv(file_path)
        
        # 1. 自动寻找代表“准确率”的列（忽略大小写）
        acc_col = None
        for col in df.columns:
            if 'acc' in col.lower() or 'accuracy' in col.lower():
                acc_col = col
                break
        
        # 如果没有名为 acc 的列，默认取最后一列数值列
        if acc_col is None:
            acc_col = df.select_dtypes(include=[np.number]).columns[-1]

        # 提取准确率数据并去除空值
        accs = df[acc_col].dropna().values
        if len(accs) == 0:
            return None
            
        # 2. 自动寻找代表“轮数/Epoch”的列
        round_col = None
        for col in df.columns:
            if 'round' in col.lower() or 'epoch' in col.lower() or 'step' in col.lower():
                round_col = col
                break
        
        # 如果没有 round 列，则默认按行号 (1, 2, 3...) 作为轮数
        rounds = df[round_col].values if round_col else np.arange(1, len(accs) + 1)

        # 3. 计算核心指标
        final_acc = accs[-1]
        
        # 提取最后10轮，如果总轮数不足10轮，则取所有轮
        last10 = accs[-10:] if len(accs) >= 10 else accs
        last10_avg = np.mean(last10)
        # 使用 ddof=1 计算样本标准差
        last10_std = np.std(last10, ddof=1) if len(last10) > 1 else 0.0 
        
        # 寻找最佳轮数及其对应的准确率
        best_idx = np.argmax(accs)
        best_acc = accs[best_idx]
        best_round = rounds[best_idx]

        return {
            'File': os.path.basename(file_path),
            'Total_Rounds': len(accs),
            'Final_Acc': final_acc,
            'Best_Acc': best_acc,
            'Best_Round': best_round,
            'Last10_Avg': last10_avg,
            'Last10_Std': last10_std
        }
    except Exception as e:
        print(f"❌ 解析 {os.path.basename(file_path)} 失败: {e}")
        return None

def main():
    # 指定的目标文件夹路径
    target_dir = "/workspace/FedPHA/output/dtd/GL_SVDMSE/shot_16/beta_0.5/ep1_r50/alpha1.0_ratio0.8/seed_42"
    
    # 递归查找目录下所有包含 acc 的 .csv 文件
    search_pattern = os.path.join(target_dir, "**", "*acc*.csv")
    csv_files = glob.glob(search_pattern, recursive=True)
    
    # 如果没找到带 acc 的文件，退而求其次找所有 csv
    if not csv_files:
        csv_files = glob.glob(os.path.join(target_dir, "**", "*.csv"), recursive=True)

    if not csv_files:
        print(f"⚠️ 在目录 '{target_dir}' 下没有找到任何 CSV 文件！")
        return

    print(f"🔍 找到 {len(csv_files)} 个 CSV 文件，开始计算指标...\n")
    
    results = []
    for file in sorted(csv_files):
        metrics = analyze_csv(file)
        if metrics:
            results.append(metrics)
            
    if not results:
        return

    # 使用 Pandas 打印漂亮的表格
    results_df = pd.DataFrame(results)
    
    # 调整列的顺序和格式
    results_df = results_df[['File', 'Total_Rounds', 'Last10_Avg', 'Last10_Std', 'Best_Acc', 'Best_Round', 'Final_Acc']]
    
    # 打印到控制台
    print("-" * 110)
    # 将浮点数格式化为保留两位小数，方便阅读
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    # 打印时不对齐换行
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(results_df.to_string(index=False))
    print("-" * 110)
    
    # 可选：将结果汇总保存为一个新的 CSV 文件到原目录
    save_path = os.path.join(target_dir, "summary_metrics.csv")
    results_df.to_csv(save_path, index=False)
    print(f"\n✅ 汇总结果已保存至: {save_path}")

if __name__ == "__main__":
    main()