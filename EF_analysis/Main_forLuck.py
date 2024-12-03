import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from itertools import product
from EF_analysis import EFAnalyzer  # 假设EFAnalyzer定义在EF_analysis.py中
from itertools import product
from tqdm import tqdm
import csv


def get_matching_files(folder1, folder2):
    def extract_key(filename):
        parts = filename.split("_")
        return f"{parts[-2]}_{parts[-1]}" if len(parts) >= 2 else filename

    files1 = {extract_key(f): f for f in os.listdir(folder1) if f.endswith(".xlsx")}
    files2 = {extract_key(f): f for f in os.listdir(folder2) if f.endswith(".xlsx")}
    matched_files = [(os.path.join(folder1, files1[key]), os.path.join(folder2, files2[key]))
                     for key in files1.keys() & files2.keys()]
    return matched_files


def analyze_data(sham_file, active_file, sheet_name, method, contamination, lower_threshold, upper_threshold, std_devs,
                 task):
    def load_metrics(file_path, sheet_name, method, contamination, lower_threshold, upper_threshold, std_devs):
        analyzer = EFAnalyzer(
            directory_path=os.path.dirname(file_path),
            file_name=os.path.basename(file_path),
            sheet_name=sheet_name,
            method=method,
            contamination=contamination,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            std_devs=std_devs,
            task=task
        )
        try:
            data = analyzer.load_data()
            if data is not None:
                analyzer.mark_outliers()
                return analyzer.calculate_metrics()
            else:
                return {}
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
            return {}

    metrics_active = load_metrics(active_file, sheet_name, method, contamination, lower_threshold, upper_threshold,
                                  std_devs)
    metrics_sham = load_metrics(sham_file, sheet_name, method, contamination, lower_threshold, upper_threshold,
                                std_devs)

    return metrics_active, metrics_sham


def perform_statistical_tests(all_results_active, all_results_sham):
    active_df = pd.DataFrame(all_results_active)
    sham_df = pd.DataFrame(all_results_sham)
    stats = {}
    for metric in active_df.columns:
        if metric in sham_df.columns:
            active_vals = active_df[metric].dropna()
            sham_vals = sham_df[metric].dropna()
            if len(active_vals) > 1 and len(sham_vals) > 1:
                t_stat, p_value = ttest_ind(active_vals, sham_vals, nan_policy='omit')
                stats[metric] = {"t_stat": t_stat, "p_value": p_value}
            else:
                stats[metric] = {"t_stat": np.nan, "p_value": np.nan}
    return stats


def main():
    sham_folder = r"C:\Users\lgh\Desktop\脑计划交差\齐鲁tPBM_ADHD\EF_behavior\Sham"
    active_folder = r"C:\Users\lgh\Desktop\脑计划交差\齐鲁tPBM_ADHD\EF_behavior\Active"
    matched_files = get_matching_files(sham_folder, active_folder)

    if not matched_files:
        print("没有匹配的文件，跳过分析。")
        return

    sheet_names = ["EmotionSwitch"]
    task = "Switch"

    methods = ["lof", "abod", "threshold", "std"]
    contaminations = [0.1, 0.2, 0.3, 0.4, 0.5]
    lower_thresholds = [0.2, 0.3, 0.4, 0.5]
    upper_thresholds = [1.5, 1.8, 2.0, 2.2, 2.5]
    std_devs = [2, 2.5, 3.0]


    results = []
    max_attempts = 1200
    attempt_count = 0

    # 计算总尝试数（用于进度条）
    total_combinations = len(sheet_names) * len(methods) * len(contaminations) * len(lower_thresholds) * len(
        upper_thresholds) * len(std_devs)

    with tqdm(total=total_combinations, desc="参数搜索进度", unit="combination") as pbar:
        for sheet_name, method, contamination, lower_threshold, upper_threshold, std_dev in product(
                sheet_names, methods, contaminations, lower_thresholds, upper_thresholds, std_devs
        ):
            if attempt_count >= max_attempts:
                break

            all_results_active = []
            all_results_sham = []
            for sham_file, active_file in matched_files:
                metrics_active, metrics_sham = analyze_data(
                    sham_file=sham_file,
                    active_file=active_file,
                    sheet_name=sheet_name,
                    method=method,
                    contamination=contamination,
                    lower_threshold=lower_threshold,
                    upper_threshold=upper_threshold,
                    std_devs=std_dev,
                    task=task
                )

                if metrics_active:
                    all_results_active.extend(metrics_active.values())
                if metrics_sham:
                    all_results_sham.extend(metrics_sham.values())

            stats = perform_statistical_tests(all_results_active, all_results_sham)

            for metric, values in stats.items():
                if values["p_value"] < 0.1:
                    results.append({
                        "sheet_name": sheet_name,
                        "method": method,
                        "contamination": contamination,
                        "lower_threshold": lower_threshold,
                        "upper_threshold": upper_threshold,
                        "std_dev": std_dev,
                        "metric": metric,
                        "t_stat": values["t_stat"],
                        "p_value": values["p_value"]
                    })

            attempt_count += 1
            pbar.update(1)  # 更新进度条

    # 保存结果到 CSV 文件
    if results:
        with open("significant_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "sheet_name", "method", "contamination", "lower_threshold",
                "upper_threshold", "std_dev", "metric", "t_stat", "p_value"
            ])
            writer.writeheader()
            writer.writerows(results)

        print("分析完成，结果已保存到 significant_results.csv")

    else:
        print("没有显著结果。")


if __name__ == "__main__":
    main()
