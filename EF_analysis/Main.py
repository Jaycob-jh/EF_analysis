import os
import pandas as pd
from scipy.stats import ttest_ind
from EF_analysis import EFAnalyzer  # 假设EFAnalyzer定义在EF_analysis.py中


def get_matching_files(folder1, folder2):
    # 提取文件名的唯一标识符为“姓名_任务名称”
    def extract_key(filename):
        parts = filename.split("_")
        return f"{parts[-2]}_{parts[-1]}" if len(parts) >= 2 else filename

    files1 = {extract_key(f): f for f in os.listdir(folder1) if f.endswith(".xlsx")}
    files2 = {extract_key(f): f for f in os.listdir(folder2) if f.endswith(".xlsx")}

    # 使用唯一标识符进行匹配
    matched_files = [(os.path.join(folder1, files1[key]), os.path.join(folder2, files2[key]))
                     for key in files1.keys() & files2.keys()]
    return matched_files


def analyze_data(sham_file, active_file, sheet_name="Number2Back", method="lof"):
    # 使用完整文件路径
    def load_metrics(file_path, sheet_name, method):
        analyzer = EFAnalyzer(directory_path=os.path.dirname(file_path),
                              file_name=os.path.basename(file_path),
                              sheet_name=sheet_name,
                              method=method)
        try:
            data = analyzer.load_data()
            if data is not None:
                analyzer.mark_outliers()
                return analyzer.calculate_metrics()
            else:
                print(f"Sheet '{sheet_name}' not found in {file_path}. Metrics set to empty.")
                return {}  # 如果没有该 sheet，返回空字典
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
            return {}

    metrics_active = load_metrics(active_file, sheet_name, method)
    metrics_sham = load_metrics(sham_file, sheet_name, method)

    return metrics_active, metrics_sham


def perform_statistical_tests(all_results_active, all_results_sham):
    # 将所有被试的结果合并到 DataFrame 中
    active_df = pd.DataFrame(all_results_active)
    sham_df = pd.DataFrame(all_results_sham)

    # 执行统计检验
    stats = {}
    for metric in active_df.columns:
        if metric in sham_df.columns:
            t_stat, p_value = ttest_ind(active_df[metric].dropna(), sham_df[metric].dropna(), nan_policy='omit')
            stats[metric] = {"t_stat": t_stat, "p_value": p_value}
    return stats


def main():
    sham_folder = r"C:\Users\lgh\Desktop\脑计划交差\齐鲁tPBM_ADHD\EF_behavior\Sham"
    active_folder = r"C:\Users\lgh\Desktop\脑计划交差\齐鲁tPBM_ADHD\EF_behavior\Active"
    matched_files = get_matching_files(sham_folder, active_folder)

    if not matched_files:
        print("没有匹配的文件，跳过分析。")
        return

    all_results_active = []
    all_results_sham = []
    for sham_file, active_file in matched_files:
        metrics_active, metrics_sham = analyze_data(
            sham_file=sham_file, active_file=active_file, sheet_name="Number2Back", method="threshold"
        )

        # 收集所有被试的结果
        if metrics_active:
            for condition, metrics in metrics_active.items():
                all_results_active.append(metrics)
        if metrics_sham:
            for condition, metrics in metrics_sham.items():
                all_results_sham.append(metrics)

    # 统计所有被试的结果
    stats = perform_statistical_tests(all_results_active, all_results_sham)

    # 输出统计结果
    print("Overall Statistical Results:")
    for metric, values in stats.items():
        print(f"{metric}: t-stat={values['t_stat']}, p-value={values['p_value']}")


if __name__ == "__main__":
    main()
