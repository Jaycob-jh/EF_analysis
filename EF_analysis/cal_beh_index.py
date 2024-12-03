# for N-back, visual working memory task analysis
# BY Chenguang918 2024/11/1

import pandas as pd
import numpy as np
from scipy.stats import norm

def calculate_metrics_wm(correct_responses, participant_responses, reaction_times, clean_flag,
                         condition_list=None, include_all_trials=True, true_value=1, false_value=0):
    """
    计算工作记忆任务的各种表现指标，支持不同条件的分组计算，支持根据clean_flag排除不符合标准的试次

    参数:
    - correct_responses: np.array, 正确答案
    - participant_responses: np.array, 参与者的回答
    - reaction_times: np.array, 每个试次的反应时间
    - clean_flag: np.array, 由0和1组成，1代表纳入分析，0表示将试次排除分析
    - condition_list: np.array, 每个试次的条件（整数或类别），可选
    - include_all_trials: bool, 是否计算所有试次的反应时间和标准差
    - true_value: 任意数据类型，定义命中（“True”）的值，默认为1
    - false_value: 任意数据类型，定义虚报（“False”）的值，默认为0

    返回:
    - dict, 各项计算结果，包括不同条件下的准确率、反应时间、d'等
    """
    if not (len(correct_responses) == len(participant_responses) == len(reaction_times) == len(clean_flag)):
        raise ValueError("输入数组的长度必须相同")

    # 如果没有提供condition_list，默认所有试次属于同一条件
    if condition_list is None:
        condition_list = np.ones_like(correct_responses)

    # 获取所有的唯一条件
    unique_conditions = np.unique(condition_list)

    # 用字典存储每个条件的结果
    results = {}

    # 遍历每个条件，计算对应指标
    for condition in unique_conditions:
        # 筛选出属于当前条件且符合clean_flag的试次
        condition_mask = (condition_list == condition) & (clean_flag == 1)
        correct_responses_cond = correct_responses[condition_mask]
        participant_responses_cond = participant_responses[condition_mask]
        reaction_times_cond = reaction_times[condition_mask]

        # 计算准确率 (acc)
        correct_trials = (correct_responses_cond == participant_responses_cond)
        acc = np.mean(correct_trials) if correct_trials.size > 0 else np.nan

        # 计算反应时间 (rt) 和反应时间标准差 (rtsd)
        if include_all_trials:
            rt = np.mean(reaction_times_cond) if reaction_times_cond.size > 0 else np.nan
            rtsd = np.std(reaction_times_cond) if reaction_times_cond.size > 0 else np.nan
        else:
            correct_rt = reaction_times_cond[correct_trials]
            rt = np.mean(correct_rt) if correct_rt.size > 0 else np.nan
            rtsd = np.std(correct_rt) if correct_rt.size > 0 else np.nan

        # 计算 d' (敏感性指数)
        hits = np.sum((correct_responses_cond == true_value) & (participant_responses_cond == true_value))
        misses = np.sum((correct_responses_cond == true_value) & (participant_responses_cond == false_value))
        false_alarms = np.sum((correct_responses_cond == false_value) & (participant_responses_cond == true_value))
        correct_rejections = np.sum((correct_responses_cond == false_value) & (participant_responses_cond == false_value))

        # 计算命中率和虚报率
        hit_rate = hits / (hits + misses) if hits + misses > 0 else 0.5
        false_alarm_rate = false_alarms / (false_alarms + correct_rejections) if false_alarms + correct_rejections > 0 else 0.5

        # 调整命中率和虚报率，防止无限z分数
        hit_rate = np.clip(hit_rate, 1e-5, 1 - 1e-5)
        false_alarm_rate = np.clip(false_alarm_rate, 1e-5, 1 - 1e-5)

        # 计算 d'
        d_prime = norm.ppf(hit_rate) - norm.ppf(false_alarm_rate)

        # 将当前条件的计算结果存储在字典中
        results[f"Condition {condition}"] = {
            "Accuracy (acc)": acc,
            "Average Reaction Time (rt)": rt,
            "Reaction Time Standard Deviation (rtsd)": rtsd,
            "d' (sensitivity index)": d_prime,
            "Hit ratio": hit_rate,
            "Correct Rejections": correct_rejections
        }

    return results


def calculate_flanker_conflict_index(data, clean_flag):
    """
    Calculate the conflict index for FLANKER task data with filtering conditions.

    Args:
        data (pd.DataFrame): Input DataFrame containing FLANKER task data.
                             Must include columns:
                             '正式阶段刺激图片/Item名', '正式阶段正确答案', '正式阶段被试按键', '相对时间(秒)', clean_flag column.
        clean_flag (str): The name of the column indicating valid trials (e.g., 1 for valid).

    Returns:
        dict: A dictionary with the conflict index based on accuracy and reaction time.
    """
    # Filter data based on clean_flag and non-None responses
    filtered_data = data[(data[clean_flag] == 1) & (data['正式阶段被试按键'].notna())].copy()

    if filtered_data.empty:
        raise ValueError("No valid trials after filtering. Check your data or filter criteria.")

    # Extract orientations from the stimulus names
    filtered_data['环境朝向'] = filtered_data['正式阶段刺激图片/Item名'].str[8]
    filtered_data['目标朝向'] = filtered_data['正式阶段刺激图片/Item名'].str[9]

    # Define conditions
    filtered_data['条件'] = filtered_data.apply(
        lambda row: '一致' if row['环境朝向'] == row['目标朝向'] else '冲突',
        axis=1
    )

    # Compute accuracy per condition
    filtered_data['正确'] = filtered_data['正式阶段正确答案'] == filtered_data['正式阶段被试按键']
    accuracy_by_condition = filtered_data.groupby('条件')['正确'].mean() * 100
    reaction_time_by_condition = filtered_data.groupby('条件')['相对时间(秒)'].mean()

    # Calculate conflict index
    conflict_index = {
        '正确率差异 (%)': accuracy_by_condition.get('一致', 0) - accuracy_by_condition.get('冲突', 0),
        '反应时间差异 (秒)': reaction_time_by_condition.get('冲突', 0) - reaction_time_by_condition.get('一致', 0)
    }

    return conflict_index

def calculate_metrics_sst(correct_responses, participant_responses, reaction_times,SSRT,clean_flag,
                         condition_list=None, include_all_trials=True, true_value=["Right", "Left"], false_value=None):
    """
    计算sst任务的各种表现指标，支持不同条件的分组计算，支持根据clean_flag排除不符合标准的试次

    参数:
    - correct_responses: np.array, 正确答案
    - participant_responses: np.array, 参与者的回答
    - reaction_times: np.array, 每个试次的反应时间
    - clean_flag: np.array, 由0和1组成，1代表纳入分析，0表示将试次排除分析
    - condition_list: np.array, 每个试次的条件（整数或类别），可选，
    - include_all_trials: bool, 是否计算所有试次的反应时间和标准差
    - true_value: 任意数据类型，定义命中（“True”）的值，默认为1
    - false_value: 任意数据类型，定义虚报（“False”）的值，默认为0

    返回:
    - dict, 各项计算结果，包括不同条件下的准确率、反应时间、d'等
    """
    if not (len(correct_responses) == len(participant_responses) == len(reaction_times) == len(clean_flag)):
        raise ValueError("输入数组的长度必须相同")

    # 如果没有提供condition_list，默认所有试次属于同一条件
    if condition_list is None:
        condition_list = np.ones_like(correct_responses)

    # 获取所有的唯一条件
    unique_conditions = np.unique(condition_list)

    # 用字典存储每个条件的结果
    results = {}

    # 遍历每个条件，计算对应指标
    for condition in unique_conditions:
        # 筛选出属于当前条件且符合clean_flag的试次
        condition_mask = (condition_list == condition) & (clean_flag == 1)
        correct_responses_cond = correct_responses[condition_mask]
        participant_responses_cond = participant_responses[condition_mask]
        reaction_times_cond = reaction_times[condition_mask]

        # 计算准确率 (acc)
        correct_trials = (correct_responses_cond == participant_responses_cond)
        acc = np.mean(correct_trials) if correct_trials.size > 0 else np.nan

        # 计算反应时间 (rt) 和反应时间标准差 (rtsd)
        if include_all_trials:
            rt = np.mean(reaction_times_cond) if reaction_times_cond.size > 0 else np.nan
            rtsd = np.std(reaction_times_cond) if reaction_times_cond.size > 0 else np.nan
        else:
            correct_rt = reaction_times_cond[correct_trials]
            rt = np.mean(correct_rt) if correct_rt.size > 0 else np.nan
            rtsd = np.std(correct_rt) if correct_rt.size > 0 else np.nan

        # 计算 d' (敏感性指数)
        hits = np.sum(np.isin(correct_responses_cond, true_value) & (participant_responses_cond == correct_responses_cond))
        misses = np.sum(np.isin(correct_responses_cond, true_value) & (participant_responses_cond != correct_responses_cond))
        false_alarms = np.sum((correct_responses_cond == false_value) & (participant_responses_cond != false_value))
        correct_rejections = np.sum((correct_responses_cond == false_value) & (participant_responses_cond == false_value))

        # 计算命中率和虚报率
        hit_rate = hits / (hits + misses) if hits + misses > 0 else 0.5
        false_alarm_rate = false_alarms / (false_alarms + correct_rejections) if false_alarms + correct_rejections > 0 else 0.5

        # 调整命中率和虚报率，防止无限z分数
        hit_rate = np.clip(hit_rate, 1e-5, 1 - 1e-5)
        false_alarm_rate = np.clip(false_alarm_rate, 1e-5, 1 - 1e-5)

        # 计算 d'
        d_prime = norm.ppf(hit_rate) - norm.ppf(false_alarm_rate)

        # 将数据转换为浮点类型，并将非数值替换为 np.nan
        ssrt = pd.to_numeric(SSRT, errors='coerce')  # 将非数值类型替换为 np.nan

        # 筛选最后 10 个非空值
        valid_values = ssrt[-10:]  # 获取最后 10 个值
        valid_values = valid_values.dropna()  # 去除空值

        # 计算平均值
        mean_ssrt = valid_values.mean()

        # 将当前条件的计算结果存储在字典中
        results[f"Condition {condition}"] = {
            "Accuracy (acc)": acc,
            "Average Reaction Time (rt)": rt,
            "Reaction Time Standard Deviation (rtsd)": rtsd,
            "d' (sensitivity index)": d_prime,
            "mean_ssrt": mean_ssrt
        }

    return results

def calculate_metrics_gng(correct_responses, participant_responses, reaction_times, clean_flag,
                         condition_list=None, include_all_trials=True, true_value=1, false_value=0):
    """
    Wrapper for calculate_metrics_wm with a different name.
    Accepts the same arguments as calculate_metrics_wm.
    """
    return calculate_metrics_wm(correct_responses, participant_responses, reaction_times, clean_flag,
                         condition_list=None, include_all_trials=True, true_value=1, false_value=0)


def calculate_metrics_flanker(correct_responses, participant_responses, reaction_times, clean_flag,
                         condition_list=None, include_all_trials=True, true_value=1, false_value=0):
    """
    计算Flanker任务的各种表现指标，支持不同条件的分组计算，支持根据clean_flag排除不符合标准的试次
    ****请将 data['正式阶段刺激图片/Item名'] 作为 condition_list 进行输入****
    参数:
    - correct_responses: np.array, 正确答案
    - participant_responses: np.array, 参与者的回答
    - reaction_times: np.array, 每个试次的反应时间
    - clean_flag: np.array, 由0和1组成，1代表纳入分析，0表示将试次排除分析
    - condition_list: np.array, 每个试次的条件（整数或类别），可选，此任务中为data['正式阶段刺激图片/Item名']***
    - include_all_trials: bool, 是否计算所有试次的反应时间和标准差
    - true_value: 任意数据类型，定义命中（“True”）的值，默认为1
    - false_value: 任意数据类型，定义虚报（“False”）的值，默认为0

    返回:
    - dict, 各项计算结果，包括不同条件下的准确率、反应时间、d'等
    """
    if not (len(correct_responses) == len(participant_responses) == len(reaction_times) == len(clean_flag)):
        raise ValueError("输入数组的长度必须相同")

    # 如果没有提供condition_list，默认所有试次属于同一条件
    if condition_list is None:
        condition_list = np.ones_like(correct_responses)

    condition_list = np.array([1 if s[-2] == s[-1] else 2 for s in condition_list])

    # 获取所有的唯一条件
    unique_conditions = np.unique(condition_list)

    # 用字典存储每个条件的结果
    results = {}

    # 遍历每个条件，计算对应指标
    for condition in unique_conditions:
        # 筛选出属于当前条件且符合clean_flag的试次
        condition_mask = (condition_list == condition) & (clean_flag == 1)
        correct_responses_cond = correct_responses[condition_mask]
        participant_responses_cond = participant_responses[condition_mask]
        reaction_times_cond = reaction_times[condition_mask]

        # 计算准确率 (acc)
        correct_trials = (correct_responses_cond == participant_responses_cond)
        acc = np.mean(correct_trials) if correct_trials.size > 0 else np.nan

        # 计算反应时间 (rt) 和反应时间标准差 (rtsd)
        if include_all_trials:
            rt = np.mean(reaction_times_cond) if reaction_times_cond.size > 0 else np.nan
            rtsd = np.std(reaction_times_cond) if reaction_times_cond.size > 0 else np.nan
        else:
            correct_rt = reaction_times_cond[correct_trials]
            rt = np.mean(correct_rt) if correct_rt.size > 0 else np.nan
            rtsd = np.std(correct_rt) if correct_rt.size > 0 else np.nan

        # 将当前条件的计算结果存储在字典中
        results[f"Condition {condition}"] = {
            "Accuracy (acc)": acc,
            "Average Reaction Time (rt)": rt,
            "Reaction Time Standard Deviation (rtsd)": rtsd,
        }
    # 获取前两个条件的键
    condition_keys = list(results.keys())[:2]
    condition_1 = results[condition_keys[0]]
    condition_2 = results[condition_keys[1]]

    # 计算新条件的值（前两个条件相减）
    results[f"Condition {3}"] = {
        "Accuracy_diff (acc)": condition_1["Accuracy (acc)"] - condition_2["Accuracy (acc)"],
        "Average Reaction Time_diff (rt)": condition_1["Average Reaction Time (rt)"] - condition_2["Average Reaction Time (rt)"],
        "Reaction Time Standard Deviation_diff (rtsd)": condition_1["Reaction Time Standard Deviation (rtsd)"] - condition_2["Reaction Time Standard Deviation (rtsd)"],
    }
    return results

def calculate_metrics_stroop(correct_responses, participant_responses, reaction_times, clean_flag,
                         condition_list=None, include_all_trials=True, true_value=1, false_value=0):
    """
    计算Stroop任务的各种表现指标，支持不同条件的分组计算，支持根据clean_flag排除不符合标准的试次
    ****请将 data['正式阶段刺激图片/Item名'] 作为 condition_list 进行输入****
    参数:
    - correct_responses: np.array, 正确答案
    - participant_responses: np.array, 参与者的回答
    - reaction_times: np.array, 每个试次的反应时间
    - clean_flag: np.array, 由0和1组成，1代表纳入分析，0表示将试次排除分析
    - condition_list: np.array, 每个试次的条件（整数或类别），可选，此任务中为data['正式阶段刺激图片/Item名']***
    - include_all_trials: bool, 是否计算所有试次的反应时间和标准差
    - true_value: 任意数据类型，定义命中（“True”）的值，默认为1
    - false_value: 任意数据类型，定义虚报（“False”）的值，默认为0
    - condition1: congruent Condition2: incongruent
    返回:
    - dict, 各项计算结果，包括不同条件下的准确率、反应时间、d'等
    """
    if not (len(correct_responses) == len(participant_responses) == len(reaction_times) == len(clean_flag)):
        raise ValueError("输入数组的长度必须相同")

    # 如果没有提供condition_list，默认所有试次属于同一条件
    if condition_list is None:
        condition_list = np.ones_like(correct_responses)

    # 判断语句
    if all("_" in s for s in condition_list):  # 判断是否属于第一种类型
        condition_list = np.array([
            1 if s.split('_')[1] == s.split('_')[3] else 2
            for s in condition_list
        ])
    else:  # 如果不是第一种类型，按照第二种方案
        condition_list = np.array([
            1 if int(s[1:]) % 4 == 0 else 2
            for s in condition_list
        ])

    # 获取所有的唯一条件
    unique_conditions = np.unique(condition_list)

    # 用字典存储每个条件的结果
    results = {}

    # 遍历每个条件，计算对应指标
    for condition in unique_conditions:
        # 筛选出属于当前条件且符合clean_flag的试次
        condition_mask = (condition_list == condition) & (clean_flag == 1)
        correct_responses_cond = correct_responses[condition_mask]
        participant_responses_cond = participant_responses[condition_mask]
        reaction_times_cond = reaction_times[condition_mask]

        # 计算准确率 (acc)
        correct_trials = (correct_responses_cond == participant_responses_cond)
        acc = np.mean(correct_trials) if correct_trials.size > 0 else np.nan

        # 计算反应时间 (rt) 和反应时间标准差 (rtsd)
        if include_all_trials:
            rt = np.mean(reaction_times_cond) if reaction_times_cond.size > 0 else np.nan
            rtsd = np.std(reaction_times_cond) if reaction_times_cond.size > 0 else np.nan
        else:
            correct_rt = reaction_times_cond[correct_trials]
            rt = np.mean(correct_rt) if correct_rt.size > 0 else np.nan
            rtsd = np.std(correct_rt) if correct_rt.size > 0 else np.nan

        # 将当前条件的计算结果存储在字典中
        results[f"Condition {condition}"] = {
            "Accuracy (acc)": acc,
            "Average Reaction Time (rt)": rt,
            "Reaction Time Standard Deviation (rtsd)": rtsd,
        }
    # 获取前两个条件的键
    condition_keys = list(results.keys())[:2]
    condition_1 = results[condition_keys[0]]
    condition_2 = results[condition_keys[1]]

    # 计算新条件的值（前两个条件相减）
    results[f"Condition {3}"] = {
        "Accuracy_diff (acc)": condition_1["Accuracy (acc)"] - condition_2["Accuracy (acc)"],
        "Average Reaction Time_diff (rt)": condition_1["Average Reaction Time (rt)"] - condition_2["Average Reaction Time (rt)"],
        "Reaction Time Standard Deviation_diff (rtsd)": condition_1["Reaction Time Standard Deviation (rtsd)"] - condition_2["Reaction Time Standard Deviation (rtsd)"],
    }
    return results



def calculate_metrics_switch(correct_responses, participant_responses, reaction_times, clean_flag,
                         condition_list=None, include_all_trials=True, true_value=1, false_value=0):
    """
    计算Switch任务的各种表现指标，支持不同条件的分组计算，支持根据clean_flag排除不符合标准的试次
    ****此任务中前一半为条件1，后一半为条件2，不需要额外添加condition_list****
    参数:
    - correct_responses: np.array, 正确答案
    - participant_responses: np.array, 参与者的回答
    - reaction_times: np.array, 每个试次的反应时间
    - clean_flag: np.array, 由0和1组成，1代表纳入分析，0表示将试次排除分析
    - condition_list: np.array, 每个试次的条件（整数或类别），可选，此任务中前一半为条件1，后一半为条件2，不需要额外添加condition_list
    - include_all_trials: bool, 是否计算所有试次的反应时间和标准差
    - true_value: 任意数据类型，定义命中（“True”）的值，默认为1
    - false_value: 任意数据类型，定义虚报（“False”）的值，默认为0

    返回:
    - dict, 各项计算结果，包括不同条件下的准确率、反应时间、d'等
    """
    if not (len(correct_responses) == len(participant_responses) == len(reaction_times) == len(clean_flag)):
        raise ValueError("输入数组的长度必须相同")

    # 如果没有提供condition_list，默认所有试次属于同一条件
    if condition_list is None:
        condition_list = np.ones_like(correct_responses)

    # 获取数组长度
    n = len(correct_responses)

    # 前一半标记为 1，后一半标记为 2
    condition_list[:n // 2] = 1
    condition_list[n // 2:] = 2

    # 获取所有的唯一条件
    unique_conditions = np.unique(condition_list)

    # 用字典存储每个条件的结果
    results = {}

    # 遍历每个条件，计算对应指标
    for condition in unique_conditions:
        # 筛选出属于当前条件且符合clean_flag的试次
        condition_mask = (condition_list == condition) & (clean_flag == 1)
        correct_responses_cond = correct_responses[condition_mask]
        participant_responses_cond = participant_responses[condition_mask]
        reaction_times_cond = reaction_times[condition_mask]

        # 计算准确率 (acc)
        correct_trials = (correct_responses_cond == participant_responses_cond)
        acc = np.mean(correct_trials) if correct_trials.size > 0 else np.nan

        # 计算反应时间 (rt) 和反应时间标准差 (rtsd)
        if include_all_trials:
            rt = np.mean(reaction_times_cond) if reaction_times_cond.size > 0 else np.nan
            rtsd = np.std(reaction_times_cond) if reaction_times_cond.size > 0 else np.nan
        else:
            correct_rt = reaction_times_cond[correct_trials]
            rt = np.mean(correct_rt) if correct_rt.size > 0 else np.nan
            rtsd = np.std(correct_rt) if correct_rt.size > 0 else np.nan

        # 将当前条件的计算结果存储在字典中
        results[f"Condition {condition}"] = {
            "Accuracy (acc)": acc,
            "Average Reaction Time (rt)": rt,
            "Reaction Time Standard Deviation (rtsd)": rtsd,
        }
    # 获取前两个条件的键
    condition_keys = list(results.keys())[:2]
    condition_1 = results[condition_keys[0]]
    condition_2 = results[condition_keys[1]]

    # 计算新条件的值（前两个条件相减）
    results[f"Condition {3}"] = {
        "Accuracy_diff (acc)": condition_1["Accuracy (acc)"] - condition_2["Accuracy (acc)"],
        "Average Reaction Time_diff (rt)": condition_1["Average Reaction Time (rt)"] - condition_2["Average Reaction Time (rt)"],
        "Reaction Time Standard Deviation_diff (rtsd)": condition_1["Reaction Time Standard Deviation (rtsd)"] - condition_2["Reaction Time Standard Deviation (rtsd)"],
    }
    return results


# 测试数据
correct_responses = np.array([True, True, False, True, False, False, True, False, True, False, True, True, False, True, False, False, True, False, True, False])
participant_responses = np.array([True, False, False, True, True, False, True, False, False, True, True, False, False, True, True, False, True, False, False, True])
reaction_times = np.array([1.2, 1.5, 1.3, 1.7, 1.8, 1.6, 1.4, 1.9, 1.1, 2.0, 1.2, 1.5, 1.3, 1.7, 1.8, 1.6, 1.4, 1.9, 1.1, 2.0])
condition_list = np.array([1, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 1, 2, 1, 2, 1, 2])
clean_flag = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])

# 运行测试函数
def main():
    print("Testing with all trials (include_all_trials=True):\n")
    result_all_trials = calculate_metrics_wm(correct_responses, participant_responses, reaction_times, clean_flag, condition_list, include_all_trials=True, true_value=True, false_value=False)
    for condition, metrics in result_all_trials.items():
        print(f"{condition}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
        print()

    print("Testing with correct trials only (include_all_trials=False):\n")
    result_correct_trials = calculate_metrics_wm(correct_responses, participant_responses, reaction_times, clean_flag, condition_list, include_all_trials=False, true_value=True, false_value=False)
    for condition, metrics in result_correct_trials.items():
        print(f"{condition}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
        print()

if __name__ == "__main__":
    main()
