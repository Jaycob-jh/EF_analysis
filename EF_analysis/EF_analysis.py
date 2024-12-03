import pandas as pd
import numpy as np
from pyod.models.iforest import IForest
import matplotlib.pyplot as plt
import cal_beh_index as cal_beh
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.abod import ABOD


class EFAnalyzer:
    def __init__(self, directory_path, file_name, sheet_name, method, contamination=0.1, lower_threshold=0.3,
                 upper_threshold=2.0, std_devs=2.5, task="flanker"):
        self.task_type = task
        self.directory_path = directory_path
        self.file_name = file_name
        self.sheet_name = sheet_name
        self.contamination = contamination
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.std_devs = std_devs
        self.data = None
        self.method = method

    def load_data(self):
        """
    Load data from an Excel file.

    :param directory_path: str, path to the directory containing the Excel file
    :param file_name: str, name of the Excel file
    :param sheet_name: str, name of the sheet to load
    :return: DataFrame containing the data from the specified sheet
    """
        file_path = f"{self.directory_path}/{self.file_name}"
        excel_data = pd.ExcelFile(file_path)
        self.data = excel_data.parse(self.sheet_name)
        return self.data

    def mark_outliers(self):
        if self.method == 'iforest':
            model = IForest(contamination=self.contamination)
            data_reshaped = self.data['相对时间(秒)'].values.reshape(-1, 1)
            model.fit(data_reshaped)
            self.data[f'iforest'] = 1 - model.labels_
        elif self.method == 'knn':
            model = KNN(contamination=self.contamination)
            data_reshaped = self.data['相对时间(秒)'].values.reshape(-1, 1)
            model.fit(data_reshaped)
            self.data[f'knn'] = 1 - model.labels_
        elif self.method == 'lof':
            model = LOF(contamination=self.contamination)
            data_reshaped = self.data['相对时间(秒)'].values.reshape(-1, 1)
            model.fit(data_reshaped)
            self.data[f'lof'] = 1 - model.labels_
        elif self.method == 'abod':
            model = ABOD(contamination=self.contamination)
            data_reshaped = self.data['相对时间(秒)'].values.reshape(-1, 1)
            model.fit(data_reshaped)
            self.data[f'abod'] = 1 - model.labels_
        elif self.method == 'threshold':
            self.data['threshold'] = self.data['相对时间(秒)'].apply(
                lambda x: 1 if self.lower_threshold <= x <= self.upper_threshold else 0)
        elif self.method == 'std':
            mean_time = self.data['相对时间(秒)'].mean()
            std_dev_time = self.data['相对时间(秒)'].std()
            lower_bound = mean_time - self.std_devs * std_dev_time
            upper_bound = mean_time + self.std_devs * std_dev_time
            self.data['std'] = self.data['相对时间(秒)'].apply(
                lambda x: 1 if lower_bound <= x <= upper_bound else 0)
        else:
            raise ValueError("Invalid method selected.")

    def calculate_metrics(self):
        """
        Dynamically calculate metrics based on task type.

        Args:
            task_type (str): Type of the task, e.g., 'N-Back' or 'Flanker'.

        Returns:
            dict: Calculated metrics based on the provided task type.
        """
        if self.task_type == 'N-Back':
            return cal_beh.calculate_metrics_wm(
                correct_responses=self.data['正式阶段正确答案'],
                participant_responses=self.data['正式阶段被试按键'],
                reaction_times=self.data['相对时间(秒)'],
                clean_flag=self.data[self.method],
                include_all_trials=True,
                true_value='Left',
                false_value='Right'
            )
        elif self.task_type == 'Flanker':
            return cal_beh.calculate_flanker_conflict_index(self.data,
                                                            self.method)
        elif self.task_type == 'SST':
            return cal_beh.calculate_metrics_sst(
                correct_responses=self.data['正式阶段正确答案'],
                SSRT=self.data['SSRT'],
                participant_responses=self.data['正式阶段被试按键'],
                reaction_times=self.data['相对时间(秒)'],
                clean_flag=self.data[self.method],
                include_all_trials=True,
                true_value='Left',
                false_value='Right'
            )
        elif self.task_type == 'GNG':
            return cal_beh.calculate_metrics_gng(
                correct_responses=self.data['正式阶段正确答案'],
                participant_responses=self.data['正式阶段被试按键'],
                reaction_times=self.data['相对时间(秒)'],
                clean_flag=self.data[self.method],
                include_all_trials=True,
                true_value='Left',
                false_value='Right'
            )
        elif self.task_type == 'Stroop':
            return cal_beh.calculate_metrics_stroop(
                correct_responses=self.data['正式阶段正确答案'],
                participant_responses=self.data['正式阶段被试按键'],
                reaction_times=self.data['相对时间(秒)'],
                clean_flag=self.data[self.method],
                condition_list=self.data['正式阶段刺激图片/Item名'],
                include_all_trials=True,
                true_value='Left',
                false_value='Right'
            )
        elif self.task_type == 'Switch':
            return cal_beh.calculate_metrics_switch(
                correct_responses=self.data['正式阶段正确答案'],
                participant_responses=self.data['正式阶段被试按键'],
                reaction_times=self.data['相对时间(秒)'],
                clean_flag=self.data[self.method],
                include_all_trials=True,
                true_value='Left',
                false_value='Right'
            )
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")


if __name__ == "__main__":
    analyzer = EFAnalyzer(directory_path="C:/Users/lgh/Desktop/脑计划交差/齐鲁tPBM_ADHD/EF_behavior/Active/",
                          file_name="ql_20240412_001_TYL_2_1唐玉卢_GameData.xlsx", sheet_name="Number2Back",
                          method="lof")
    analyzer.load_data()
    analyzer.mark_outliers()
    metrics = analyzer.calculate_metrics()
    print(metrics)
