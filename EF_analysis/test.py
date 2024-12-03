import numpy as np
import pandas as pd
from pyod.models.iforest import IForest
from pyod.utils.data import evaluate_print
import matplotlib.pyplot as plt

def main():
    # 1. 生成示例行为数据
    # 假设我们有两列行为特征数据，使用随机数进行模拟
    data = np.random.randn(100, 2)

    # 添加一些异常点
    outliers = np.array([[3, 3], [4, 4], [5, 5]])
    data = np.vstack([data, outliers])

    # 将数据转换为DataFrame，便于查看
    df = pd.DataFrame(data, columns=['feature1', 'feature2'])
    print("行为数据：")
    print(df.head())

    # 2. 使用Isolation Forest模型进行异常值检测
    clf = IForest(contamination=0.5)  # contamination表示数据中异常值的大致比例
    clf.fit(data)

    # 3. 进行预测，1表示正常，-1表示异常
    y_pred = clf.predict(data)  # 获取预测的标签 (-1表示异常, 1表示正常)
    y_scores = clf.decision_scores_  # 获取每个数据点的异常得分

    # 4. 输出预测结果
    df['label'] = y_pred
    print("\n预测结果：")
    print(df)

    # 5. 可视化异常值检测结果
    plt.scatter(data[y_pred == 1, 0], data[y_pred == 1, 1], c='b', label='Normal')
    plt.scatter(data[y_pred == -1, 0], data[y_pred == -1, 1], c='r', label='Anomaly', marker='x')
    plt.title('Isolation Forest Outlier Detection')
    plt.xlabel('feature1')
    plt.ylabel('feature2')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
