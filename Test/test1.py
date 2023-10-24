import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    breast_cancer = load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target
    # 预处理
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, random_state=1, test_size=0.2)
    preDealData = StandardScaler()  # 使数据符合正态分布
    XTrain = preDealData.fit_transform(XTrain)
    XTest = preDealData.fit_transform(XTest)
    # 建立逻辑回归模型
    lr = LogisticRegression()
    lr.fit(XTrain, yTrain)
    # 预测
    yPredict = lr.predict(XTest)
    # 评估结果
    print("准确度：", lr.score(XTest, yTest))
    # 输出模型参数
    print("theta_0:", lr.intercept_)
    print("theta_1-theta_n:",lr.coef_)
