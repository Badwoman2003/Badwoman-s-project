import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
import sys

sys.path.append("..")


def loss_func(X, y, theta):
    sumValue = 0
    for i in range(X.shape[0]):  # 遍历数为训练量
        h = sigmoid(X[i:i + 1, :], theta)[0][0]
        epsilon = 1e-5
        loss = y[i] * np.log(h + epsilon) + (1 - y[i]) * np.log(1 - h + epsilon)  # 求偏导
        sumValue += loss
    return (-1.0 / X.shape[0]) * sumValue


def sigmoid(X: np.array([]), theta: np.array([])):
    return 1.0 / (1.0 + np.exp(-1 * np.dot(X, theta.T)))


# 计算梯度
def grandient(X, y, theta):
    grand = []
    for j in range(X.shape[1]):
        sumValue = 0.0
        for i in range(X.shape[0]):
            sumValue += (y[i] - sigmoid(X[i:i + 1, :], theta)[0][0]) * X[i][j]
        grand_i = (-1.0 / X.shape[0]) * sumValue
        grand.append(grand_i)
    return np.array([grand])


iris = load_iris()
irisPd = pd.DataFrame(iris.data)
X = np.array(irisPd[[2, 3]])  # 只选用两种特征项
y = iris.target

# 标准化预处理
XTrain, XTest, yTrain, yTest = train_test_split(X, y, random_state=1, test_size=0.2)
preDealData = StandardScaler()  # 使数据符合正态分布
XTrain = preDealData.fit_transform(XTrain)
XTest = preDealData.fit_transform(XTest)

index = 0
for yValue in yTrain:
    if yValue == 2:
        yTrain[index] = 1
    index += 1
index = 0
for yValue in yTest:
    if yValue == 2:
        yTest[index] = 1
    index += 1

# 梯度下降
oldTheta = np.array([[-10, -10]])  # 调参
oldLoss = loss_func(XTrain, yTrain, oldTheta)
newTheta = np.array([[-10, -10]])
newLoss = oldLoss
alpha = 0.03  # 设置学习率
maxIter = 1000
minDiffLoss = 1e-3
for i in range(maxIter):
    grand = grandient(XTrain, yTrain, oldTheta)
    newTheta = oldTheta - alpha * grand
    newLoss = loss_func(XTrain, yTrain, newTheta)
    diffLoss = np.abs(newLoss - oldLoss)
    print(f"第{i + 1}次迭代，误差项为：{newLoss},两次迭代差为:{diffLoss}")
    if diffLoss <=minDiffLoss:
        break
    oldLoss = newLoss
    oldTheta = newTheta.copy()
print(f"参数值：{newTheta}")
