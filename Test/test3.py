from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd

iris = load_iris()
irisPd = pd.DataFrame(iris.data)
X = np.array(irisPd[[2, 3]])  # 只选用两种特征项
y = iris.target

# 预处理，不用使用standardscaler
XTrain, XTest, yTrain, yTest = train_test_split(X, y, random_state=1, test_size=0.2)

# 建立高斯分布的朴素贝叶斯并训练
gnb = GaussianNB()
gnb.fit(XTrain, yTrain)

print("各类别的先验概率：", gnb.class_prior_)
print("各类别的样本数：", gnb.class_count_)
print("特征数据在各个样本的均值：\n", gnb.theta_)
print("特征数据在各个类别的标准差：\n", gnb.var_)
yPredictProb = gnb.predict_proba(XTest[0:1, :])
print("模型预测值：", yPredictProb)


# 计算正态分布概率
def normFunc(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / sigma * np.sqrt(2 * np.pi)
    return pdf


# 计算后验概率之挤
p_xij_yki = np.ones((1, len(gnb.class_count_))).flatten()
for i in range(len(gnb.class_count_)):
    for j in range(XTest.shape[1]):
        p_xij_yki[i] *= normFunc(XTest[0, j], gnb.theta_[i, j], gnb.var_[i, j])

# 计算分母
fMu = 0
for i in range(len(gnb.class_count_)):
    fMu += gnb.class_prior_[i] * p_xij_yki[i]

# 计算各分类概率
compRate = np.zeros((1, len(gnb.class_count_))).flatten()
for i in range(len(gnb.class_count_)):
    compRate[i] = gnb.class_prior_[i]*p_xij_yki[i]/fMu
print(f"预测值为{compRate}")