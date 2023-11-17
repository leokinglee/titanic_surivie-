import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, precision_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler

# # 训练集
# 读取数据
train_titanic = pd.read_csv(r'D:\桌面\数据集\titanic\train.csv')
print(train_titanic)
# print(train_titanic.isnull().sum())

#  对非数值类型数据处理
encoder = LabelBinarizer()
feature_cat = train_titanic.Sex
train_titanic.Sex = encoder.fit_transform(feature_cat)

# 对缺失值进行处理
# 均值插补
train_titanic["Age"] = train_titanic["Age"].replace(np.NaN, train_titanic["Age"].mean())
# # LOCF - 前一次观测结果
# train_titanic["Age"] = train_titanic["Age"].fillna(method='ffill')
# print(train_titanic.isnull().sum())

# 获取特征列表
feature = train_titanic.columns.tolist()[2:]
del feature[1]
del feature[-4]
del feature[-2]
del feature[-1]
print(feature)
X_train = np.array(train_titanic.loc[:, feature])
print(X_train)
y_train = np.array(train_titanic.iloc[:, 1])
print(y_train)

# # 测试集
test_titanic = pd.read_csv(r'D:\桌面\数据集\titanic\test.csv')
print(test_titanic)
# print(test_titanic.isnull().sum())

#  对非数值类型数据处理
encoder = LabelBinarizer()
feature_cat = test_titanic.Sex
test_titanic.Sex = encoder.fit_transform(feature_cat)

feature = test_titanic.columns.tolist()[1:]
del feature[1]
del feature[-4]
del feature[-2]
del feature[-1]
print(feature)

# 对缺失值进行处理
# 均值插补
test_titanic["Age"] = test_titanic["Age"].replace(np.NaN, test_titanic["Age"].mean())
test_titanic["Fare"] = test_titanic["Fare"].replace(np.NaN, test_titanic["Fare"].mean())
# # LOCF - 前一次观测结果
# train_titanic["Age"] = train_titanic["Age"].fillna(method='ffill')
# print(test_titanic.isnull().sum())

X_test = np.array(test_titanic.loc[:, feature])
print(X_test)

# 数据标准化
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

# KNN训练模型
knn = KNeighborsClassifier(n_neighbors=3, p=2, metric="minkowski")
knn.fit(X_train, y_train)
y_pre= knn.predict(X_test)
print(y_pre)
# print(accuracy_score(y_test, y_pre))

# 验证
y_verify = pd.read_csv(r'D:\桌面\数据集\titanic\gender_submission.csv')
print(y_verify)
y_test = np.array(y_verify.loc[:, 'Survived'])
print(y_test)

# 计算模型指标
accuracy = accuracy_score(y_test, y_pre)
precision = precision_score(y_test, y_pre, average='binary')
recall = recall_score(y_test, y_pre, average='binary')
f1score = f1_score(y_test, y_pre, average='binary')
auc = roc_auc_score(y_test, y_pre)
print(f'accuracy_score:{accuracy}')
print(f'precision_score:{precision}')
print(f'recall_score:{recall}')
print(f'f1score:{f1score}')
print(f"AUC:{auc}")