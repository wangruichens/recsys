'''
Forest prediecting ---multi-class
'''
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
import pandas as pd
import numpy as np


def mergeToOne(X, X2):
    X3 = []
    for i in range(X.shape[0]):
        tmp = np.array([list(X.iloc[i]), list(X2[i])])
        X3.append(list(np.hstack(tmp)))
    X3 = np.array(X3)
    return X3


data = pd.read_csv("train.csv")
del data['Id']
# 打乱数据
data = data.sample(len(data))
y = data.Cover_Type
# X = data
X = data.drop("Cover_Type", axis=1)

# 划分训练集测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)  ##test_size测试集合所占比例

##X_train_1用于生成模型  X_train_2用于和新特征组成新训练集合
X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_train, y_train, test_size=0.6, random_state=0)

clf = XGBClassifier(
    learning_rate=0.2,  # 默认0.3
    n_estimators=1,  # 树的个数, 多类问题，每个类都会生成一个树，这里有7个类，所以会有7棵树
    max_depth=5,
    min_child_weight=10,
    gamma=0.5,
    subsample=0.75,
    colsample_bytree=0.75,
    objective='multi:softprob',  # 逻辑回归损失函数
    nthread=8,  # cpu线程数
    scale_pos_weight=1,
    reg_alpha=1e-05,
    reg_lambda=10,
    seed=1024)  # 随机种子

clf.fit(X_train_1, y_train_1)
# print(clf.feature_importances_)

new_feature = clf.apply(X_train_2)
new_y=clf.predict_proba(X_train_2)
y_pre = clf.predict(X_train_1)
# y_pro = model.predict_proba(X_test_new)[:, 1]
# print("AUC Score :", (metrics.roc_auc_score(y_test, y_pro)))
print("Accuracy :", (metrics.accuracy_score(y_train_1, y_pre)))

#
# from xgboost import plot_tree,plot_importance
# import matplotlib.pyplot as plt
# import matplotlib
# plot_tree(clf,num_trees=0)
# fig = matplotlib.pyplot.gcf()
# fig.set_size_inches(50, 50)
# fig.savefig('tree3.png')

# plot_importance(clf)
# fig = matplotlib.pyplot.gcf()
# fig.set_size_inches(50, 50)
# fig.savefig('importance.png')


X_train_new2 = mergeToOne(X_train_2, new_feature)
new_feature_test = clf.apply(X_test)
X_test_new = mergeToOne(X_test, new_feature_test)

model = XGBClassifier(
    learning_rate=0.05,  # 默认0.3
    n_estimators=300,  # 树的个数
    max_depth=7,
    min_child_weight=1,
    gamma=0.5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',  # 逻辑回归损失函数
    nthread=8,  # cpu线程数
    scale_pos_weight=1,
    reg_alpha=1e-05,
    reg_lambda=1,
    seed=1024)  # 随机种子

model.fit(X_train_2, y_train_2)
y_pre = model.predict(X_test)
# y_pro = model.predict_proba(X_test)[:, 1]

# print("AUC Score :", (metrics.roc_auc_score(y_test, y_pro)))
print("Accuracy :", (metrics.accuracy_score(y_test, y_pre)))

model = XGBClassifier(
    learning_rate=0.05,  # 默认0.3
    n_estimators=300,  # 树的个数
    max_depth=7,
    min_child_weight=1,
    gamma=0.5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',  # 逻辑回归损失函数
    nthread=8,  # cpu线程数
    scale_pos_weight=1,
    reg_alpha=1e-05,
    reg_lambda=1,
    seed=1024)  # 随机种子

model.fit(X_train_new2, y_train_2)
y_pre = model.predict(X_test_new)
# y_pro = model.predict_proba(X_test_new)[:, 1]
# print("AUC Score :", (metrics.roc_auc_score(y_test, y_pro)))
print("Accuracy :", (metrics.accuracy_score(y_test, y_pre)))
