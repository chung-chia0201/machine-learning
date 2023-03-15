from cgi import test
import numpy as np
import matplotlib.pyplot as plt
from re import L
import pandas as pd
import os
import time
import seaborn as sns; sns.set()
from sklearn import manifold, datasets
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn import decomposition
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

##基礎設定
i_filepath=os.path.dirname(__file__)
pd.set_option('display.max_columns',None)   #顯示所有列
pd.set_option('display.max_columns',None)   #顯示所有列
pd.set_option('max_colwidth',100)           #顯示值(最大長度100)

##資料預處理
df=pd.read_csv(i_filepath+'\\company\\data.csv')
df=df.drop_duplicates()                     #刪除重複的
# print(df.info())

##設定特徵值
df_x=df.select_dtypes(exclude=['object'])
df_x=df_x.drop(df_x.columns[0],axis=1)                        # 刪除Id

##特徵值y
df_y=df[df.columns[0]]


# ##劃分訓練、測試用資料
# x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.2)  #decission tree 不使用pac
# ##test_size=0.2表示20%測試用80%訓練用

# ####pca
# pca=decomposition.PCA()
# pca.fit(x_train) # 用PCA降維
# # 打印降維後的新特徵
# variances=pca.explained_variance_ratio_
# # print(variances)                                              # 在資料中不同維度的貢獻度(資料占比)

# # 故而可以為重要性設置一個閾值，小於該閾值的認為該特徵不重要，可刪除
# thresh=0.0001                                                    # 剩下前22個
# useful_features=variances>thresh
# # print(useful_features)                                      # 標記為True的表示重要特徵，要保留，False則刪除

# useful_features_num=np.sum(useful_features)                   # 計算True的個數
# # print('useful_features_num:',useful_features_num)

# # 進行PCA降維之後的新數據集為：
# pca.n_components=useful_features_num                          # 即設置PCA的新特徵數量為n_components
# new_dataset_X=pca.fit_transform(x_train)
# print('before PCA, dataset shape: ', x_train.shape)
# print('after PCA, dataset shape: ', new_dataset_X.shape)

# x_train=new_dataset_X
# x_test=pca.fit_transform(x_test)

##測試哪一個最好(knn)
# s=[]
# knn_index=[]
# for i in range(2,20):
#     k=i
#     knn=KNeighborsClassifier(n_neighbors=k)
#     knn.fit(x_train,y_train)
#     print('k={},準確率={}'.format(k,knn.score(x_test,y_test)))
#     # print(type(knn.score(x_test,y_test)))
#     s.append(knn.score(x_test,y_test))
#     knn_index.append(k)

# df_knn=pd.Series(s,index=knn_index)
# df_knn.plot(grid=True)
# plt.show()

##選擇k
# start = time.time()
# k=13
# knn=KNeighborsClassifier(n_neighbors=k)  #weights='distance'
# knn.fit(x_train,y_train)   ################

# ##以測試集進行預測
# print('分類的預測結果：')
# y_pred = knn.predict(x_test) #產生Test data預測結果
# end = time.time()
# print(y_pred)                #測試結果
# print(y_test.values)       #實際結果，y_test會包含索引，不好比對，所以用y_test.values只有值

# ##計算分類準確率
# print(accuracy_score(y_test,y_pred))       #accuracy_score(真實數據,預測結果)

# ##計算混淆矩陣
# print(confusion_matrix(y_test,y_pred))     #confusion_matrix(真實數據,預測結果)
# print(classification_report(y_test,y_pred))

# ##交叉驗證
# s = cross_val_score(knn, df_x, df_y, scoring='accuracy', cv=10)
# #cross_val_score(knn模型名稱, 特徵值資料框, 標籤序列, scoring='accuracy', cv=重複次數)   
# print('交叉驗證每次的準確率：{}'.format(s))
# print('交叉驗證得到的平均準確率：{}'.format(s.mean()))    #平均正確率0.9648
# print("執行時間：%f 秒" % (end - start))                #0.128374秒

##naive bayes
# start = time.time()
# GNB = GaussianNB()
# GNB = GNB.fit(x_train, y_train)
# y_pred=GNB.predict(x_test)
# end = time.time()
# print("GaussianNB,樣本總數： %d 錯誤樣本數 : %d" % (x_train.shape[0],(y_test != y_pred).sum()))
# print(GNB.predict_proba(x_test))
# print(GNB.score(x_test,y_test))            #分類成功率
# print(classification_report(y_test,y_pred))

# ##交叉驗證
# score = cross_val_score(GNB, df_x, df_y, scoring='accuracy', cv=10,n_jobs=1)           #平均正確率0.23358
# print('交叉驗證每次的準確率：{}'.format(score))
# print('交叉驗證得到的平均準確率：{}'.format(score.mean()))
# print("執行時間：%f 秒" % (end - start))                       #0.004秒

###svm
## SVM=Support Vector Machine 是支持向量
## SVC=Support Vector Classification就是支持向量机用于分类
x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.2)
start = time.time()
SVM_svc = SVC()
SVM_svc.fit(x_train,y_train)

y_pred = SVM_svc.predict(x_test)
end = time.time()

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

##交叉驗證
score = cross_val_score(SVM_svc, df_x, df_y, scoring='accuracy', cv=10,n_jobs=1)           #平均正確率0.9677371986168349
print('交叉驗證每次的準確率：{}'.format(score))
print('交叉驗證得到的平均準確率：{}'.format(score.mean()))
print("執行時間：%f 秒" % (end - start))                    #0.265212秒

#decisiontree tree
##劃分測試集和訓練集，這邊不使用PCA降維
# x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.2)

# start = time.time()
# dtree = DecisionTreeClassifier(criterion='entropy', max_depth=4)    # set hyperparameter 
# # classsklearn.tree.DecisionTreeClassifier(criterion=’gini’, splitter=’best’, max_depth=None, min_samples_split=2,
# # min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, 
# # min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)

# dtree.fit(x_train,y_train)

# y_pred_dt = dtree.predict(x_test)
# end = time.time()
# print(classification_report(y_test,y_pred_dt))
# print(confusion_matrix(y_test,y_pred_dt))

# ##交叉驗證
# score = cross_val_score(dtree, df_x, df_y, scoring='accuracy', cv=10,n_jobs=1)           #平均正確率0.9461825157931454
# print('交叉驗證每次的準確率：{}'.format(score))
# print('交叉驗證得到的平均準確率：{}'.format(score.mean()))
# print("執行時間：%f 秒" % (end - start))                                                  #0.125572 秒

# #畫出樹
# plt.figure(figsize=(12,12))  # set plot size (denoted in inches)
# tree.plot_tree(dtree,fontsize=9)
# plt.show()

# #random forest
# #n_estimator代表要使用多少CART樹（CART樹為使用GINI算法的決策樹）
# start = time.time()
# rfc = RandomForestClassifier(n_estimators=100)

# #從訓練組資料中建立隨機森林模型
# rfc.fit(x_train,y_train)

# #預測測試組的駝背是否發生
# y_pred_rfc = rfc.predict(x_test)
# end = time.time()
# print(confusion_matrix(y_test,y_pred_rfc))

# ##交叉驗證
# score = cross_val_score(rfc, df_x, df_y, scoring='accuracy', cv=10,n_jobs=1)           #平均正確率0.9680313149973516
# print('交叉驗證每次的準確率：{}'.format(score))
# print('交叉驗證得到的平均準確率：{}'.format(score.mean()))
# #利用classification report來看precision、recall、f1-score、support
# print(classification_report(y_test,y_pred_rfc))
# print("執行時間：%f 秒" % (end - start))                                                #1.843231 秒