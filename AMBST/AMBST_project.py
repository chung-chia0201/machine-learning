import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.decomposition import PCA

##基礎設定
i_filepath=os.path.dirname(__file__)
pd.set_option('display.max_columns',None)   #顯示所有列
pd.set_option('display.max_columns',None)   #顯示所有列
pd.set_option('max_colwidth',100)           #顯示值(最大長度100)

##資料預處理
df=pd.read_excel(i_filepath+'\\AMBST\\topz-20220511.xlsx')
# print(df.info())

df=df.fillna(0)
# print(df.info())

##輸出
# file_path=os.path.dirname(__file__)
# df.to_csv(file_path+'/blood_pressure.csv',encoding='utf-8-sig',index=None)

pca=decomposition.PCA()
pca.fit(df) # 用PCA降維
# 打印降維後的新特徵
variances=pca.explained_variance_ratio_
print(variances)                                              # 在資料中不同維度的貢獻度(資料占比)

# 故而可以為重要性設置一個閾值，小於該閾值的認為該特徵不重要，可刪除
thresh=0.0001                                                 
useful_features=variances>thresh
print(useful_features)                                      # 標記為True的表示重要特徵，要保留，False則刪除

useful_features_num=np.sum(useful_features)                   # 計算True的個數
print('useful_features_num:',useful_features_num)

# 進行PCA降維之後的新數據集為：
pca.n_components=useful_features_num                          # 即設置PCA的新特徵數量為n_components
new_dataset_X=pca.fit_transform(df)
print('before PCA, dataset shape: ', df.shape)
print('after PCA, dataset shape: ', new_dataset_X.shape)