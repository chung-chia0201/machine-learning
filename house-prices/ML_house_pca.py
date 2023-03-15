import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from re import L
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import seaborn as sns; sns.set()
from sklearn import decomposition

##基礎設定
i_filepath=os.path.dirname(__file__)
pd.set_option('display.max_columns',None)   #顯示所有列
pd.set_option('display.max_columns',None)   #顯示所有列
pd.set_option('max_colwidth',100)           #顯示值(最大長度100)

##資料預處理
df=pd.read_csv(i_filepath+'\\house-prices\\train.csv')
df=df.drop_duplicates()                     #刪除重複的

df_x_t=pd.read_csv(i_filepath+'\\house-prices\\test.csv')
df_x_t=df_x_t.drop_duplicates()                          #刪除重複的

##轉換數字
s1={'Reg':3,'IR1':2,'IR2':1,'IR3':0}
df[df.columns[7]]=df[df.columns[7]].map(s1)
df_x_t[df_x_t.columns[7]]=df_x_t[df_x_t.columns[7]].map(s1)

s2={'Lvl':3,'Bnk':2,'HLS':1,'Low':0}
df[df.columns[8]]=df[df.columns[8]].map(s2)
df_x_t[df_x_t.columns[8]]=df_x_t[df_x_t.columns[8]].map(s2)

s3={'Gtl':3,'Mod':2,'Sev':1}
df[df.columns[11]]=df[df.columns[11]].map(s3)
df_x_t[df_x_t.columns[11]]=df_x_t[df_x_t.columns[11]].map(s3)

s4={'1Story':1,'1.5Fin':2,'1.5Unf':3,'2Story':4,'2.5Fin':5,'2.5Unf':6,'SFoyer':7,'SLvl':8}
df[df.columns[16]]=df[df.columns[16]].map(s4)
df_x_t[df_x_t.columns[16]]=df_x_t[df_x_t.columns[16]].map(s4)

s5={'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0}
data_s5_index=[30,31,40,53,57,63,64,72]
for i in data_s5_index:
    df[df.columns[i]]=df[df.columns[i]].map(s5)
    df_x_t[df_x_t.columns[i]]=df_x_t[df_x_t.columns[i]].map(s5)

s6={'Gd':4,'Av':3,'Mn':2,'No':1,'NA':0}
df[df.columns[32]]=df[df.columns[32]].map(s6)
df_x_t[df_x_t.columns[32]]=df_x_t[df_x_t.columns[32]].map(s6)

s7={'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NA':0}
data_s7_index=[33,35]
for i in data_s7_index:
    df[df.columns[i]]=df[df.columns[i]].map(s7)
    df_x_t[df_x_t.columns[i]]=df_x_t[df_x_t.columns[i]].map(s7)

s8={'N':0,'Y':1}
df[df.columns[41]]=df[df.columns[41]].map(s8)
df_x_t[df_x_t.columns[41]]=df_x_t[df_x_t.columns[41]].map(s8)

s9={'Typ':7,'Min1':6,'Min2':5,'Mod':4,'Maj1':3,'Maj2':2,'Sev':1,'Sal':0}
df[df.columns[55]]=df[df.columns[55]].map(s9)
df_x_t[df_x_t.columns[55]]=df_x_t[df_x_t.columns[55]].map(s9)

s10={'Fin':3,'RFn':2,'Unf':1,'NA':0}
df[df.columns[60]]=df[df.columns[60]].map(s10)
df_x_t[df_x_t.columns[60]]=df_x_t[df_x_t.columns[60]].map(s10)

s11={'Y':2,'P':1,'N':0}
df[df.columns[65]]=df[df.columns[65]].map(s11)
df_x_t[df_x_t.columns[65]]=df_x_t[df_x_t.columns[65]].map(s11)

s12={'GdPrv':4,'MnPrv':3,'GdWo':2,'MnWw':1,'NA':0}
df[df.columns[73]]=df[df.columns[73]].map(s12)
df_x_t[df_x_t.columns[73]]=df_x_t[df_x_t.columns[73]].map(s12)

##補上空值
min_data_index=[3,59]                         #LotFrontage,GarageYrBlt#GarageQual,GarageCond
for i in min_data_index:
    df[df.columns[i]]=df[df.columns[i]].fillna(df[df.columns[i]].min())
    df_x_t[df_x_t.columns[i]]=df_x_t[df_x_t.columns[i]].fillna(df_x_t[df_x_t.columns[i]].min()) 
 
df=df.fillna(0) 

##設定特徵值
data_index=[7,8,11,16,17,18,30,31,32,33,35,40,41,47,48,49,50,51,52,53,54,55,56,57,60,61,63,64,65,72,73]        #評分一組(不大於10)  0.27142
# data_index=[17,18,31,32,33,35,40,53,57,60,61,63,64,72,73]                                                      #主觀評分一組  
# data_index=[7,8,11,16,30,41,47,48,49,50,51,52,54,55,56,65]                                                     #客觀評分一組  
# data_index=[3,4,19,20,26,34,36,37,38,43,44,45,46,59,62,66,67,68,69,70,71,75,77]                                #數字一組           0.27826
# data_index=[7,8,11,16,30,31,32,33,35,40,41,53,55,57,60,63,64,65,72,73]                                         #上面有的一組        0.30234
df_x=pd.DataFrame()
df_x_t_f=pd.DataFrame()
df_x_max=[]
df_x_t_f_max=[]

for i in data_index:
    df_x[df.columns[i]]=df[df.columns[i]]
    df_x_max.append(df_x[df.columns[i]].max())
    df_x_t_f[df_x_t.columns[i]]=df_x_t[df_x_t.columns[i]]
    df_x_t_f_max.append(df_x_t_f[df_x_t.columns[i]].max())

##補上空值df_x_t_f 
df_x_t_f=df_x_t_f.fillna(0)

##特徵值y
df_y=df[df.columns[80]]

####pca
pca=decomposition.PCA()
pca.fit(df_x) # 用PCA降維
# 打印降維後的新特徵
variances=pca.explained_variance_
# print(variances)                                            # 可以理解成該特徵的重要性，後面三個數字非常小，即特徵不重要

# 故而可以為重要性設置一個閾值，小於該閾值的認為該特徵不重要，可刪除
thresh=0.8                           ##剩下前9個
useful_features=variances>thresh
# print(useful_features)                                      # 標記為True的表示重要特徵，要保留，False則刪除

useful_features_num=np.sum(useful_features)                 # 計算True的個數
# 進行PCA降維之後的新數據集為：
pca.n_components=useful_features_num                        # 即設置PCA的新特徵數量為n_components
new_dataset_X=pca.fit_transform(df_x)
# print('before PCA, dataset shape: ', df_x.shape)
# print('after PCA, dataset shape: ', new_dataset_X.shape)

##測試哪一個最好
# s=[]
# knn_index=[]
# for i in range(2,20):
#     k=i
#     knn=KNeighborsClassifier(n_neighbors=k,weights='distance')
#     knn.fit(new_dataset_X,df_y)
#     print('k={},準確率={}'.format(k,knn.score(new_dataset_X,df_y)))
#     # print(type(knn.score(new_dataset_X,df_y)))
#     s.append(knn.score(new_dataset_X,df_y))
#     knn_index.append(k)

# df_knn=pd.Series(s,index=knn_index)
# df_knn.plot(grid=True)
# plt.show()

##選擇k
k=5
knn=KNeighborsClassifier(n_neighbors=k,weights='distance')  #weights='distance'
knn.fit(new_dataset_X,df_y)   ################

##進行分類預測
df_flash=pca.fit_transform(df_x_t_f)
df_answer_1=df_x_t[df_x_t.columns[0]]
answer=[]

for i in range(0,len(df_x_t)):
    new =df_flash[i]
    v=knn.predict([new])                           #呼叫knn模型進行預測
    answer.append(float(v))

df_answer=pd.DataFrame()
df_answer[df_x_t.columns[0]]=df_x_t[df_x_t.columns[0]]
df_answer['SalePrice']=answer
df_answer.to_csv(i_filepath+'\\house-prices\\df_answer_test.csv',encoding='utf-8-sig',index=False)