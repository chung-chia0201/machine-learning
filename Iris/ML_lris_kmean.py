import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

i_filepath=os.path.dirname(__file__)
pd.set_option('display.max_columns',None)   #顯示所有列
pd.set_option('display.max_columns',None)   #顯示所有列
pd.set_option('max_colwidth',100)           #顯示值(最大長度100)

df=pd.read_csv(i_filepath+'\\Iris\\Iris.csv')
df=df.drop(df.columns[0],axis=1)            #刪除Id
df=df.drop_duplicates()                     #刪除重複的
# print(df.info())
df[df.columns[4]]=df[df.columns[4]].map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2 })
# print(df.info())

##設定特徵值
df_x=df[[df.columns[0],df.columns[1],df.columns[2],df.columns[3]]]

##建立kmean模型
# k=1                               #分幾群
# km=KMeans(n_clusters=k)

# ##訓練模型
# km.fit(df_x)

##檢測準確性
# print('分群準確性:',km.inertia_)   #km.inertia_所有資料點和所屬中心群中心的距離總和

# ##檢測哪一個k值最好
# s = []
# for k in range(1,15):
#     km = KMeans(n_clusters=k)
#     km.fit(df_x)
#     s.append(km.inertia_)
# print(s)

# ## 看視覺化圖表決定參數K值
# df_kmeans = pd.DataFrame()
# df_kmeans['inertia_'] = s
# df_kmeans.index = list(range(1,15))   
# df_kmeans.plot(grid=True)
# plt.show()
# #可選下降幅度由快速轉為平緩的k值

##選k=3
k=3                               
km=KMeans(n_clusters=k)
km.fit(df_x)

##以測試集進行預測
print('分群的預測結果：')
pred = km.fit_predict(df_x) 
print(pred) 

##進行分群預測
df1 = df_x.copy()
df1['pred'] = pred    #最右邊加入一行資料
c = {0:'r', 1:'g', 2:'b'}
df1['colors'] = df1['pred'].map(c)
print(df1.columns)
df1.plot(kind='scatter', x='SepalLengthCm',y='SepalWidthCm',c=df1['colors'])
plt.show()
