import pandas as pd
import os
import matplotlib.pyplot as plt
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
print(df.info())
df[df.columns[4]]=df[df.columns[4]].map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2 })
# print(df.info())

##分類成x,y
df_x=df[[df.columns[0],df.columns[1],df.columns[2],df.columns[3]]]
df_y=df[df.columns[4]]
# print(df_x.head())

##劃分訓練、測試用資料
x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.2)
#test_size=0.2表示20%測試用80%訓練用

##建立模型knn
# k=1
# knn=KNeighborsClassifier(n_neighbors=k)

# ##訓練模型
# knn.fit(x_train,y_train)

# ##計算模型準確率
# print('----KNN模式訓練後,取test data 進行分類的正確率計算-------')
# print('準確率:',knn.score(x_test,y_test))

s=[]
knn_index=[]
for i in range(2,10):
    k=i
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    print('k={},準確率={}'.format(k,knn.score(x_test,y_test)))
    print(type(knn.score(x_test,y_test)))
    s.append(knn.score(x_test,y_test))
    knn_index.append(k)

df_knn=pd.Series(s,index=knn_index)
df_knn.plot(grid=True)
plt.show()

##決定採用k=8
k=8
knn=KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train,y_train)

##以測試集進行預測
print('分類的預測結果：')
pred = knn.predict(x_test) #產生Test data預測結果
print(pred)                #測試結果
print(y_test.values)       #實際結果，y_test會包含索引，不好比對，所以用y_test.values只有值

##計算分類準確率
print(accuracy_score(y_test,pred))       #accuracy_score(真實數據,預測結果)

##計算混淆矩陣
print(confusion_matrix(y_test,pred))     #confusion_matrix(真實數據,預測結果)

##交叉驗證
s = cross_val_score(knn, df_x, df_y, scoring='accuracy', cv=10)
#cross_val_score(knn模型名稱, 特徵值資料框, 標籤序列, scoring='accuracy', cv=重複次數)   
print('交叉驗證每次的準確率：{}'.format(s))
print('交叉驗證得到的平均準確率：{}'.format(s.mean()))

##進行分類預測
new = [[6.6,3.1,5.2,2.4]]  #某一朵花的四個特徵值
v=knn.predict(new)         #呼叫knn模型進行預測
if v==0:
  s='Iris-Setosa'
elif v==1:
  s='Iris-Versicolour'
elif v==2:
  s='Iris-Virginica'
else:
  s='錯誤'
print('預測結果為：', s)
