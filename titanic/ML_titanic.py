import pandas as pd
import os
import matplotlib.pyplot as plt

##基礎設定
i_filepath=os.path.dirname(__file__)
pd.set_option('display.max_columns',None)   #顯示所有列
pd.set_option('display.max_columns',None)   #顯示所有列
pd.set_option('max_colwidth',100)           #顯示值(最大長度100)

df=pd.read_csv(i_filepath+'\\titanic\\train.csv')

df[df.columns[5]]=df[df.columns[5]].fillna(df[df.columns[5]].mean())     #以平均年齡補上空值
# print(df[df.columns[10]].value_counts())                                 #計算港口上船人數   
df[df.columns[11]]=df[df.columns[11]].fillna('S')                        #將空值補上
df=df.drop(df.columns[10],axis=1)                                        #由於客艙缺少過多值，所以刪除此列  
df=df.drop_duplicates()                                                  #刪除重複           
# print(df.info())

df[df.columns[4]]=df[df.columns[4]].map({'female':0,'male':1})            #將性別轉換為數值    
df[df.columns[10]]=df[df.columns[10]].map({'S':0,'C':1,'Q':2})           #將登船港口轉換為數值
# df[df.columns[1]].value_counts().plot(kind='pie',title='survived',autopct='%1.2f%%')
# plt.show()

##分群
# print(df.info())
print(df.groupby([df.columns[4]])[df.columns[0]].count())
print(df.groupby([df.columns[4],df.columns[1]])[df.columns[0]].count())

df.groupby(['Sex','Survived'])['PassengerId'].count().plot(kind='bar',rot=1)
plt.show()