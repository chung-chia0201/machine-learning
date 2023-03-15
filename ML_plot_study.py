from turtle import color
from matplotlib import colors
import pandas as pd
import matplotlib.pyplot as plt
import os
i_filepath=os.path.dirname(__file__)


##基礎畫圖
# a=['E','W','S','N']
# m=[344,231,342,345]
# pd=pd.Series(m,index=a)
# # pd.plot(kind='pie',rot=0,title='test',colors=['red','blue','#00FF00','yellow'],fontsize=24,figsize=(1,2),autopct='%.2f')  
# pd.plot(kind='pie',colors=['red','blue','#00FF00','yellow'],rot=0,title='test')  
# #畫各種類型的圖,rot:轉動x軸字的角度,title='標題',color=['顏色'],fontsize=數字大小,figsize=(長,寬),autopct=數值格式顯示
# plt.show()

##使用.csv畫圖
# df=pd.read_csv(i_filepath+'\Python-for-Titanic\Ch05\觀光人數統計.csv')
# df.index=df['Month']
# df=df.drop('Month',axis=1)
# print(df.head())
# df.plot()
# df_T=df.T     #將資料轉置
# df_T.plot()
# plt.show()

##散佈圖
# df=pd.read_csv(i_filepath+'\Python-for-Titanic\Ch05\sunshine.csv')
# df.plot(kind='scatter',x='Temperature',y='SunShine',xlim=(0,40),ylim=(0,200))
# plt.show()



