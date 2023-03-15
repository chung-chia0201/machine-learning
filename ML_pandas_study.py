import pandas as pd
import os
i_filepath=os.path.dirname(__file__)
##目錄
##pandas_function
# #使用內定索引
# #使用自訂索引
##將資料匯出成.csv檔
##從網址抓取資料
##網路爬蟲
##資料觀察
##資料處理空值
##資料代換
##一次印出多筆資料
##印出指定列
##增加行資料
##增加列資料

##pandas_function
# c=['a','b','c']
# s=[1,2,3]
# cs=pd.Series(c)
# ss=pd.Series(s)
# print(cs)
# print(ss)
# print(cs[0])
# print(ss[0])

# sc=pd.Series(s,index=c)
# print(sc)

# #使用內定索引
# print(sc[1])
# sc[1]=100
# print(sc[1])

# #使用自訂索引
# print(sc['a'])
# sc['a']=1000
# print(sc['a'])
# print(sc)

# c=['國文','英文','數學','資訊科技']
# s=[84,92,88,95]
# df=pd.DataFrame()
# print(df)
# df['科目']=c
# df['分數']=s
# print(df)
# for i in range(len(df)):
#     print(df['科目'][i],df['分數'][i])
# df.index=['第一科','第二科','第三科','第四科']
# for i in range(len(df)):
#     print(df.index[i],df['科目'][i],df['分數'][i])

##mount_function(colab)



##將資料匯出成.csv檔
# c=['國文','英文','數學','社會','自然']
# s=[[84,92,88,95,77],[84,92,88,95,77],[84,92,88,95,77]]
# ind=['一','二','三']
# # df=pd.DataFrame(s,index=ind,columns=c)
# df=pd.DataFrame(s,columns=c)
# print(df)
# file_path=os.path.dirname(__file__)
# df.to_csv(file_path+'/五科成績.csv',encoding='utf-8-sig')  #index=None沒有index

##從網址抓取資料
# import pandas as pd
# df=pd.read_csv('https://data.epa.gov.tw/api/v1/aqx_p_322?limit=1000&api_key=9be7b239-557b-4c10-9775-78cadfc555e9&format=csv')
# print(df.head())

##網路爬蟲
# url='https://www.taiwanlottery.com.tw/Lotto/Lotto649/history.aspx'
# df = pd.read_html(url)
# goal=df[2]
# for i in range(5):
#     print(goal[1][i],goal[2][i],goal[3][i],goal[4][i],goal[5][i],goal[6][i],goal[7][i],goal[8][i])

##資料觀察
# i_filepath=os.path.dirname(__file__)
# df=pd.read_csv(i_filepath+'\學生成績檔-4-1.1.csv')
# print(df.head(3))  #預設為前五行
# print(df.info())
# print(df.describe())
# print(df.duplicated()) #檢查有無重複
# df=df.drop_duplicates()  #刪掉重複
# print(df.info())
# print(df.columns[4])
# df1=df.reset_index()     #重新排列index
# print(df1.head(7))
# print(df.head(7))

##資料處理空值
# df=pd.read_csv(i_filepath+'\Python-for-Titanic\Ch04\學生成績檔-4-1.3.csv')
# for i in range(4,9):
#     df[df.columns[i]]=df[df.columns[i]].fillna(df[df.columns[i]].mean())
# print(df.info())

##資料代換
# df=pd.read_csv(i_filepath+'\Python-for-Titanic\Ch04\學生成績檔-4-1.4.csv')
# print(df.head())
# s={'男':1,'女':0}
# df[df.columns[2]]=df[df.columns[2]].map(s)
# print(df.head())

#一次印出多筆資料
# df=pd.read_csv(i_filepath+'\Python-for-Titanic\Ch04\學生成績檔-4-2.1.csv')
# print(df[[df.columns[4],df.columns[5],df.columns[6],df.columns[7],df.columns[8]]].head())

#印出指定列
# df=pd.read_csv(i_filepath+'\Python-for-Titanic\Ch04\學生成績檔-4-2.2.csv')
# print(df[1:3])   #印出1-2行
# print(df.iloc[1:2,:])
# print(df.iloc[1:4,3:6])
# print(df.iloc[1,2])
# print(df.iloc[1,[1,2,4,5]])

##增加行資料
# df=pd.read_csv(i_filepath+'\Python-for-Titanic\Ch04\學生成績檔-4-2.7.csv')
# c=[df.columns[4],df.columns[5],df.columns[6],df.columns[7],df.columns[8]]
# df['平均']=df[c].mean(axis='columns')          #增加一行總分
# df=df.sort_values('平均',ascending=False)     #依照分數排名
# df['排名']=list(range(1,len(df)+1))           #加上排名欄位
# # print(df.iloc[3:6,0])
# print(df.head())

##增加列資料
# df=pd.read_csv(i_filepath+'\Python-for-Titanic\Ch04\學生成績檔-4-2.8.csv')
# print(df.tail())        #印出最後五行
# s=['1080041','張三','1','Aarfpoghko@gmail.com',90,94,59,69,34]
# df1=pd.DataFrame(data=[s],columns=df.columns)
# df=df.append(df1,ignore_index=True)
# print(df.tail())


