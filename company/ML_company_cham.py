import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from collections import Counter
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action="ignore")

# Preprocessing Libraries

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

# Machine Learning Libraries

import sklearn
import xgboost as xgb
from sklearn import tree
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
from imblearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import classification_report
from mlxtend.classifier import StackingCVClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import recall_score, f1_score, roc_auc_score
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

##基礎設定
i_filepath=os.path.dirname(__file__)
pd.set_option('display.max_columns',None)   #顯示所有列
pd.set_option('display.max_columns',None)   #顯示所有列
pd.set_option('max_colwidth',100)           #顯示值(最大長度100)

bank_data = pd.read_csv(i_filepath+'\\company\\data.csv')
# print(bank_data.head())
# print(bank_data.info())

##刪除重複的資料
bank_data=bank_data.drop_duplicates()  

##多少公司破產
# print(bank_data['Bankrupt?'].value_counts())
# print('non-bankrupt: ', round(bank_data['Bankrupt?'].value_counts()[0]/len(bank_data) * 100,2), '% of the dataset')
# print('bankrupt: ', round(bank_data['Bankrupt?'].value_counts()[1]/len(bank_data) * 100,2), '% of the dataset')

# sns.set_theme(context = 'paper')
# plt.figure(figsize = (10,5))
# sns.countplot(bank_data['Bankrupt?'])
# plt.title('0: non-bankrupt || 1: bankrupt', fontsize=14)
# plt.show()

##EDA
#correlation heatmap
# bank_data.hist(figsize = (35,30), bins = 50)   ##會當掉
# plt.show()
# f, ax = plt.subplots(figsize=(30, 25))
# mat = bank_data.corr('spearman')
# mask = np.triu(np.ones_like(mat, dtype=bool))
# cmap = sns.diverging_palette(230, 20, as_cmap=True)
# sns.heatmap(mat, mask=mask, cmap=cmap, vmax=1, center=0,# annot = True,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5})
# plt.show()


# Plotting other interesting features
# f, axes = plt.subplots(ncols=4, figsize=(24,6))
# sns.boxplot(x='Bankrupt?', y=" Net Income to Total Assets", data=bank_data, ax=axes[0])
# axes[0].set_title('Bankrupt vs Net Income to Total Assets')

# sns.boxplot(x='Bankrupt?', y=" Total debt/Total net worth", data=bank_data, ax=axes[1]) 
# axes[1].set_title('Bankrupt vs Tot Debt/Net worth Correlation')

# sns.boxplot(x='Bankrupt?', y=" Debt ratio %", data=bank_data, ax=axes[2])
# axes[2].set_title('Bankrupt vs Debt ratio Correlation')

# sns.boxplot(x='Bankrupt?', y=" Net worth/Assets", data=bank_data, ax=axes[3])  
# axes[3].set_title('Bankrupt vs Net Worth/Assets Correlation') 

# plt.show()

# f, axes = plt.subplots(ncols=4, figsize=(24,6))
# sns.boxplot(x='Bankrupt?', y=" Working Capital to Total Assets", data=bank_data, ax=axes[0])
# axes[0].set_title('Bankrupt vs  working capital to total assets')

# sns.boxplot(x='Bankrupt?', y=" Cash/Total Assets", data=bank_data, ax=axes[1])
# axes[1].set_title('Bankrupt vs cash / total assets')

# sns.boxplot(x='Bankrupt?', y=" Current Liability to Assets", data=bank_data, ax=axes[2])
# axes[2].set_title('Bankrupt vs current liability to assets')


# sns.boxplot(x='Bankrupt?', y=" Retained Earnings to Total Assets", data=bank_data, ax=axes[3])
# axes[3].set_title('Bankrupt vs  Retained Earnings to Total Assets')

# plt.show()

# Plotting the feature distributions for close to bankrputcy companies
# 繪製接近破產公司的特徵分佈

# f, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(24, 6))

# cash_flow_rate = bank_data[' Net Income to Total Assets'].loc[bank_data['Bankrupt?'] == 1].values
# sns.distplot(cash_flow_rate,ax=ax1, fit=norm, color='#FB8861')
# ax1.set_title(' Net Income to Total Assets \n (Unstable companies)', fontsize=14)

# tot_debt_net = bank_data[' Total debt/Total net worth'].loc[bank_data['Bankrupt?'] == 1].values
# sns.distplot(tot_debt_net ,ax=ax2, fit=norm, color='#56F9BB')
# ax2.set_title('total debt/tot net worth \n (Unstable companies)', fontsize=14)


# debt_ratio = bank_data[' Debt ratio %'].loc[bank_data['Bankrupt?'] == 1].values
# sns.distplot(debt_ratio,ax=ax3, fit=norm, color='#C5B3F9')
# ax3.set_title('debt_ratio \n (Unstable companies)', fontsize=14)

# net_worth_assets = bank_data[' Net worth/Assets'].loc[bank_data['Bankrupt?'] == 1].values
# sns.distplot(net_worth_assets,ax=ax4, fit=norm, color='#C5B3F9')
# ax4.set_title('net worth/assets \n (Unstable companies)', fontsize=14)

# plt.show()

# f, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(24, 6))
# working_cap = bank_data[' Working Capital to Total Assets'].loc[bank_data['Bankrupt?'] == 1].values
# sns.distplot(working_cap,ax=ax1, fit=norm, color='#FB8861')
# ax1.set_title('working capitals to total assets \n (Unstable companies)', fontsize=14)

# cash_tot_assets = bank_data[' Cash/Total Assets'].loc[bank_data['Bankrupt?'] == 1].values
# sns.distplot(cash_tot_assets ,ax=ax2, fit=norm, color='#56F9BB')
# ax2.set_title('cash/total assets \n (Unstable companies)', fontsize=14)

# asset_liab = bank_data[' Current Liability to Assets'].loc[bank_data['Bankrupt?'] == 1].values
# sns.distplot(asset_liab,ax=ax3, fit=norm, color='#C5B3F9')
# ax3.set_title('liability to assets \n (Unstable companies)', fontsize=14)

# operating_funds = bank_data[' Retained Earnings to Total Assets'].loc[bank_data['Bankrupt?'] == 1].values
# sns.distplot(operating_funds,ax=ax4, fit=norm, color='#C5B3F9')
# ax4.set_title('retain earnings to total assets \n (Unstable companies)', fontsize=14)

# plt.show()

def outliers_removal(feature,feature_name,dataset):
    
    # Identify 25th & 75th quartiles

    q25, q75 = np.percentile(feature, 25), np.percentile(feature, 75)
    # print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
    feat_iqr = q75 - q25
    # print('iqr: {}'.format(feat_iqr))
    
    feat_cut_off = feat_iqr * 1.5
    feat_lower, feat_upper = q25 - feat_cut_off, q75 + feat_cut_off
    # print('Cut Off: {}'.format(feat_cut_off))
    # print(feature_name +' Lower: {}'.format(feat_lower))
    # print(feature_name +' Upper: {}'.format(feat_upper))
    
    outliers = [x for x in feature if x < feat_lower or x > feat_upper]
    # print(feature_name + ' outliers for close to bankruptcy cases: {}'.format(len(outliers)))
    #print(feature_name + ' outliers:{}'.format(outliers))

    dataset = dataset.drop(dataset[(dataset[feature_name] > feat_upper) | (dataset[feature_name] < feat_lower)].index)
    # print('-' * 65)
    
    return dataset

for col in bank_data:
    new_df = outliers_removal(bank_data[col],str(col),bank_data)
# List to append the score and then find the average

# Dividing Data and Labels

labels = new_df['Bankrupt?']
new_df = new_df.drop(['Bankrupt?'], axis = 1)

def log_trans(data):
    
    for col in data:
        skew = data[col].skew()
        if skew > 0.5 or skew < -0.5:
            data[col] = np.log1p(data[col])
        else:
            continue
            
    return data

data_norm = log_trans(new_df)
data_norm.hist(figsize = (35,30),bins = 50)

##劃分資料
X_raw,X_test,y_raw,y_test  = train_test_split(data_norm,labels, test_size=0.1,stratify = labels,random_state = 42)

sss = StratifiedKFold(n_splits=5, random_state=42, shuffle=False)

for train_index, test_index in sss.split(X_raw,y_raw):
    
    # print("Train:", train_index, "Test:", test_index)
    X_train_sm, X_val_sm = X_raw.iloc[train_index], X_raw.iloc[test_index]
    y_train_sm, y_val_sm = y_raw.iloc[train_index], y_raw.iloc[test_index]

# Check the Distribution of the labels


# Turn into an array
X_train_sm = X_train_sm.values
X_val_sm = X_val_sm.values
y_train_sm = y_train_sm.values
y_val_sm = y_val_sm.values

# See if both the train and test label distribution are similarly distributed
train_unique_label, train_counts_label = np.unique(y_train_sm, return_counts=True)
test_unique_label, test_counts_label = np.unique(y_val_sm, return_counts=True)
# print('-' * 84)

# print('Label Distributions: \n')
# print(train_counts_label/ len(y_train_sm))
# print(test_counts_label/ len(y_val_sm))

accuracy_lst_reg = []
precision_lst_reg = []
recall_lst_reg = []
f1_lst_reg = []
auc_lst_reg = []

log_reg_sm = LogisticRegression()
#log_reg_params = {}
log_reg_params = {"penalty": ['l2'],
                  'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  'class_weight': ['balanced',None],
                  'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}

rand_log_reg = RandomizedSearchCV(LogisticRegression(), log_reg_params, n_iter=4)


for train, val in sss.split(X_train_sm, y_train_sm):
    pipeline_reg = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'), rand_log_reg) # SMOTE happens during Cross Validation not before..
    model_reg = pipeline_reg.fit(X_train_sm[train], y_train_sm[train])
    best_est_reg = rand_log_reg.best_estimator_
    prediction_reg = best_est_reg.predict(X_train_sm[val])
    
    accuracy_lst_reg.append(pipeline_reg.score(X_train_sm[val], y_train_sm[val]))
    precision_lst_reg.append(precision_score(y_train_sm[val], prediction_reg))
    recall_lst_reg.append(recall_score(y_train_sm[val], prediction_reg))
    f1_lst_reg.append(f1_score(y_train_sm[val], prediction_reg))
    auc_lst_reg.append(roc_auc_score(y_train_sm[val], prediction_reg))


print('---' * 45)
print('')
print('Logistic Regression (SMOTE) results:')
print('')
print("accuracy: {}".format(np.mean(accuracy_lst_reg)))
print("precision: {}".format(np.mean(precision_lst_reg)))
print("recall: {}".format(np.mean(recall_lst_reg)))
print("f1: {}".format(np.mean(f1_lst_reg)))
print('')
print('---' * 45)