#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
from scipy.stats import skew, boxcox

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import roc_auc_score, accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import average_precision_score,recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from time import time
from sklearn.ensemble import GradientBoostingClassifier
# ignore warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn import tree

import xgboost as xgb
from xgboost import plot_tree


import os


# In[2]:


# read dataframe
df = pd.read_csv('/Users/nalingadihoke/Desktop/Stout_Case_Study/Question_1/data/PS_20174392719_1491204439457_log.csv.zip')


# In[3]:


df.head()


# In[4]:


df1 = df.copy()

df1.rename(columns={'newbalanceOrig':'newbalanceOrg'},inplace=True)
df1.drop(labels=['nameOrig','nameDest'],axis=1,inplace=True)


# In[5]:


path = '/Users/nalingadihoke/Desktop/Stout_Case_Study/Question_1/output/'


# In[7]:



var = df1.groupby('type').amount.sum()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
plt.rcParams.update({'font.size': 43})
var.plot(kind='bar', color=['red', 'green', 'blue', 'yellow', 'orange'], figsize=(40, 20))
ax1.set_title("Total amount per transaction type")
ax1.set_xlabel('Type of Transaction')
ax1.set_ylabel('Amount')


# In[8]:


fig.savefig(path+'plot1.png', bbox_inches='tight')


# In[89]:


fig2 = plt.figure()
plt.rcParams.update({'font.size': 43})
plt.title('Heat Map')
corr = df1.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)


# In[90]:


# fig2.savefig(path+'plot2.png', bbox_inches='tight')


# In[91]:


piedata = df1.groupby(['isFlaggedFraud']).sum()

fig3, axes = plt.subplots(1,1, figsize=(6,6))
axes.set_title("% of fraud transaction detected")
piedata.plot(kind='pie',y='isFraud',ax=axes, fontsize=18,shadow=False,autopct='%1.1f%%');
axes.set_ylabel('');
plt.legend(loc='upper left',labels=['Not Detected','Detected'])
plt.show()

fig3.savefig(path+'plot3.png', bbox_inches='tight')


# In[103]:


fraud = df1.loc[df1.isFraud == 1]
nonfraud = df1.loc[df1.isFraud == 0]

sns.set_style(rc={ 'figure.facecolor':'white'})

fig4 = plt.figure()
ax = fig4.add_subplot(1,1,1)
ax.scatter(nonfraud['oldbalanceOrg'],nonfraud['amount'],c='g')
ax.scatter(fraud['oldbalanceOrg'],fraud['amount'],c='r')
plt.legend(loc='upper right',labels=['Not Fraud','Fraud'])
ax.set_xlabel('Balance')
ax.set_ylabel('Amount')
ax.set_title('Balance Before Transaction vs Transaction Amount')
plt.show()


# In[104]:


fig4.savefig(path+'plot4.png', bbox_inches='tight')


# In[121]:


fig5 = plt.figure()

ax5 = fig5.add_subplot(1,1,1)

sns.scatterplot(data=df1, x='newbalanceDest', y='amount', color = 'orange')

ax5.set_xlabel('Balance')
ax5.set_ylabel('Amount')
ax5.set_title('Balance at New Destination vs Transaction Amount')

plt.show()


# In[122]:


fig5.savefig(path+'plot5.png', bbox_inches='tight')


# In[123]:


# ML

# Method 1

raw_data = df.copy()

print('Data preview:')
print(raw_data.head())
 
print('Data statistics:')
print(raw_data.describe())
 
print('Basic information of data set:')
print(raw_data.info())


# In[126]:


used_data = raw_data[(raw_data['type'] =='TRANSFER') | (raw_data['type'] =='CASH_OUT')] 

#Only the row data TRANSFER and CASH_OUT types are retained

used_data.drop(['step','nameOrig','nameDest','isFlaggedFraud'], axis=1, inplace=True) #drop useless feature data column
 # Reset index     
used_data = used_data.reset_index(drop=True)
 
 #Convert type into category data, ie 0, 1
type_label_encoder = preprocessing.LabelEncoder()
type_category = type_label_encoder.fit_transform(used_data['type'].values)
used_data['typeCategory'] = type_category

used_data.head()


# In[128]:


xx=pd.value_counts(used_data['isFraud'],sort=True)

xx

# number of positive samples is small compared to negative samples


# In[129]:


feature_names=['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','typeCategory']
X=used_data[feature_names]
Y=used_data['isFraud']
#X.head()
Y.head()


# In[131]:


# Method 1 - Logistic Regression

# Scaling sample size first

number_records_fraud=len(used_data[used_data['isFraud']==1])


fraud_indices=used_data[used_data['isFraud']==1].index.values 
#Index of positive samples

nonfraud_indices=used_data[used_data['isFraud']==0].index

random_nonfraud_indices=np.random.choice(nonfraud_indices,number_records_fraud,replace=False) #In the negative sample index, randomly select 8213 indexes as the new negative sample!

random_nonfraud_indices=np.array(random_nonfraud_indices)

under_sample_indices=np.concatenate([fraud_indices,random_nonfraud_indices]) #New downsampling data index! !
under_sample_data=used_data.iloc[under_sample_indices,:]
X_undersample = under_sample_data[feature_names].values 
y_undersample = under_sample_data['isFraud'].values

print("Non-fraud record ratio: ", len(under_sample_data[under_sample_data['isFraud'] == 0]) / len(under_sample_data))
print("Fraud record ratio: ", len(under_sample_data[under_sample_data['isFraud'] == 1]) / len(under_sample_data))



# In[132]:




X_train, X_test, y_train, y_test = train_test_split(X_undersample, y_undersample, test_size=0.3, random_state=0) #7: 3 split
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred_score = lr_model.predict_proba(X_test)
y_pred_score


# In[133]:


# explain from documentation
y_score = lr_model.score(X_train, y_train)
y_score


# In[136]:


precision_score = metrics.average_precision_score(y_test, y_pred_score[:, 1])
precision_score


# In[138]:



sns.reset_defaults()

fig6 = plt.figure()
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_score[:, 1]) #Attention threshold
roc_auc = metrics.auc(fpr,tpr)
plt.title('ROC curve')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

fig6.savefig(path+'plot6.png', bbox_inches='tight')


# In[139]:


# Method 2 - XGBoost (Gradient Boosting)

# read dataframe again
df2 = pd.read_csv('/Users/nalingadihoke/Desktop/Stout_Case_Study/Question_1/data/PS_20174392719_1491204439457_log.csv.zip')

df = df2.copy()

df = df2.rename(columns={'oldbalanceOrg': 'Old_Balance_Orig',
                        'newbalanceOrig': 'New_Balance_Orig',
                        'oldbalanceDest': 'Old_Balance_Dest',
                        'newbalanceDest': 'New_Balance_Dest',
                        'nameOrig': 'Name_Orig',
                        'nameDest': 'Name_Dest'})

df.head()


# In[140]:


df['Error_Orig']=df['Old_Balance_Orig']-df['New_Balance_Orig']-df['amount']
df['Error_Dest']=df['Old_Balance_Dest']-df['New_Balance_Dest']+df['amount']
print(df.head())


# In[141]:


X = df.loc[(df.type == 'CASH_OUT')]
X['Error_Orig']=X['Old_Balance_Orig']-X['New_Balance_Orig']-X['amount']
X['Error_Dest']=X['Old_Balance_Dest']-X['New_Balance_Dest']+X['amount']
y = X['isFraud']
del X['isFraud']

print(df.describe())


# In[143]:


X = X.drop(['Name_Orig', 'Name_Dest', 'type'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

# X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.2, random_state=1)

# weight calculation
weights = (y == 0).sum() / (1.0 * (y == 1).sum())

# XGBoost
clf = xgb.XGBClassifier(max_depth=3, scale_pos_weight=weights, n_jobs=4)
probabilities = clf.fit(X_train, y_train).predict_proba(X_test)


y_pred=clf.predict(X_test)
acc=accuracy_score(y_test,y_pred)
print('accuracy', acc)

print('AUPRC = {}'.format(
    average_precision_score(y_test, probabilities[:, 1])))
# recall score
print('Recall:{0:2f}'.format(recall_score(y_test,y_pred)))


# In[144]:


print('F1 macro score')
print(f1_score(y_test, y_pred, average='macro'))
print('F1 micro score')
print(f1_score(y_test, y_pred, average='micro'))

print('confusion matrix of decision tree with .2 random test data:')
print(confusion_matrix(y_test, y_pred))


# In[148]:


clf.fit(X_train, y_train)


# In[161]:


plot_tree(clf)
fig = matplotlib.pyplot.gcf()
fig.savefig('plot7.png')


# In[ ]:




