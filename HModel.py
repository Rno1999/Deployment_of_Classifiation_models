#!/usr/bin/env python
# coding: utf-8

# In[32]:


#For data manipulation
import pandas as pd
import numpy as np

#For plot
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#For modeling
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
import plotly.express as px
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import StackingClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import tree


#For store dataset into database
from sqlalchemy import *

#For inserting images
from IPython.display import Image

#For warning
import warnings
warnings.filterwarnings("ignore")

import pickle


# In[8]:


df=pd.read_csv('heart_Disease_Dataset.csv')


# In[9]:


engine=create_engine('sqlite:///HD.db')
df.to_sql('HD_data',engine,if_exists='replace',index=False)


# In[10]:


#To check if the HD_data table existing
tables=engine.table_names()
print(tables);


# In[11]:


df=pd.read_sql('select * from HD_data',engine)


# In[12]:


df.head()


# In[13]:


categorical_val = []
continous_val = []
for column in df.columns:
    print('==============================')
    print(f"{column} : {df[column].unique()}")
    if len(df[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)


# In[16]:


fig = plt.figure(figsize=(20, 20))
ax = fig.gca()
df.plot(kind='box', subplots=True, layout=(4, 4), sharex=False, ax=ax)
plt.show()


# In[17]:


def handle_outlier_IQR(df):
    Q1=df.quantile(0.25)
    Q3=df.quantile(0.75)
    IQR=Q3-Q1
    mean=df.mean()
    df_new=df.map(lambda x:x if (x>= Q1 - 1.5 * IQR) & (x <= Q3 + 1.5 *IQR) else mean)
    return df_new


# In[18]:


df["trestbps"]=handle_outlier_IQR(df['trestbps'])
df["chol"]=handle_outlier_IQR(df['chol'])
df["thalach"]=handle_outlier_IQR(df['thalach'])
df["oldpeak"]=handle_outlier_IQR(df['oldpeak'])


# In[20]:


fig = plt.figure(figsize=(20, 20))
ax = fig.gca()
df.plot(kind='box', subplots=True, layout=(4, 4), sharex=False, ax=ax)
plt.show()


# In[21]:


y = df.target.values
x_data =df[continous_val]


# In[22]:


categorical_val.remove('target')
dataset = pd.get_dummies(df, columns = categorical_val)


# In[23]:


rest=dataset.drop(['age','trestbps','chol','thalach','oldpeak','target'], axis='columns')
# Normalize
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
frames = [rest, x]
x = pd.concat(frames, axis = 1)


# In[24]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.4,random_state=0)


# In[25]:


#transpose matrices
x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T


# In[26]:


accuracies = {}
precisions ={}
recalls ={}


# ### Creating Model for K-Nearest Neighbour (KNN)

# In[27]:


# try to find best k value
scoreList = []
for i in range(1,20):
    knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k
    knn2.fit(x_train.T, y_train.T)
    scoreList.append(knn2.score(x_test.T, y_test.T))
    
plt.plot(range(1,20), scoreList)
plt.xticks(np.arange(1,20,1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()

acc = max(scoreList)*100
accuracies['KNN'] = np.round(acc,2)
print("Maximum KNN Score is {:.2f}%".format(acc))


# In[28]:


knn = KNeighborsClassifier(n_neighbors = 1)  # n_neighbors means k
knn.fit(x_train.T, y_train.T)
prediction = knn.predict(x_test.T)
print("{} NN Score: {:.2f}%".format(2, knn.score(x_test.T, y_test.T)*100))


# In[29]:


pred = knn.predict(x_test.T)
precision, recall, fscore, support = precision_recall_fscore_support(y_test, pred, average='binary')
precisions['KNN'] = np.round(precision,2)
recalls['KNN'] = np.round(recall,2)


# In[30]:


print(confusion_matrix(y_test.T, pred))
print(classification_report(y_test.T, pred))


# In[33]:


with open('Heart_Model.pkl', 'wb') as file:
  pickle.dump(knn, file)


# In[36]:


model=pickle.load(open('Heart_Model.pkl','rb'))
print(model.predict([[52,125.0,212.0,168.0,1.0,0,0,1,1,0,0,0,1,0,0,1,0,1,0,0,0,1,0,0,1,0,0,0,0,0]]))

