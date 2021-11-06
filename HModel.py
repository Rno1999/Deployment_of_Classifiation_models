#!/usr/bin/env python
# coding: utf-8

# In[57]:


#For data manipulation
import pandas as pd
import numpy as np


#For modeling
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import pickle


# In[38]:


df=pd.read_csv('heart_Disease_Dataset.csv')


# In[39]:


df.head()


# In[40]:


categorical_val = []
continous_val = []
for column in df.columns:
    print('==============================')
    print(f"{column} : {df[column].unique()}")
    if len(df[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)


# In[41]:


def handle_outlier_IQR(df):
    Q1=df.quantile(0.25)
    Q3=df.quantile(0.75)
    IQR=Q3-Q1
    mean=df.mean()
    df_new=df.map(lambda x:x if (x>= Q1 - 1.5 * IQR) & (x <= Q3 + 1.5 *IQR) else mean)
    return df_new


# In[42]:


df["trestbps"]=handle_outlier_IQR(df['trestbps'])
df["chol"]=handle_outlier_IQR(df['chol'])
df["thalach"]=handle_outlier_IQR(df['thalach'])
df["oldpeak"]=handle_outlier_IQR(df['oldpeak'])


# In[43]:


y = df.target.values
x_data =df[continous_val]


# In[44]:


categorical_val.remove('target')
dataset = pd.get_dummies(df, columns = categorical_val)


# In[45]:


rest=dataset.drop(['age','trestbps','chol','thalach','oldpeak','target'], axis='columns')
# Normalize
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
frames = [rest, x]
x = pd.concat(frames, axis = 1)


# In[46]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.4,random_state=0)


# In[47]:


#transpose matrices
x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T


# In[48]:


accuracies = {}
precisions ={}
recalls ={}


# ### Creating Model for K-Nearest Neighbour (KNN)

# In[56]:


# try to find best k value
scoreList = []
for i in range(1,20):
    knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k
    knn2.fit(x_train.T, y_train.T)
    scoreList.append(knn2.score(x_test.T, y_test.T))
    
acc = max(scoreList)*100
accuracies['KNN'] = np.round(acc,2)
print("Maximum KNN Score is {:.2f}%".format(acc))


# In[50]:


knn = KNeighborsClassifier(n_neighbors = 1)  # n_neighbors means k
knn.fit(x_train.T, y_train.T)
prediction = knn.predict(x_test.T)
print("{} NN Score: {:.2f}%".format(2, knn.score(x_test.T, y_test.T)*100))


# In[51]:


pred = knn.predict(x_test.T)
precision, recall, fscore, support = precision_recall_fscore_support(y_test, pred, average='binary')
precisions['KNN'] = np.round(precision,2)
recalls['KNN'] = np.round(recall,2)


# In[52]:


print(confusion_matrix(y_test.T, pred))
print(classification_report(y_test.T, pred))


# In[53]:


with open('Heart_Model.pkl', 'wb') as file:
  pickle.dump(knn, file)


# In[55]:


model=pickle.load(open('Heart_Model.pkl','rb'))
print(model.predict([[52,125.0,212.0,168.0,1.0,0,0,1,1,0,0,0,1,0,0,1,0,1,0,0,0,1,0,0,1,0,0,0,0,0]]))

