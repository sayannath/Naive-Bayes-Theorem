#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[2]:


print(os.listdir())


# In[3]:


# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#to get graphs inline
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


dataSet = pd.read_csv('Social_Network_Ads.csv')


# In[5]:


dataSet.info()


# In[6]:


dataSet.head()


# In[7]:


# spliting data into dependent and independent matrix
X = dataSet.iloc[:,2:4].values
y = dataSet.iloc[:,4].values


# In[8]:


X


# In[9]:


y


# In[10]:


# splitting data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[11]:


X_test


# In[12]:


y_test


# In[13]:


# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[14]:


X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[15]:


from sklearn.naive_bayes import GaussianNB


# In[16]:


classifier = GaussianNB()


# In[17]:


classifier.fit(X_train, y_train)


# In[18]:


y_predict = classifier.predict(X_test)


# In[19]:


y_predict


# In[20]:


from sklearn.metrics import confusion_matrix
falseCaci = confusion_matrix(y_test, y_predict)


# In[21]:


falseCaci

