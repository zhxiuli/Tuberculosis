#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
import numpy as np
from pandas import Series,DataFrame
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import metrics
from time import time
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import re
from feature_selector import FeatureSelector



# In[14]:


data = pd.read_csv("J:\科研小助手\神经网络keras\keras\数据分析-v10_整合_v6.txt")


# In[15]:


data.head()#此处y为数值型，0,0.5,1，但未零均值归一化，99个特征都归一化过。。。。。


# In[16]:


dcorr = data.corr()



# In[32]:


y = df["yNUM"]
X = data.ix[:,:99]


# In[33]:


X


# In[34]:


fs = FeatureSelector(data = X, labels = y)


# In[35]:


fs.identify_missing(missing_threshold=0.6)


# In[36]:


missing_features = fs.ops['missing']
missing_features[:10]


# In[37]:


fs.plot_missing()


# In[38]:


fs.missing_stats.head(10)


# In[39]:


fs.identify_single_unique()


# In[40]:


single_unique = fs.ops['single_unique']
single_unique


# In[41]:


fs.plot_unique()


# In[42]:


fs.unique_stats.sample(5)


# In[43]:


fs.identify_collinear(correlation_threshold=0.975)


# In[44]:


correlated_features = fs.ops['collinear']
correlated_features[:5]


# In[45]:


fs.plot_collinear()


# In[46]:


fs.plot_collinear(plot_all=True)



# In[53]:



fs.record_collinear.head(99)


# In[54]:


fs.identify_zero_importance(task = 'classification', eval_metric = 'auc', 
                            n_iterations = 10, early_stopping = True)


# In[56]:


fs = FeatureSelector(data = X, labels = df["yNUM"])
fs.identify_zero_importance(task = 'regression', eval_metric = 'auc', 
                            n_iterations = 10, early_stopping = True)


# In[57]:


one_hot_features = fs.one_hot_features
base_features = fs.base_features
print('There are %d original features' % len(base_features))
print('There are %d one-hot features' % len(one_hot_features))



# In[58]:


fs.data_all.head(10)


# In[59]:


zero_importance_features = fs.ops['zero_importance']
zero_importance_features[10:15]


# In[60]:


zero_importance_features


# In[61]:


fs.plot_feature_importances(threshold = 0.99, plot_n = 12)


# In[62]:


fs.plot_feature_importances(threshold = 0.999, plot_n = 20)


# In[63]:


fs.feature_importances.head(40)


# In[64]:


one_hundred_features = list(fs.feature_importances.loc[:10, 'feature'])
len(one_hundred_features)


# In[65]:



fs.identify_low_importance(cumulative_importance = 0.99)


# In[66]:


low_importance_features = fs.ops['low_importance']
low_importance_features[:5]


# In[67]:



fs = FeatureSelector(data = X, labels = df["yNUM"])

fs.identify_all(selection_params = {'missing_threshold': 0.6, 'correlation_threshold': 0.98, 
                                    'task': 'regression', 'eval_metric': 'auc', 
                                     'cumulative_importance': 0.99})


# In[68]:


train_removed_all_once = fs.remove(methods = 'all', keep_one_hot = True)


# In[69]:


fs.feature_importances.head()


# In[70]:


fs.feature_importances.head(40)


# In[71]:


all_to_remove = fs.check_removal()
all_to_remove[:99]



