
# coding: utf-8

# In[320]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, NuSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import csv


# In[35]:


df_train = pd.read_csv('./task2_s92hdj/train.csv')
df_test = pd.read_csv('./task2_s92hdj/test.csv')


# **Discovery**  
# - Balanced dataset -> 0,1,2 around 1/3 each  
# - feature x18 is exactly the same as feature x20 
# 
# **Goal**
# - hard baseline: 0.814814814815  
# - medium baseline: 0.779100529101  
# - easy baseline: 0.763888888889

# In[214]:


X = df_train.loc[:,'x1':'x20']
y = df_train.loc[:,'y']
X_submission = df_test.loc[:, 'x1':'x20']


# In[189]:


# scaler = StandardScaler()
# X = scaler.fit_transform(X)


# In[293]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# In[ ]:


deg = 2
params = [.1,.15,.2,.25,.5,.6]
score = np.zeros(len(params))
kf = KFold(n_splits=10, shuffle=False)

print("Training Start...")
for train_idx, val_idx in kf.split(X_train):
    for i in range(len(params)):
        clf = NuSVC(nu=params[i], kernel='poly', degree=deg)
        clf.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        y_val = clf.predict(X_train.iloc[val_idx])
        score[i]+=accuracy_score(y_train.iloc[val_idx], y_val)
    print("Finished %d" % i)


# In[ ]:


best_param = params[np.argmax(score)]
print("Best nu: %.2f" % best_param)


# In[ ]:


clf=NuSVC(nu=best_param, kernel='poly', degree=deg)
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)


# In[ ]:


ts = accuracy_score(y_test, y_predict) 
print("Test score: %.5f" % ts)


# In[ ]:


y_submission = clf.predict(X_submission)
df = pd.DataFrame({'Id': range(2000, 5000),'y':y_submission})


# In[336]:


df.to_csv('./task2_s92hdj/NuSVC_%f_%f_%d.csv' % (best_param, ts, deg), index=False)

