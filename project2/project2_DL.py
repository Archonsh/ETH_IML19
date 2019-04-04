
# coding: utf-8

# In[37]:


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras import optimizers
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# In[38]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[39]:


df_train = pd.read_csv('./task2_s92hdj/train.csv')
df_test = pd.read_csv('./task2_s92hdj/test.csv')


# In[40]:


df_train = df_train.drop(['x9','x10','x18'], axis=1)
df_test = df_test.drop(['x9','x10','x18'], axis=1)
X = df_train.loc[:,'x1':'x20']
y = df_train.loc[:,'y']
X_submission = df_test.loc[:, 'x1':'x20']


# In[5]:


# %matplotlib inline
# sns.pairplot(df_train.iloc[:,1:], hue='y')


# In[78]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# In[79]:
width = 512
dp = 0.3
def baseline_model():
    model = Sequential()
    model.add(Dense(width, input_dim=17, activation='relu'))
    model.add(Dropout(dp))
    model.add(Dense(width, activation='relu'))
    model.add(Dropout(dp))
    model.add(Dense(width, activation='relu'))
    model.add(Dropout(dp))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.002), metrics=['accuracy'])
    return model


# In[80]:

ep = 200
bs = 10
estimator = KerasClassifier(build_fn=baseline_model, epochs=ep, batch_size=10, verbose=0)


# # In[69]:


# kf = KFold(n_splits=10, shuffle=False)


# # In[66]:


# results = cross_val_score(estimator, X_train, y_train, cv=kf)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# results


# In[81]:


estimator.fit(X_train, y_train)


# In[82]:


y_predicted = estimator.predict(X_test)
ts = accuracy_score(y_test, y_predicted)
print("Test Score: %f" % ts)


# In[55]:


y_submission = estimator.predict(X_submission)
df = pd.DataFrame({'Id': range(2000, 5000),'y':y_submission})
df.to_csv('./task2_s92hdj/NN_%d*3_dropout%f_%depoch_lr0.002_%f.csv' % (width, dp, ep, ts), index=False)

