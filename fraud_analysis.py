#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
#import data
data = pd.read_csv('creditcard.csv')


# In[3]:


data.head()


# In[5]:


#because this is a data from external source, check the condition
#check for data type and null count
data.info()


# In[6]:


#check if the data is imbalanced
data['Class'].value_counts()


# In[13]:


pip install -U imbalanced-learn


# In[15]:


# the data is imbalanced as the class[1] is undersampled, it takes only 0.9900990099009901% of the total data.
#to tackle this issue, oversampling the minority class will be conducted.
#prepare the data by seperating the features and the target variable, then oversample the minority class
from imblearn.over_sampling import RandomOverSampler
def prepare(df: pd.DataFrame):
    x = df.iloc[:, 2:30].values
    y = df.Class.values
    sampler = RandomOverSampler()
    x_resampled, y_resampled =  sampler.fit_resample(x, y)
    return x_resampled, y_resampled
x_resampled, y_resampled = prepare(data)


# In[27]:


#check the result of the oversampling
(unique, counts) = np.unique(y_resampled, return_counts=True)
frequencies = np.asarray((unique, counts)).T
display(frequencies)


# In[31]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, train_size=0.8, random_state=123)


# In[33]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
predicted = rfc.predict(x_test)
print(f'Accuracy Score:\n{accuracy_score(y_test, predicted)}')
#to evaluate the model performance for each class, evaluate the model using precision, recall, and confusion matrix
print('\nClassification Report:')
print(classification_report(y_test, predicted))
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, predicted))


# In[ ]:




