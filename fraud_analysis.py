#Random Forest Classifier

import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#import data
data = pd.read_csv('creditcard.csv')

#check for data type and null count
data.info()

#check if the data is imbalanced
data['Class'].value_counts()

#the data is imbalanced as the class[1] is undersampled, it takes only 0.9900990099009901% of the total data.
#to tackle this issue, oversampling the minority class will be conducted.
#prepare the data by seperating the features and the target variable, then oversample the minority class
def prepare(df: pd.DataFrame):
    x = df.iloc[:, 2:30].values
    y = df.Class.values
    sampler = RandomOverSampler()
    x_resampled, y_resampled =  sampler.fit_resample(x, y)
    return x_resampled, y_resampled
x_resampled, y_resampled = prepare(data)

#check the result of the oversampling
(unique, counts) = np.unique(y_resampled, return_counts=True)
frequencies = np.asarray((unique, counts)).T
display(frequencies)

#split the test and training data
x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, train_size=0.8, random_state=123)

rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
predicted = rfc.predict(x_test)
print(f'Accuracy Score:\n{accuracy_score(y_test, predicted)}')
#to evaluate the model performance for each class, evaluate the model using precision, recall, and confusion matrix
print('\nClassification Report:')
print(classification_report(y_test, predicted))
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, predicted))



