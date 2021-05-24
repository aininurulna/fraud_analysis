# fraud_analysis
Fraud Analysis in Python

A Random Forest Classifier implemented in Python. The following arguments was passed to the object:

n_estimators = 50, random_state = 123

Before the modelling, an issue was found, that the data is imbalanced; thus, oversampling the minority class was done to tackle the problem.
The model works quite well on the data. To get better understanding of the model, evaluation using precision, recall, and confusion matrix were conducted.

Data is from kaggle competition:
A masked data.
https://www.kaggle.com/mlg-ulb/creditcardfraud?select=creditcard.csv
