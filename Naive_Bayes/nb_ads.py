# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 12:05:37 2021

@author: jeehang

acknowledgement: https://www.analyticsvidhya.com/blog/2021/01/a-guide-to-the-naive-bayes-algorithm/
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# Importing the dataset
dataset = pd.read_csv('./Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_transform = sc.fit_transform(X)

# Training the Naive Bayes model on the Training set
classifier = GaussianNB()

k_fold = KFold(10,shuffle=True,random_state=0)

accuracy = cross_val_score(classifier,X_transform,y,cv=k_fold,scoring='accuracy')


mean_accuracy = accuracy.mean()
std_accuracy = accuracy.std()

for train_index, test_index in k_fold.split(X_transform):
    X_train, X_test = X_transform[train_index], X_transform[test_index]
    y_train, y_test = y[train_index], y[test_index]
    classifier.fit(X_train, y_train)
    lr_probs = classifier.predict_proba(X_test)[:, 1]
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

    plt.plot(lr_fpr, lr_tpr, marker='.')

print(mean_accuracy)
print(std_accuracy)
plt.show()

# Predicting the Test set results
y_pred = classifier.predict(X_test)
ac = accuracy_score(y_test,y_pred)
print(ac)
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
precision = cm[0,0] / (cm[0,0] + cm[1, 0])

# Recall 계산
recall = cm[0,0] / (cm[0,0] + cm[0, 1])

# Sensitivity 계산
sensitivity = recall

# Specificity 계산
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

print("Precision:", precision)
print("Recall:", recall)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
lr_probs = classifier.predict_proba(X_test)[:,1]
lr_auc = roc_auc_score(y_test, lr_probs)
print(lr_auc)
# calculate roc curves
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.show()
