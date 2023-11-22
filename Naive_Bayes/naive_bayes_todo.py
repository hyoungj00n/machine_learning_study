# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 15:52:28 2016

@author: JeeHang Lee
@date: 20160926
@description: This is an example code showing how to use Naive Bayes 
        implemented in scikit-learn.  
"""

# Import library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets

import pandas as pd
import numpy as np

#
def replace(df):
    df = df.replace(['paid', 'current', 'arrears'], [2, 1, 0])
    df = df.replace(['none', 'guarantor', 'coapplicant'], [0, 1, 1])
    df = df.replace(['coapplicant'], [1])
    df = df.replace(['rent', 'own'], [0, 1])
    df = df.replace(['False', 'True'], [0, 1])
    df = df.replace(['none'], [float('NaN')])
    df = df.replace(['free'], [-1])
    return df
    
df = pd.read_csv('./fraud_data.csv')
res = replace(df)


history = res.iloc[:,1].values
coapplicant = res.iloc[:,2].values
accommodation = res.iloc[:,3].values


X = np.vstack([history,coapplicant,accommodation])
X= X.T
Y = np.array(res.iloc[:,4].values)

model = GaussianNB()
model.fit(X, Y)
predicted = model.predict([[2,0,0],[2,1,0],[0,1,0],[0,1,1],[0,2,1]])
pred_prob = model.predict_proba([[2,0,0],[2,1,0],[0,1,0],[0,1,1],[0,2,1]])

print (predicted)
print(pred_prob)
