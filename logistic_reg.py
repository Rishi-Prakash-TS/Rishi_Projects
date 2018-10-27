# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 14:17:47 2018

@author: ssn
"""

import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')
x=dataset.iloc[:,[2,3]].values 
y=dataset.iloc[:,4].values

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

y_pred= classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred,y_test)
print(cm)
accuracy=cm[0,0]+cm[1,1]/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
print(accuracy)

from sklearn.metrics import roc_curve,auc
fpr,tpr,thresholds = roc_curve(y_pred,y_test)
roc_auc= auc(fpr,tpr)
print(roc_auc)
#FPR STANDS FOR FALSE POSITIVE RATE AND TPR FOR TRUE POSITIVE RATE

import matplotlib.pyplot as plt
plt.plot(fpr,tpr,color='blue')
plt.show()



