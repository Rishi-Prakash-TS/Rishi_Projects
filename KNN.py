# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 11:57:09 2018

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
x_test=sc.fit_transform(x_test)

#FITTING K-NN MODEL
from sklearn.neighbors import KNeighborsClassifier
classifier =KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
#MINKOWSKI CHOOSES WHICH DISTANCE FORMULA TO USE, IF P=1 MANHATTEN DISTANCE AND P=2 IT TAKES EUCLADIAN DISTANCE
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy=cm[0,0]+cm[1,1]/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
print(accuracy)















