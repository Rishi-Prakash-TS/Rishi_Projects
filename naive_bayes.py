# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 14:39:50 2018

@author: ssn
"""

dataset = pd.read_csv('Social_Network_Ads.csv')
x=dataset.iloc[:,[2,3]].values 
y=dataset.iloc[:,4].values

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)
Y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,Y_pred)
print(cm)
accuracy=cm[0,0]+cm[1,1]/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
print(accuracy)




