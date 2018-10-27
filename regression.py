# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 14:22:10 2018

@author: ssn
"""

import pandas as pd
import matplotlib as plt

dataset= pd.read_csv('Salary_Data.csv')

x=dataset.iloc[:,0].values
y=dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)


x_train=x_train.reshape(-1,1)
y_train=y_train.reshape(-1,1)

x_test=x_test.reshape(-1,1)
y_test=y_test.reshape(-1,1)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)



