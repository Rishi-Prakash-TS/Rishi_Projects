# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 10:55:30 2018

@author: ssn
"""

import pandas as pd
import matplotlib as plt

dataset = pd.read_csv('mul_reg.csv')

x=dataset.iloc[:,0:4].values
y=dataset.iloc[:,4].values

#IN ORDER TO SPLIT THE LABEL COLUMN OF STATE FROM TABLE AS D1,D2 AND D3 OF DIFFERENT STATES OF NEW YORK AND OTHERS

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()
x[:,3]= labelencoder.fit_transform(x[:,3]) #FIT WILL ONLY APPLY THE FORMULA AS IN A FUNCTION AND TRANSFORM KRYWORD WILL ONLY CONVERT THE VALUES BY APPLYING THE FORMULA IN FIT
onehotencoder= OneHotEncoder(categorical_features=[3])
x=onehotencoder.fit_transform(x).toarray()

#HERE THE LABELENCODER WILL ASSIGN VALUES TO THE DIFFERENT CATREGORY (HERE THE STATE), PRINT THE x[:,3] TO SEE THE RESULT
#AFTER USING LABELENCODER THE ONEFOTENCODER IS USED TO SPLIT THE STATE COLUMN INTO D1,D2 AND D3    

x=x[:,1:] #AS WE NEED TO REDUCE ANY ONE COLUMN FROM D1MD2,D3 IN ORDER TO REDUCE MULTICOLLINEARITY  

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

#THE LINEAR REGRESSION USED WILL BY DEFAULT GIVE US THE MODEL WITH BACKWARD ELIMINITION 

import statsmodels.formula.api as sf
regressor_OLS= sf.OLS(y,x).fit()
regressor_OLS.summary()

#IN ORDER TO REPRESENT THE BACKWARD ELIMINATION PROCESS WE USE THE FOLLOWING

import numpy as np
x=np.append(np.ones((50,1)).astype(int),x,axis=1)

x_b = x[:,[0,1,2,3,4,5]]
regressor_OLS= sf.OLS(y,x_b).fit()
regressor_OLS.summary()

x_b = x[:,[0,1,3,4,5]]
regressor_OLS= sf.OLS(y,x_b).fit()
regressor_OLS.summary()

x_b = x[:,[0,3,4,5]]
regressor_OLS= sf.OLS(y,x_b).fit()
regressor_OLS.summary()

x_b = x[:,[0,3,5]]
regressor_OLS= sf.OLS(y,x_b).fit()
regressor_OLS.summary()


x_b = x[:,[0,3]]
regressor_OLS= sf.OLS(y,x_b).fit()
regressor_OLS.summary()

















