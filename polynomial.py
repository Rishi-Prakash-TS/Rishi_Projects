# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 12:12:23 2018

@author: ssn
"""

import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')

x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values
# performing linear regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x,y)


# performing polynomial linear regression because the prediction is gettinng wrong
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)


poly_reg.fit(x_poly,y)
regressor2=LinearRegression()
regressor2.fit(x_poly,y)


#plotting linear regression graph
plt.scatter(x,y,color='red')
plt.plot(x,regressor2.predict(x_poly),color='blue')
plt.show()

#NEW VALUE PREDICTION
x_test=poly_reg.fit_transform(6.5)
regressor2.predict(x_test)


#DECISION TREE
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor()

regressor.fit(x,y)



y_predd=regressor.predict(x)

plt.scatter(x,y,color='red')
plt.plot(x,regressor2.predict(x),color='blue')
plt.show()

xi=regressor.fit_transform(x)




import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor()

regressor.fit(x,y)



y_predd=regressor.predict(x)

plt.scatter(x,y,color='red')
plt.plot(x,y_predd,color='blue')
plt.show()

xi=regressor.fit_transform(x)


