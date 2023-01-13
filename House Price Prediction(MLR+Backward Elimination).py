import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


dataset = pd.read_csv(r"C:\Users\dell\OneDrive\Documents\Data Science\14th,15th\TASK 12 -  TASK 17\TASK-17\kc_house_data.csv")


print(dataset.isnull().any())


print(dataset.dtypes)


dataset = dataset.drop(['id','date'], axis = 1)


with sns.plotting_context("notebook",font_scale=2.5):
    g = sns.pairplot(dataset[['sqft_lot','sqft_above','price','sqft_living','bedrooms']], 
                 hue='bedrooms', palette='tab20',size=6)
g.set(xticklabels=[]);


x = dataset.iloc[:, 1:].values
y = dataset.iloc[:, :1].values


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)


import statsmodels.formula.api as sm
x = np.append(arr = np.ones((21613,1)).astype(int), values = x, axis = 1)


import statsmodels.api as sm


x_opt = x[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()


x_opt = x[:,[0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()


bias = regressor.score(x_train,y_train)
bias


variance = regressor.score(x_test,y_test)
variance


#From the ML model which we built, We came to know wthat the "floors" attribute does not affects
#the prediction of the house price using Multiple Linear Regression Algorithm.
#Hence, We can easily remove that attributes.
s