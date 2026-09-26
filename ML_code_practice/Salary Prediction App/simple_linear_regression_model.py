import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv(r"C:\Users\suraj\Downloads\Salary_Data.csv")

x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state=0)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)
print(y_pred)

comparison=pd.DataFrame({'Actual':y_test,'predicted':y_pred})
print(comparison)

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('salary vs Experience (Test set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

model_coef=regressor.coef_
print(model_coef)

model_const=regressor.intercept_
print(model_const)

y_12=model_coef*12+model_const
print(y_12)

y_20=model_coef*12+model_const
print(y_20)

# Statistics for ML

dataset.mean()

dataset.median()

dataset['Salary'].median()

# Variance

dataset['Salary'].var()

# Standard deviation

dataset['Salary'].std()

# Coefficient of variation(cv)

from scipy.stats import variation

variation(dataset.values)

variation(dataset['Salary'])

# Correlatoin

dataset.corr()

dataset['Salary'].corr(dataset['YearsExperience'])

# Skewness

dataset.skew()

dataset['Salary'].skew()

# Standard Error

dataset.sem()

dataset['Salary'].sem()

# Z-score

import scipy.stats as stats

dataset.apply(stats.zscore)

stats.zscore(dataset['Salary'])

# SSR
y_mean=np.mean(y)

SSR=np.sum((y_pred-y_mean)**2)
print(SSR)

#SSE
y=y[0:6]
SSE=np.sum((y-y_pred)**2)
print(SSE)

# SST
mean_total=np.mean(dataset.values)
SST=np.sum((dataset.values-mean_total)**2)
print(SST)

#R2
r_square=1-SSR/SST
print(r_square)

bias=regressor.score(x_train,y_train)
print(bias)

Variance=regressor.score(x_test,y_test)
print(Variance)


import pickle

# save a file in write-binary mode and dump the model
filename='linear_regression_model.pkl'
with open(filename,'wb') as file:
    pickle.dump(regressor,file)

print('Model has been pickled and saved as linear_regression_model.pkl')

import os
os.getcwd()