#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 23:32:59 2020

@author: liuyi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 21:42:16 2020

@author: liuyi
"""

#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Import Dataset
dataset = pd.read_csv('PRSA_Data.csv')
print(dataset.isnull().sum())
print(dataset.shape)
dataset.dropna(inplace = True)
dataset.columns = ['NO','YEAR','MONTH','DAY','HOUR','PM','DEWP','TEMP','PRES','CBWD','IWS','IS','IR']
print(dataset.isnull().sum())
dataset = dataset.drop(columns=['NO'])
x = dataset.drop(columns=['PM'])
y = dataset.iloc[:,4:5]



# In[3]:


#Data Cleaning & Rename & Values Assignments


from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x['CBWD'] = labelencoder_x.fit_transform(x['CBWD'])
x['YEAR'] = labelencoder_x.fit_transform(x['YEAR'])
x['PRES'] = labelencoder_x.fit_transform(x['PRES'])


# In[4]:


#Description of Data



pd.set_option('display.max_columns', None)
print(dataset.describe().to_string())
print(dataset.head(20))


dataset_corr = dataset.corr()
sns.heatmap(dataset_corr, center=0, annot=True)
plt.show()



# In[5]:


#Linear Model Prep
sns.lmplot(x="YEAR", y="PM", data=dataset, x_estimator=np.mean)
sns.lmplot(x="MONTH", y="PM", data=dataset, x_estimator=np.mean)
sns.lmplot(x="DAY", y="PM", data=dataset, x_estimator=np.mean)
sns.lmplot(x="HOUR", y="PM", data=dataset, x_estimator=np.mean)
sns.lmplot(x="DEWP", y="PM", data=dataset, x_estimator=np.mean)
sns.lmplot(x="TEMP", y="PM", data=dataset, x_estimator=np.mean)
sns.lmplot(x="PRES", y="PM", data=dataset, x_estimator=np.mean)
sns.lmplot(x="IWS", y="PM", data=dataset, x_estimator=np.mean)
sns.lmplot(x="IR", y="PM", data=dataset, x_estimator=np.mean)
sns.lmplot(x="IS", y="PM", data=dataset, x_estimator=np.mean)


# In[6]:


#Visualization relationships between PM2.5 and each Varibles
x_year = dataset.iloc[:,1:2]
plt.scatter(x_year, y)
plt.xlabel('YEAR')
plt.ylabel('PM')
plt.title('YEAR vs PM')
plt.savefig('YEAR vs PM.png')

x_month = dataset.iloc[:,2:3]
plt.scatter(x_month, y)
plt.xlabel('MONTH')
plt.ylabel('PM')
plt.title('MONTH vs PM')
plt.savefig('MONTH vs PM.png')

x_day = dataset.iloc[:,3:4]
plt.scatter(x_day, y)
plt.xlabel('DAY')
plt.ylabel('PM')
plt.title('DAY vs PM')
plt.savefig('DAY vs PM.png')


x_hour = dataset.iloc[:,4:5]
plt.scatter(x_hour, y)
plt.xlabel('HOUR')
plt.ylabel('PM')
plt.title('HOUR vs PM')
plt.savefig('HOUR vs PM.png')

x_DEWP = dataset.iloc[:,6:7]
plt.scatter(x_DEWP, y)
plt.xlabel('DEWP')
plt.ylabel('PM')
plt.title('DEWP vs PM')
plt.savefig('DEWP vs PM.png')

x_TEMP = dataset.iloc[:,7:8]
plt.scatter(x_TEMP, y)
plt.xlabel('TEMP')
plt.ylabel('PM')
plt.title('TEMP vs PM')
plt.savefig('TEMP vs PM.png')

x_PRES = dataset.iloc[:,8:9]
plt.scatter(x_PRES, y)
plt.xlabel('PRES')
plt.ylabel('PM')
plt.title('PRES vs PM')
plt.savefig('PRES vs PM')

x_cbwd = dataset.iloc[:,9:10]
uniques, x_cbwd_nums = np.unique(x_cbwd, return_inverse=True)
plt.scatter(x_cbwd_nums, y)
plt.xlabel('CBWD')
plt.ylabel('PM')
plt.title('CBWD vs PM')
plt.savefig('CBWD vs PM.png')

x_Iws = dataset.iloc[:,10:11]
plt.scatter(x_Iws, y)
plt.xlabel('IWS')
plt.ylabel('PM')
plt.title('IWS vs PM')
plt.savefig('IWS vs PM.png')

x_Is = dataset.iloc[:,11:12]
plt.scatter(x_Is, y)
plt.xlabel('Is')
plt.ylabel('PM')
plt.title('IS vs PM')
plt.savefig('IS vs PM.png')

x_Ir = dataset.iloc[:,12:13]
plt.scatter(x_Ir, y)
plt.xlabel('Ir')
plt.ylabel('PM')
plt.title('IR vs PM')
plt.savefig('IR vs PM.png')


# In[7]

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2,random_state=0)

# In[8]
#Build Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

#Evaluation of the Regression Model
regressor.score(x_test, y_test)
regressor.score(x_train, y_train)

#Building the optimal
import statsmodels.api as sm
x_train = np.append(arr = np.ones((33405, 1)), values = x_train, axis = 1)
x_opt = x_train[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
regressor_OLS = sm.OLS(endog = y_train, exog = x_opt).fit()
regressor_OLS.summary()

# In[9]:


#logistic regression
#Build Logistic Regression Model

dataset.PM.loc[dataset.PM<=100] = 0
dataset.PM.loc[(dataset.PM <= 200) & (dataset.PM>100)] = 1
dataset.PM.loc[dataset.PM >200] = 2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=1)
lr = LogisticRegression(penalty='none',solver='newton-cg',multi_class='multinomial', max_iter=1)
print("start training")
lr.fit(x_train,y_train)
print("end training")

print("Logistic Regression模型训练集的准确率：%.3f" %lr.score(x_train, y_train))
print("Logistic Regression模型测试集的准确率：%.3f" %lr.score(x_test, y_test))



# In[10]:

get_ipython().system('pip install mlxtend ')
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from mlxtend.regressor import StackingCVRegressor

y = dataset['PM']
x = dataset.drop(columns=['PM','CBWD'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
lr = LinearRegression()
dtr = DecisionTreeRegressor()
svr_rbf = SVR(kernel='rbf', gamma='auto')
knr = KNeighborsRegressor()
ridge = Ridge()
lasso = Lasso()
regression_models = [lr, dtr, svr_rbf, knr, ridge, lasso]

sclf = StackingCVRegressor(regression_models, meta_regressor=ridge)
sclf.fit(x_train, y_train)
pred = sclf.predict(x_test)

print(sclf.score(x_train, y_train))
%matplotlib inline
plt.scatter([i*10 for i in range(len(y_test))], y_test, c='red', lw=1)
plt.plot([i*10 for i in range(len(y_test))], pred, c='black', lw=1)
plt.show()


`# In[ ]:




