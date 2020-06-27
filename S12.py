#!/usr/bin/env python
# coding: utf-8

# # Regression Model

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])


# In[3]:


x, y


# In[4]:


model = LinearRegression()


# In[5]:


model.fit(x, y)


# In[6]:


model = LinearRegression().fit(x, y)


# In[8]:


r_sq = model.score(x, y)
print(r_sq)


# In[9]:


# yhat = b0 + b1 X

print('intercept (b0):', model.intercept_)
print('slope (b1):', model.coef_)


# In[10]:


y_pred=model.predict(x)
print('predict response:', y_pred)


# In[ ]:





# In[12]:


model.predict([[66]])


# In[16]:


plt.scatter(x, y, alpha=0.5)
plt.plot(x, y_pred)
plt.title('Scatter plot x and y')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[17]:


import numpy as np
from sklearn.linear_model import LinearRegression

x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)


# In[19]:


x,y


# In[20]:


model = LinearRegression().fit(x, y)


# In[21]:


r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)


# In[22]:


y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')


# In[23]:


y_pred = model.intercept_ + np.sum(model.coef_ * x, axis=1)
print('predicted response:', y_pred, sep='\n')


# In[25]:


model.predict([[20, 100]])


# In[26]:


x_new = np.arange(10).reshape((-1, 2))


# In[27]:


x_new


# In[28]:


model.predict(x_new)


# In[29]:


# Polynomial Regression


# In[30]:


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# In[31]:


x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([15, 11, 2, 8, 25, 32])


# In[32]:


x, y


# In[34]:


# Transform Data

transformer=PolynomialFeatures(degree=2, include_bias=False)


# In[35]:


transformer.fit(x)


# In[36]:


x_ = transformer.transform(x)


# In[37]:


x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)


# In[46]:


x_


# In[53]:


model = LinearRegression().fit(x_, y)


# In[ ]:


model


# In[41]:


r_sq = model.score(x_, y)
print('R2:', r_sq)
print('B0:', model.intercept_)
print('B1:', model.coef_)


# In[47]:


y_pred = model.predict(x_)
print('predicted response:', y_pred)


# In[55]:


plt.scatter(x, y, alpha=0.5)
plt.plot(x, y_pred)
plt.title('scatter plot x and y')
plt.xlable('x')
plt.ylable('y')

plt.show()


# In[56]:


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Step 2a: Provide data
x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)

# Step 2b: Transform input data
x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)

# Step 3: Create a model and fit it
model = LinearRegression().fit(x_, y)

# Step 4: Get results
r_sq = model.score(x_, y)
intercept, coefficients = model.intercept_, model.coef_

# Step 5: Predict
y_pred = model.predict(x_)


# In[57]:


print('coefficient of determination:', r_sq)
print('intercept:', intercept)
print('coefficients:', coefficients, sep='\n')
print('predicted response:', y_pred, sep='\n')


# In[59]:


# Linear Regression With statsmodels


# In[60]:


import numpy as np
import statsmodels.api as sm


# In[61]:


x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)


# In[62]:


x = sm.add_constant(x)


# In[64]:


print(x)
print(y)


# In[65]:


model = sm.OLS(y, x)


# In[66]:


results = model.fit()


# In[67]:


print(results.summary())


# In[70]:


print('R2:', results.rsquared)
print(results.rsquared_adj)
print(results.params)


# In[71]:


results.fittedvalues


# In[72]:


results.predict(x)


# In[73]:


import statsmodels.api as sm
from sklearn import datasets


# In[74]:


data = datasets.load_boston()


# In[75]:


print(data.DESCR)


# In[76]:


import numpy as np
import pandas as pd


# In[77]:


df = pd.DataFrame(data.data, columns=data.feature_names)

target = pd.DataFrame(data.target, columns=["MEDV"])


# In[78]:


df.head()


# In[79]:


target.head()


# In[81]:


X = df["RM"]
y = target["MEDV"]


# In[82]:


model = sm.OLS(y, X).fit()
predictions = model.predict(X)

model.summary()


# In[ ]:





# In[83]:


plt.scatter(X, y, alpha=0.5)
plt.plot(X, predictions)
plt.title('Scatter plot x and y')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[84]:


from sklearn import linear_model


# In[85]:


from sklearn import datasets
data = datasets.load_boston()


# In[86]:


df = pd.DataFrame(data.data, columns=data.feature_names)

target = pd.DataFrame(data.target, columns=["MEDV"])


# In[88]:


x=df
y=target['MEDV']


# In[91]:


lm = linear_model.LinearRegression()


# In[93]:


lm.fit(X,y)


# In[94]:


lm.score(x, y)


# In[95]:


df = pd.read_csv('https://raw.githubusercontent.com/ardhiraka/PFDS_sources/master/CarPrice_Assignment.csv')


# In[96]:


df


# In[97]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[98]:


x=df['horsepower'].values.reshape(-1,1)
y=df['price'].values.reshape(-1,1)


# In[99]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[100]:


model=LinearRegression()


# In[101]:


model.fit(x_train, y_train)


# In[102]:


model.score(x_test, y_test)


# In[103]:


y_pred=model.predict(x_test)


# In[107]:


plt.scatter(x_test, y_test, alpha=0.5)
plt.plot(x_test, y_pred, c='r')
plt.title('scatter plot X and Y')
plt.xlabel('x')
plt.ylabel('y')

plt.show()


# In[ ]:




