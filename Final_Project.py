#!/usr/bin/env python
# coding: utf-8

# # FIFA 19 Player

# In[1]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('data.csv')


# In[3]:


df.shape


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.drop(["Photo","Flag","Club Logo", "Value", "Special","LS", 
    "ST", "RS", "LW","LF","CF", "RF", "RW",
    "LAM", "CAM", "RAM", "LM", "LCM", "CM", 
    "RCM", "RM", "LWB", "LDM", "CDM", "RDM",
    "RWB","LB", "LCB", "CB", "RCB", "RB"
 ], inplace=True, axis=1)


# In[9]:


df.shape


# In[10]:


df.describe()


# In[13]:


pd.set_option("display.precision", 1)


# In[14]:


df.describe()


# In[18]:


df[
    df['Nationality'] == 'Italy'
]


# In[20]:


df[
    df['Nationality'] == 'Indonesia'
]


# In[21]:


df[
    df['Club'] == 'Chelsea'
]


# In[22]:


df.loc[df["Nationality"] == "England", "Club"].value_counts()


# In[24]:


mpl.style.use('ggplot') # optional: for ggplot-like style

# check for latest version of Matplotlib
print('Matplotlib version: ', mpl.__version__) # >= 2.0.0


# In[28]:


# group countries by continents and apply sum() function 
df_club = df.groupby('Club', axis=0).sum()

# note: the output of the groupby method is a `groupby' object. 
# we can not use it further until we apply a function (eg .sum())
print(type(df.groupby('Club', axis=0)))

df_club.head()


# In[34]:


# group countries by continents and apply sum() function 
df_reputation = df.groupby('International Reputation', axis=0).sum()

# note: the output of the groupby method is a `groupby' object. 
# we can not use it further until we apply a function (eg .sum())
print(type(df.groupby('International Reputation', axis=0)))

df_reputation.head()


# In[37]:


# autopct create %, start angle represent starting point
df_reputation['Age'].plot(kind='pie',
                            figsize=(5, 6),
                            autopct='%1.1f%%', # add in percentages
                            startangle=90,     # start angle 90° (Africa)
                            shadow=True,       # add shadow      
                            )

plt.title('Age Player')
plt.axis('equal') # Sets the pie chart to look like a circle.

plt.show()


# In[43]:



colors_list = ['lightgreen', 'yellowgreen', 'lightcoral', 'lightskyblue', 'pink']
explode_list = [0.1, 0, 0, 0.1, 0.1] 

df_reputation['Age'].plot(kind='pie',
                            figsize=(15, 6),
                            autopct='%1.1f%%', 
                            startangle=90,    
                            shadow=True,       
                            labels=None,       
                            pctdistance=1.12,    
                            colors=colors_list,  
                            explode=explode_list 
                            )


plt.title('Age Player', y=1.12) 

plt.axis('equal') 


plt.legend(labels=df_reputation.index, loc='upper left') 

plt.show()


# In[ ]:





# In[46]:


df.head()


# In[61]:



import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

get_ipython().run_line_magic('matplotlib', 'inline')


# In[65]:


import statsmodels.api as sm
from sklearn import datasets ## imports datasets from scikit-learn

data = datasets.load_boston() ## loads Boston dataset from datasets library 


# In[68]:


df.head()


# In[71]:


df[['Name', 'Potential']]


# In[73]:


x=df['Name'].values.reshape(-1,1)
y=df['Potential'].values.reshape(-1,1)


# In[81]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


# In[83]:


df.head()


# In[85]:


df.isnull().any()


# # Linear Regression

# In[87]:


# x = Age(independent variable)
x=df.iloc[:,3] 


# In[88]:


x.head()


# In[89]:


x.isnull().any()


# In[140]:


y=df.iloc[:,5]


# In[141]:


y.head()


# In[142]:


y.isnull().any()


# In[143]:


plt.bar(df["Age"],df["Potential"])
plt.xlabel("Age of Player")
plt.show()


# In[144]:


from sklearn.model_selection import train_test_split


# In[145]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[146]:


from sklearn.linear_model import LinearRegression


# In[147]:


regressor=LinearRegression()


# In[148]:


type(x_train)
type(y_train)


# In[149]:


x_train=np.array(x_train)
y_train=np.array(y_train)


# In[150]:


type(x_train)
type(y_train)


# In[151]:


x_train=x_train.reshape(-1,1)
y_train=y_train.reshape(-1,1)


# In[152]:


regressor.fit(x_train,y_train)


# In[153]:


x_test=np.array(x_test)
x_test=x_test.reshape(-1,1)
y_pred= regressor.predict(x_test)


# In[154]:


plt.scatter(x_train,y_train,color="red")
plt.xlabel("Age of Player")
plt.ylabel("Potential of Player")
plt.plot(x_train, regressor.predict(x_train),color="blue") # To draw line of regression
plt.show()


# In[155]:


plt.scatter(x_test,y_test,color="red")
plt.xlabel("Age of Player")
plt.ylabel("Potential of Player")
plt.plot(x_train, regressor.predict(x_train),color="blue")
plt.show()


# In[ ]:





# In[ ]:




