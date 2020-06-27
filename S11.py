#!/usr/bin/env python
# coding: utf-8

# In[1]:


# confidence interval


# In[2]:


import pandas as pd
import seaborn as sns
import scipy.stats as stats
import numpy as np
import random
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


sns.set(rc={'figure.figsize':(13, 7.5)})


# In[5]:


np.random.seed(42)

normal_distribution_male_mass_pound = np.random.normal(loc=181, scale=24, size=6000)


# In[6]:


normal_distribution_female_mass_pound = np.random.normal(loc=132, scale=22, size=6500)


# In[7]:


all_mass_value = np.concatenate((normal_distribution_female_mass_pound, normal_distribution_male_mass_pound), axis=0)


# In[26]:


df_ppl_mass = pd.DataFrame(data={'mass_pounds': all_mass_value})


# In[27]:


df_ppl_mass


# In[23]:


df_ppl_mass.head()


# In[28]:


sns.distplot(df_ppl_mass['mass_pounds'], color='darkslategrey')
plt.xlabel('mass [pounds]')
plt.ylabel('probability of occurance')
plt.title('Distribution of Mass of People', y=1.015, fontsize=20)

plt.show()


# In[29]:


# Calculation Population Mean

pop_mean_mass = df_ppl_mass['mass_pounds'].mean()
pop_mean_mass


# In[ ]:





# In[30]:


# Calculation Population Mean

pop_std_dev_mass = df_ppl_mass['mass_pounds'].std()
pop_std_dev_mass


# In[36]:


#sample 25 orang

sample_means=[]
n=25

for sample in range(0,300):
    #random sampling
    sample_values=np.random.choice(a=df_ppl_mass['mass_pounds'], size=n)
    sample_mean=np.mean(sample_values)
    sample_means.append(sample_mean)


# In[37]:


sns.distplot(sample_means)
plt.title('Distribution of sample means ($n=25$) of people\'s Mass in Pounds', y=1.015, fontsiza=20)
plt.ylabel('sample means')
plt.title('Distribution of Mass of People', y=1.015, fontsize=20)

plt.show()


# In[38]:


#Probability Distribution


# In[39]:


#uniform


# In[40]:


from scipy.stats import uniform


# In[42]:


n=10000
start=10
width=20
data_uu=niform=uniform.rvs(size=n, loc=start, scale=width)


# In[45]:


sns.distplot(data_uniform)
plt.title('Uniform Distribution')
plt.ylabel('frequency')
plt.show()


# In[47]:


#Bernoulli
from scipy.stats import bernoulli
data_bern=bernoulli.rvs(size=10000, p=0.6)


# In[48]:


sns.distplot(data_bern)
plt.title('Bernoulli Distribution')
plt.ylabel('frequency')
plt.show()


# In[49]:


# Binomial
from scipy.stats import binom
data_binom=binom.rvs(n=10, p=0.8, size=10000)


# In[50]:


sns.distplot(data_binom)
plt.title('Binomial Distribution')
plt.ylabel('frequency')
plt.show()


# In[59]:


#poisson
from scipy.stats import poisson
data_poisson=poisson.rvs(mu=3, size=10000)


# In[60]:


sns.distplot(data_poisson)
plt.title('Poisson Distribution')
plt.ylabel('frequency')
plt.show()


# In[53]:


#normal

from scipy.stats import norm
data_normal=norm.rvs(size=10000, loc=0, scale=1)


# In[54]:


sns.distplot(data_normal)
plt.title('Normal Distribution')
plt.ylabel('frequency')
plt.show()


# In[55]:


# Exponential

from scipy.stats import expon
data_expon=expon.rvs(scale=1, loc=0, size=1000)


# In[56]:


sns.distplot(data_expon)
plt.title('Exponential Distribution')
plt.ylabel('frequency')
plt.show()


# In[66]:


#Hypothesis Testing


# In[71]:


import statsmodels.api as sm


# In[81]:


url = "https://raw.githubusercontent.com/kshedden/statswpy/master/NHANES/merged/nhanes_2015_2016.csv"
da = pd.read_csv(url)
da.head()


# In[74]:


females = da[da["RIAGENDR"] == 2]
male = da[da["RIAGENDR"] == 1]


# In[75]:


n1 = len(females)
mu1 = females["BMXBMI"].mean()
sd1 = females["BMXBMI"].std()

(n1, mu1, sd1)


# In[76]:


n2 = len(male)
mu2 = male["BMXBMI"].mean()
sd2 = male["BMXBMI"].std()

(n2, mu2, sd2)


# In[78]:


sm.stats.ztest(females["BMXBMI"].dropna(), male["BMXBMI"].dropna(),alternative='two-sided')


# In[80]:


plt.title("Female BMI histogram",fontsize=16)
plt.hist(females["BMXBMI"].dropna(),edgecolor='k',color='pink',bins=25)
plt.show()

plt.title("Male BMI histogram",fontsize=16)
plt.hist(male["BMXBMI"].dropna(),edgecolor='k',color='blue',bins=25)
plt.show()


# In[ ]:




