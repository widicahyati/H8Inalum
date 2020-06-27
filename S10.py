#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
x = np.arange(10, 20)
x


# In[2]:


y = np.array([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])
y


# In[3]:


r = np.corrcoef(x, y)
r


# In[4]:


print(r[0, 1])
print(r[1, 0])


# In[5]:


import scipy.stats


# In[7]:


print(x)
print(y)


# In[8]:


scipy.stats.pearsonr(x, y)  


# In[9]:


scipy.stats.spearmanr(x, y) 


# In[10]:


scipy.stats.kendalltau(x, y)


# In[15]:


print(scipy.stats.pearsonr(x, y)[0])


# In[12]:


print(scipy.stats.spearmanr(x, y)[0])   


# In[13]:


print(scipy.stats.kendalltau(x, y)[0])


# In[16]:


import pandas as pd

x = pd.Series(range(10, 20))
x


# In[17]:


y = pd.Series([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])
y


# In[19]:


print(x)
print(y)


# In[20]:


x.corr(y)   


# In[21]:


y.corr(x)


# In[22]:


x.corr(y, method='spearman')


# In[23]:


x.corr(y, method='kendall')


# In[24]:


import numpy as np
import scipy.stats

x = np.arange(10, 20)
y = np.array([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])


# In[25]:


print(x)
print(y)


# In[26]:


result = scipy.stats.linregress(x, y)


# In[27]:


result


# In[28]:


result.slope


# In[29]:


result.intercept


# In[32]:


xy=np.array([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
[2,  1,  4,  5,  8, 12, 18, 25, 96,  48]])


# In[33]:


scipy.stats.linregress(xy)


# In[34]:


xy.T


# In[35]:


scipy.stats.linregress(xy.T)


# In[36]:


scipy.stats.linregress(np.arange(3), np.array([2, np.nan, 5]))


# In[37]:


r, p = scipy.stats.pearsonr(x, y)


# In[38]:


print(r)
print(p)


# In[39]:


np.corrcoef(x, y)


# In[40]:


np.corrcoef(xy)


# In[41]:


xyz = np.array([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                [2, 1, 4, 5, 8, 12, 18, 25, 96, 48],
                [5, 3, 2, 1, 0, -2, -8, -11, -15, -16]])


# In[42]:


np.corrcoef(xyz)


# In[43]:


"""
         x        y        z

x     1.00     0.76    -0.97
y     0.76     1.00    -0.83
z    -0.97    -0.83     1.00
"""


# In[44]:


arr_with_nan = np.array([[0, 1, 2, 3],
                         [2, 4, 1, 8],
                         [2, 5, np.nan, 2]])


# In[45]:


np.corrcoef(arr_with_nan)


# In[46]:


xyz.T


# In[47]:


np.corrcoef(xyz.T, rowvar=False)


# In[50]:


import pandas as pd


# In[49]:


x = pd.Series(range(10, 20))


# In[51]:


y = pd.Series([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])


# In[52]:


z = pd.Series([5, 3, 2, 1, 0, -2, -8, -11, -15, -16])


# In[54]:


xy = pd.DataFrame({'x-values': x, 'y-values': y})


# In[55]:


xyz = pd.DataFrame({'x-values': x, 'y-values': y, 'z-values': z})


# In[57]:


print(x)
print(y)
print(z)
print(xy)
print(xyz)


# In[60]:


x.corr(y)


# In[61]:


xy.corr()


# In[62]:


xyz.corr()


# In[66]:


u, u_with_nan = pd.Series([1, 2, 3]), pd.Series([1, 2, np.nan, 3])
v, w = pd.Series([1, 4, 8]), pd.Series([1, 4, 154, 8])


# In[67]:


u.corr(v)


# In[68]:


u_with_nan.corr(w)


# In[69]:


corr_matrix = xy.corr()


# In[70]:


corr_matrix


# In[71]:


corr_matrix.at['x-values', 'y-values']


# In[72]:


corr_matrix.iat[0, 1]


# In[73]:


xyz.corr()


# In[74]:


xy.corrwith(z)


# In[75]:


# Rank Correlation


# In[76]:


# Scipy


# In[77]:


x = np.arange(10, 20)
y = np.array([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])
z = np.array([5, 3, 2, 1, 0, -2, -8, -11, -15, -16])


# In[78]:


scipy.stats.rankdata(x)


# In[79]:


scipy.stats.rankdata(y)


# In[80]:


scipy.stats.rankdata(z)


# In[81]:


scipy.stats.rankdata([8, 2, 0, 2])


# In[82]:


scipy.stats.rankdata([8, np.nan, 0, 2])


# In[83]:


np.argsort(y) + 1


# In[89]:


result = scipy.stats.spearmanr(x, y)


# In[90]:


result.correlation


# In[91]:


rho, p = scipy.stats.spearmanr(x, y)


# In[93]:


rho


# In[86]:


xy = np.array([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
               [2, 1, 4, 5, 8, 12, 18, 25, 96, 48]])


# In[94]:


rho, p = scipy.stats.spearmanr(xy, axis=1)


# In[95]:


rho


# In[98]:


xyz = np.array([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                [2, 1, 4, 5, 8, 12, 18, 25, 96, 48],
                [5, 3, 2, 1, 0, -2, -8, -11, -15, -16]])


# In[99]:


corr_matrix, p_matrix = scipy.stats.spearmanr(xyz, axis=1)


# In[100]:


corr_matrix


# In[101]:


result = scipy.stats.kendalltau(x, y)


# In[102]:


result.correlation


# In[103]:


tau, p = scipy.stats.kendalltau(x, y)


# In[104]:


tau


# In[105]:


# Pandas


# In[106]:


x, y, z = pd.Series(x), pd.Series(y), pd.Series(z)
xy = pd.DataFrame({'x-values': x, 'y-values': y})
xyz = pd.DataFrame({'x-values': x, 'y-values': y, 'z-values': z})


# In[107]:


x.corr(y, method='spearman')


# In[108]:


xy.corr(method='spearman')


# In[109]:


xyz.corr(method='spearman')


# In[110]:


xy.corrwith(z, method='spearman')


# In[111]:


x.corr(y, method='kendall')


# In[112]:


xy.corr(method='kendall')


# In[118]:


xyz.corr(method='kendall')


# In[119]:


xy.corrwith(z, method='kendall')


# In[120]:


# Visualization of correlation


# In[121]:


import matplotlib.pyplot as plt
plt.style.use('ggplot')


# In[122]:


import numpy as np
import scipy.stats
x = np.arange(10, 20)
y = np.array([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])
z = np.array([5, 3, 2, 1, 0, -2, -8, -11, -15, -16])
xyz = np.array([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                [2, 1, 4, 5, 8, 12, 18, 25, 96, 48],
                [5, 3, 2, 1, 0, -2, -8, -11, -15, -16]])


# In[123]:


slope, intercept, r, p, stderr = scipy.stats.linregress(x, y)


# In[124]:


line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
line


# In[125]:


fig, ax = plt.subplots()
ax.plot(x, y, linewidth=0, marker='s', label='Data points')
ax.plot(x, intercept + slope * x, label=line)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(facecolor='white')
plt.show()


# In[126]:


corr_matrix = np.corrcoef(xyz).round(decimals=2)


# In[127]:


corr_matrix


# In[128]:


fig, ax = plt.subplots()
im = ax.imshow(corr_matrix)
im.set_clim(-1, 1)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1, 2), ticklabels=('x', 'y', 'z'))
ax.yaxis.set(ticks=(0, 1, 2), ticklabels=('x', 'y', 'z'))
ax.set_ylim(2.5, -0.5)
for i in range(3):
    for j in range(3):
        ax.text(j, i, corr_matrix[i, j], ha='center', va='center',
                color='r')
cbar = ax.figure.colorbar(im, ax=ax, format='% .2f')
plt.show()


# In[4]:


fig, ax = plt.subplots()
ax.boxplot((x, y, z),
          vert=False,
          showmeans=True,
          meanline=True,
          labels=('x', 'y', 'z'),
          medianprops={'linewidth': 2, 'color': 'purple'},
          meanprops={'linewidth': 2, 'color': 'red'}
          )


# In[5]:


# Histogram


# In[7]:


hist, bin_edges = np.histogram(a, bins=10)


# In[8]:


hist


# In[9]:


bin_edges


# In[10]:


# pie chart


# In[11]:


x, y, z = 128, 256, 1024


# In[13]:


fig, ax = plt.subplots()
ax.pie((x, y, z), labels=('x', 'y', 'z'), autopct='%.1.1%%')

plt.show()


# In[ ]:




