#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[4]:


get_ipython().system('pip install wordcloud')


# In[5]:


from wordcloud import WordCloud, STOPWORDS


# In[8]:


alice_novel = open('alice_novel.txt', 'r', encoding="utf-8").read()


# In[9]:


stopwords=set(STOPWORDS)


# In[10]:


alice_wc=WordCloud(
    background_color='white',
    stopwords=stopwords,
    max_words=2000)

alice_wc.generate(alice_novel)


# In[13]:


plt.imshow(alice_wc, interpolation='bilinear')
plt.axis('off')

plt.show()


# In[16]:


stopwords.add('said')

alice_wc.generate(alice_novel)

fig=plt.figure()

fig.set_figwidth(14)
fig.set_figheight(18)

plt.imshow(alice_wc, interpolation='bilinear')
plt.axis('off')

plt.show()


# In[17]:


text=" ".join(country for country in df_can.index)


# In[18]:


get_ipython().system('pip install seaborn')


# In[19]:


import seaborn as sns


# In[21]:


get_ipython().system('pip install folium')

import folium


# In[27]:


df_incidents=pd.read_csv('Police_Department_Incidents_-_Previous_Year_2016_.csv')

