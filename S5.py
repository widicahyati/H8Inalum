#!/usr/bin/env python
# coding: utf-8

# # Python for Data Science Sesi 5 - Pandas Introduction

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv('nbaallelo.csv')


# In[3]:


len(df)


# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


pd.set_option('display.max.columns', None)


# In[7]:


df.head()


# In[8]:


df.tail()


# In[9]:


df.head(3)


# In[10]:


df.tail(3)


# In[11]:


df.info()


# In[29]:


df.describe()


# ## 1.1 Exploring My Dataset

# In[16]:


df["team_id"].value_counts()


# In[17]:


df["fran_id"].value_counts()


# In[21]:


df.loc[df["fran_id"] == "lakers", "team_id"].value_counts()


# In[25]:


df.loc[df["team_id"] == "MNL", "date_game"].min()


# In[26]:


df.loc[df["team_id"] == "MNL", "date_game"].max()


# In[27]:


df.loc[df["team_id"] == "MNL", "date_game"].agg(("min", "max"))


# In[28]:


df.loc[df["team_id"] == "BOS", "pts"].sum()


# The Boston Celtics scored a total of 626,484 points in 5997 matches

# ## 1.2 Pandas Data Structure - Series

# In[30]:


revenues = pd.Series([5555, 7000, 1980])


# In[31]:


revenues


# In[32]:


revenues.index


# In[33]:


city_revenues = pd.Series(
        [4200, 8000, 6500],
        index=["Amsterdam", "Troronto", "Tokyo"])


# In[34]:


city_revenues


# In[35]:


city_revenues.values


# In[36]:


city_revenues.index


# In[37]:


city_employee_count = pd.Series({"Amsterdam": 5, "Tokyo": 8})


# In[38]:


city_employee_count


# In[39]:


city_employee_count.keys()


# In[40]:


"Tokyo" in city_employee_count


# In[55]:


"Amsterdam" in city_employee_count


# In[41]:


"Semarang" in city_employee_count


# ## 1.3 Pandas Data Structure - Data Frame

# In[42]:


city_revenues


# In[43]:


city_employee_count


# In[45]:


city_data=pd.DataFrame({
    "revenues": city_revenues,
    "employee_count": city_employee_count
})


# In[46]:


city_data


# In[47]:


city_data.index


# In[48]:


city_data.values


# In[49]:


city_data.axes


# In[51]:


city_data.axes[0]


# In[52]:


city_data.axes[1]


# In[54]:


city_data.keys()


# In[57]:


"Amsterdam" in city_data


# In[59]:


"revenues" in city_data


# In[60]:


"pts" in df


# In[61]:


df.index


# In[62]:


df.axes


# In[63]:


"pts" in df.keys()


# ## 1.4 Accessing Series Elements
# Joc
# Jloc

# In[64]:


city_revenues


# In[66]:


city_revenues["Troronto"]


# In[67]:


city_revenues[1]


# In[68]:


city_revenues[-1]


# In[69]:


city_revenues[1:]


# In[70]:


colors=pd.Series(
    ["red", "purple", "blue", "green", "yellow"],
    index=[1,2,3,5,8])


# In[71]:


colors


# In[72]:


colors[1]


# .loc=label index
# .iloc=positional index

# In[73]:


colors.loc[1]


# In[74]:


colors.iloc[1]


# In[75]:


colors.iloc[1:3]


# In[76]:


colors.loc[3:8]


# In[77]:


city_data


# In[81]:


city_data["revenues"]


# In[82]:


city_data.revenues


# In[85]:


toys=pd.DataFrame([
    {"name": "ball", "shape": "sphere"},
    {"name": "Rubik", "shape": "cube"}
])


# In[86]:


toys


# In[88]:


toys["shape"]


# In[90]:


toys.shape


# In[91]:


city_data


# In[92]:


city_data.loc["Amsterdam"]


# In[94]:


city_data.loc["Tokyo":"Troronto"]


# In[95]:


city_data.iloc[1]


# In[96]:


df.loc[2]


# In[97]:


df.iloc[-2]


# In[101]:


city_data.loc["Amsterdam": "Tokyo", "revenues"]


# In[103]:


df.head(5)


# In[104]:


df.loc[5555:5559, ["date_game", "team_id", "pts", "opp_id", "opp_pts"]]


# ## 1.6 Queying My Datasets

# In[106]:


two_years=df[
    df['year_id']>=2014
]


# In[107]:


two_years


# In[108]:


two_years.shape


# In[109]:


games_with_notes = df[
    df['notes'].notnull()
]


# In[110]:


games_with_notes


# In[111]:


games_with_notes.shape


# In[114]:


ers = df[
    df["fran_id"].str.endswith("ers")
]


# In[115]:


ers


# In[118]:


df[
    (df["pts"]>=100) &
    (df["opp_pts"]>=100) &
    (df["team_id"]=="BLB")
]


# In[123]:


# game tahun 1992, yang punya notes, yang team id nya mulai dari LA

df[
    (df["year_id"]==1992)&
    (df["notes"].notnull())&
    (df["team_id"].str.startswith("LA"))
]


# ## 1.7 Grouping Aggreating Data

# In[125]:


city_revenues


# In[127]:


city_revenues.sum()


# In[128]:


city_revenues.max()


# In[130]:


points=df["pts"]


# In[131]:


points.sum()


# In[132]:


df.head(3)


# In[134]:


df.groupby("fran_id", sort=False)["pts"].sum()


# In[135]:


df[
    (df["fran_id"]=="Spurs")&
    (df["year_id"]>2010)
].groupby(["year_id", "game_result"])["game_id"].count()


# In[136]:


df.head()


# In[138]:


df[
    (df["fran_id"]=="Knicks")&
    (df["year_id"]==2015)
].groupby(["game_location", "game_result"])["game_id"].count()


# ## 1.8 Manipulating Columns

# In[140]:


nba=df.copy()


# In[141]:


nba.shape


# In[142]:


nba["difference"]=nba.pts-nba.opp_pts


# In[144]:


nba.shape


# In[145]:


nba.head()


# In[146]:


nba["difference"].max()


# In[147]:


nba[
    nba["difference"]==68
]


# In[151]:


renamed_nba=renamed(
    columns={"game_result": "result", "game_location": "location"}
)


# In[150]:


renamed_nba.info()


# In[152]:


nba.shape


# In[153]:


elo_columns=["elo_i","elo_n", "opp_elo_i", "opp_elo_n"]


# In[ ]:


df


# In[154]:


nba.head()


# In[156]:


nba["game_location"].unique()


# In[157]:


df


# In[5]:


rows_without_missing_data.head()


# In[1]:


rows_without_missing_data = nbadropna()


# In[2]:


rows_without_missing_data.shape


# In[3]:


rows_without_missing_data.info()


# In[8]:


data_without_missing_columns.head()


# In[9]:


data_with_default_notes = nba.copy()


# ## 1.10 Combining Multiple Dataset

# ## 1.12 Standard Missing Values

# In[25]:


df = pd.read_csv('property_data.csv')


# In[26]:


df.head(10)


# In[27]:


df['ST_NUM']


# In[28]:


df['ST_NUM'].isnull()


# In[29]:


df


# In[31]:


df['NUM_BEDROOMS']


# In[32]:


df['NUM_BEDROOMS'].isnull()


# In[33]:


missing_values = ["n/a", "na", "--"]


# In[34]:


df = pd.read_csv("property_data.csv", na_values = missing_values)


# In[35]:


df['NUM_BEDROOMS']


# In[36]:


df['NUM_BEDROOMS'].isnull()


# ## 1.14 Unexpected Missing Values

# In[38]:


df


# In[39]:


df['OWN_OCCUPIED']


# In[40]:


df['OWN_OCCUPIED'].isnull()


# In[43]:


cnt = 0

for row in df['OWN_OCCUPIED']:
    try:
        int(row)
        df.loc[cnt, 'OWN_OCCUPIED']=np.nan
    except ValueError:
        pass
    cnt+=1


# In[44]:


df


# In[45]:


df.isnull()


# In[46]:


df.isnull().sum()


# In[47]:


df.isnull().sum().sum()


# ## 1.15 Replace

# In[48]:


df['ST_NUM'].fillna(125, inplace=True)


# In[50]:


data = pd.ExcelFile('obes.xls')


# In[51]:


data


# In[57]:


data.sheet_names


# In[58]:


data_age = data.parse(u'7.2', skiprows=4, skipfooter=14)


# In[59]:


data_age.head()


# In[61]:


data_age.rename(columns={u'Unnamed: 0': u'Year'}, inplace=True)


# In[62]:


data_age.head()


# In[63]:


data_age.dropna(inplace=True)


# In[64]:


data_age.head()


# In[65]:


data_age.isnull()


# In[66]:


data_age


# In[67]:


data_age_minus_total=data_age.drop('Total', axis=1)


# In[68]:


data_age['Under 16'].plot(label='Under 16', legend=True)
data_age['35-44'].plot(label='35-44', legend=True)


# ## 1.16 Time Series

# In[70]:


from datetime import datetime


# In[79]:


date_rng = pd.date_range(start='1/01/2020', end='1/08/2020', freq='H')


# In[80]:


date_rng


# In[81]:


df=pd.DataFrame(date_rng, columns=['date'])


# In[83]:


df['data']=np.random.randint(0, 100, size=(len(date_rng)))


# In[84]:


df.head()


# In[86]:


df['datetime']=pd.to_datetime(df['date'])


# In[87]:


df=df.set


# In[88]:


df.head()


# In[90]:


string_date_rng=[str(x) for x in date_rng]
string_date_rng


# In[91]:


string_date_rng_2=['Jun-01-2020', 'June-02-2020', 'June-03-2020']


# In[94]:


timestamp_date_rng_2=[datetime.strptime(x, '%B-%d-%Y') for x in string_date_rng_2]


# In[ ]:


df2=

