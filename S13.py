#!/usr/bin/env python
# coding: utf-8

# # Classification

# In[1]:


# import package
# siapin data
# bikin model dan training
# evaluasi model


# In[2]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# In[3]:


x = np.arange(10).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])


# In[4]:


print(x, y)


# In[5]:


model = LogisticRegression()


# In[6]:


model.fit(x, y)


# In[7]:


model.classes_


# In[8]:


print(model.intercept_, model.coef_)


# In[9]:


model.predict_proba(x)


# In[10]:


model.predict(x)


# In[11]:


model.score(x, y)


# In[13]:


cm = confusion_matrix(y, model.predict(x))

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()


# In[16]:


y_pred=model.predict(x)


# In[19]:


cm = confusion_matrix(y, y_pred)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0', 'Predicted 1'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0', 'Actual 1'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], color='red')
plt.show()


# In[18]:


print(classification_report(y, y_pred))


# In[20]:


x = np.arange(10).reshape(-1, 1)
y = np.array([0, 1, 0, 0, 1, 1, 1, 1, 1, 1])


# In[21]:


print(x, y)


# In[22]:


model=LogisticRegression()


# In[23]:


model.fit(x, y)


# In[24]:


p_pred=model.predict_proba(x)
y_pred=model.predict(x)
score=model.score(x, y)
conf_m=confusion_matrix(y, y_pred)
report=classification_report(y, y_pred)


# In[25]:


y


# In[26]:


y_pred


# In[27]:


print(score)
print(conf_m)
print(report)


# In[29]:


# Logistic Regression for Handwriting Recognition


# In[30]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[31]:


x, y = load_digits(return_X_y=True)


# In[32]:


print(x)


# In[33]:


print(y)


# In[34]:


train_test_split(x, y, test_size=0.2)


# In[35]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[36]:


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)


# In[41]:


model = LogisticRegression(solver='liblinear', C=0.05, multi_class='ovr',
                           random_state=0)
model.fit(x_train, y_train)


# In[38]:


x_test=scaler.transform(x_test)


# In[42]:


y_pred = model.predict(x_test)


# In[43]:


model.score(x_train, y_train)


# In[44]:


model.score(x_test, y_test)


# In[45]:


confusion_matrix(y_test, y_pred)


# In[46]:


cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.set_xlabel('Predicted outputs', color='black')
ax.set_ylabel('Actual outputs', color='black')
ax.xaxis.set(ticks=range(10))
ax.yaxis.set(ticks=range(10))
ax.set_ylim(9.5, -0.5)
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
plt.show()


# In[47]:


print(classification_report(y_test, y_pred))


# # KNN

# In[48]:


# Assigning features and label variables

# First Feature
weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']

# Second Feature
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

# Label or target varible
play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']


# In[49]:


from sklearn import preprocessing

le = preprocessing.LabelEncoder()


# In[50]:


weather_encoded=le.fit_transform(weather)
print(weather_encoded)


# In[51]:


temp_encoded=le.fit_transform(temp)
print(temp_encoded)


# In[52]:


label=le.fit_transform(play)
print(label)


# In[53]:


features=list(zip(weather_encoded,temp_encoded))

print(features)


# In[54]:


from sklearn.neighbors import KNeighborsClassifier


# In[55]:


model = KNeighborsClassifier(n_neighbors=3)

model.fit(features,label)


# In[56]:


model.predict([[1, 1]])


# In[57]:


from sklearn import datasets

wine = datasets.load_wine()


# In[58]:


print(wine.feature_names)


# In[59]:


print(wine.target_names)


# In[60]:


print(wine.data[0:5])


# In[61]:


print(wine.target)


# In[62]:


from sklearn.model_selection import train_test_split


# In[64]:


X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)


# In[65]:


from sklearn.neighbors import KNeighborsClassifier


# In[66]:


# K=5
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)


# In[67]:


from sklearn import metrics

metrics.accuracy_score(y_test, y_pred)


# In[68]:


# K=7
knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)


# In[69]:


from sklearn import metrics

metrics.accuracy_score(y_test, y_pred)


# In[71]:


error = []

for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))


# In[72]:


plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K')  
plt.xlabel('K')  
plt.ylabel('Error mean')


# In[73]:


# K=3
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)


# In[74]:


from sklearn import metrics

metrics.accuracy_score(y_test, y_pred)


# In[ ]:




