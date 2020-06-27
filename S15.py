#!/usr/bin/env python
# coding: utf-8

# # Unsupervised Learning

# In[5]:


from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


X = -2 * np.random.rand(100, 2)
X1 = 1 + 2 * np.random.rand(50, 2)

X[50:100, :] = X1


# In[10]:


plt.scatter(X[ : , 0], X[ :, 1], s = 50, c = 'b')
plt.show()


# In[11]:


wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #within cluster sum of squares
plt.show()


# In[12]:


Kmean = KMeans(n_clusters=2)
Kmean.fit(X)


# In[13]:


Kmean.cluster_centers_


# In[14]:


plt.scatter(X[ : , 0], X[ : , 1], s =50, c='b')
plt.scatter(-0.94665068, -0.97138368, s=200, c='g', marker='s')
plt.scatter(2.01559419, 2.02597093, s=200, c='r', marker='s')
plt.show()


# In[16]:


import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# create dataset
X, y = make_blobs(
    n_samples=150, n_features=2,
    centers=3, cluster_std=0.5,
    shuffle=True, random_state=0
)

# plot
plt.scatter(
    X[:, 0], X[:, 1],
    c='white', marker='o',
    edgecolor='black', s=50
)
plt.show()


# In[17]:


from sklearn.cluster import KMeans

km = KMeans(
    n_clusters=3, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(X)


# In[18]:


# plot the 3 clusters
plt.scatter(
    X[y_km == 0, 0], X[y_km == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    X[y_km == 1, 0], X[y_km == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    X[y_km == 2, 0], X[y_km == 2, 1],
    s=50, c='lightblue',
    marker='v', edgecolor='black',
    label='cluster 3'
)

# plot the centroids
plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)
plt.legend(scatterpoints=1)
plt.grid()
plt.show()


# In[19]:


distortions = []
for i in range(1, 11):
    km = KMeans(
        n_clusters=i, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit(X)
    distortions.append(km.inertia_)

# plot
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()


# # PCA

# In[20]:


from sklearn.datasets import load_breast_cancer


# In[21]:


breast = load_breast_cancer()


# In[22]:


breast_data = breast.data


# In[23]:


breast_data.shape


# In[24]:


breast_labels = breast.target


# In[25]:


breast_labels.shape


# In[40]:


breast_dataset = pd.DataFrame(final_breast_data)


# In[33]:


import pandas as pd


# In[26]:


labels = np.reshape(breast_labels,(569,1))


# In[27]:


final_breast_data = np.concatenate([breast_data,labels],axis=1)


# In[ ]:





# In[37]:


features = breast.feature_names


# In[29]:


features


# In[30]:


features_labels = np.append(features,'label')


# In[38]:


print(features_labels)


# In[41]:


breast_dataset.columns = features_labels


# In[42]:


breast_dataset.head()


# In[43]:


breast_dataset['label'].replace(0, 'Benign',inplace=True)
breast_dataset['label'].replace(1, 'Malignant',inplace=True)


# In[44]:


breast_dataset.tail()


# In[45]:


from sklearn.preprocessing import StandardScaler
x = breast_dataset.loc[:, features].values
x = StandardScaler().fit_transform(x) # normalizing the features


# In[46]:




x.shape


# In[47]:


np.mean(x),np.std(x)


# In[48]:


feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
normalised_breast = pd.DataFrame(x,columns=feat_cols)
normalised_breast.tail()


# In[49]:


from sklearn.decomposition import PCA
pca_breast = PCA(n_components=2)
principalComponents_breast = pca_breast.fit_transform(x)


# In[54]:


principal_breast_Df = pd.DataFrame(data=principalComponents_breast,
                                   columns=['PC1', 'PC2'])


# In[55]:


principal_breast_Df.head()


# In[56]:


print('Explained variation per principal component: {}'.format(pca_breast.explained_variance_ratio_))


# In[58]:


import matplotlib as mpl
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('PC - 1',fontsize=20)
plt.ylabel('PC - 2',fontsize=20)
plt.title("Principal Component Analysis of Breast Cancer Dataset",fontsize=20)
targets = ['Benign', 'Malignant']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = breast_dataset['label'] == target
    plt.scatter(principal_breast_Df.loc[indicesToKeep, 'PC1']
               , principal_breast_Df.loc[indicesToKeep, 'PC2'], c = color, s = 50)

plt.legend(targets,prop={'size': 15})

plt.show()


# In[59]:


import pickle


# In[60]:


# Assigning features and label variables
weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny', 'Rainy','Sunny','Overcast','Overcast','Rainy']
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']
play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

# Import LabelEncoder
from sklearn import preprocessing

#creating labelEncoder
le = preprocessing.LabelEncoder()

# Converting string labels into numbers.
weather_encoded=le.fit_transform(weather)

# Converting string labels into numbers
temp_encoded=le.fit_transform(temp)
label=le.fit_transform(play)

#Combinig weather and temp into single listof tuples
features=list(zip(weather_encoded,temp_encoded))

#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(features,label)


# In[62]:


import pickle
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))


# In[63]:


loaded_model = pickle.load(open(filename, 'rb'))


# In[64]:


loaded_model.predict([[0,2]])


# In[ ]:




