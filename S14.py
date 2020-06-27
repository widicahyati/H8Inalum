#!/usr/bin/env python
# coding: utf-8

# # Classification 2

# In[1]:


# Assigning features and label variables

weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny', 'Rainy','Sunny','Overcast','Overcast','Rainy']

temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']


# In[3]:


from sklearn import preprocessing

le = preprocessing.LabelEncoder()

weather_encoded=le.fit_transform(weather)
temp_encoded=le.fit_transform(temp)
label=le.fit_transform(play)
features=list(zip(weather_encoded,temp_encoded))


# In[4]:


from sklearn.naive_bayes import GaussianNB


# In[5]:


model = GaussianNB()
model.fit(features,label)


# In[6]:


print(model.predict([[0, 1]]))


# In[7]:


# NB Multiple labels


# In[8]:


from sklearn import datasets

wine = datasets.load_wine()


# In[9]:


wine.data.shape


# In[10]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3) 


# In[11]:


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)


# In[12]:


from sklearn import metrics

metrics.accuracy_score(y_test, y_pred)


# # Decision Tree

# In[13]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[19]:


col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

pima = pd.read_csv("https://raw.githubusercontent.com/ardhiraka/PFDS_sources/master/diabetes.csv", header=None, names=col_names)


# In[ ]:





# In[20]:


pima.head()


# In[21]:


pima.info()


# In[23]:


pima.columns


# In[24]:


numer = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree', 'label']
for col in numer: 
    pima[col] = pd.to_numeric(pima[col], errors='coerce')


# In[25]:


pima.info()


# In[26]:


feature_cols = ['pregnant', 'glucose', 'bp','skin', 'insulin', 'bmi', 'pedigree', 'bp']

X = pima[feature_cols] 
y = pima['label']


# In[27]:


pima.info()


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[29]:


x = pima.drop('label', 1)


# In[33]:


numer = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree', 'label']
for col in numer: # coerce for missing values
    pima[col] = pd.to_numeric(pima[col], errors='coerce')


# In[34]:


pima.dropna(inplace=True)


# In[39]:


feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']

X = pima[feature_cols]
y = pima.label 


# In[40]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 


# In[37]:


clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)


# In[41]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[42]:


get_ipython().system('pip install graphviz')
get_ipython().system('pip install pydotplus')


# In[44]:


get_ipython().system('conda install python-graphviz -y')


# In[45]:


get_ipython().system('pip install pydotplus')


# In[46]:


import sklearn.tree as tree
import pydotplus
from sklearn.externals.six import StringIO 
from IPython.display import Image
dot_data = StringIO()
tree.export_graphviz(clf, 
 out_file=dot_data, 
 class_names=['0','1'], # the target names.
 feature_names=feature_cols, # the feature names.
 filled=True, # Whether to fill in the boxes with colours.
 rounded=True, # Whether to round the corners of the boxes.
 special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
Image(graph.create_png())


# In[47]:


# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# # Random Forest

# In[51]:


from sklearn import datasets

iris = datasets.load_iris()


# In[56]:


iris.feature_names


# In[52]:


iris.data[0:5]


# In[53]:


iris.target


# In[54]:


from sklearn.model_selection import train_test_split


# In[57]:


X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)


# In[72]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)


# In[69]:


rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)


# In[70]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[71]:


rf.predict([[3, 5, 4, 2]])


# # SVM

# In[74]:


from sklearn import datasets

cancer = datasets.load_breast_cancer()


# In[75]:


cancer.feature_names


# In[76]:


cancer.target_names


# In[77]:


cancer.data.shape


# In[78]:


cancer.data[0:5]


# In[79]:


cancer.target


# In[81]:


X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, target_size=0.3)


# In[82]:


from sklearn import svm


# In[84]:


model = svm.SVC()

model.fit(X_train, y_train)


# In[85]:


y_pred = model.predict(X_test)


# In[86]:


print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))


# In[87]:


model = svm.SVC(kernel='linear')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))


# In[88]:


import pandas as pd
import numpy as np


# In[89]:


train = pd.read_csv('https://raw.githubusercontent.com/ardhiraka/PFDS_sources/master/Final_Dataset/train.csv')
test = pd.read_csv('https://raw.githubusercontent.com/ardhiraka/PFDS_sources/master/Final_Dataset/test.csv')


# In[90]:


train.head()


# In[92]:


train.info()


# In[91]:


train.describe()


# In[93]:


train['Education'].unique()


# In[94]:


train['Property_Area'].unique()


# In[95]:


train.isnull().sum()


# In[97]:


train.fillna(train.mean(), inplace=True)


# In[98]:


train.isnull().sum()


# In[99]:


train.Gender.fillna(train.Gender.mode()[0], inplace=True)

train.isnull().sum()


# In[101]:


train.Married.fillna(train.Married.mode()[0],inplace=True)
train.Dependents.fillna(train.Dependents.mode()[0],inplace=True) 
train.Self_Employed.fillna(train.Self_Employed.mode()[0],inplace=True)  

train.isnull().sum() 


# In[103]:


# import numpy as np

train.Loan_Amount_Term=np.log(train.Loan_Amount_Term)


# In[104]:


train.head()


# In[106]:


X = train.drop('Loan_Status', 1)
y = train.Loan_Status


# In[109]:


X = train.drop(['Loan_ID', 'Loan_Status'], 1)


# In[110]:


X = pd.get_dummies(X)


# In[112]:


X.head()


# In[113]:


y.head()


# In[114]:


from sklearn.model_selection import train_test_split


# In[115]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[119]:


from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X, y, test_size=0.2)


# In[121]:


x_train.info()


# In[122]:


#(a)LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression

model=LogisticRegression()

model.fit(x_train,y_train)


# In[123]:


pred_cv=model.predict(x_cv)


# In[124]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
print(accuracy_score(y_cv,pred_cv))
matrix=confusion_matrix(y_cv,pred_cv)
print(matrix)


# In[125]:


#(b)DECISION TREE ALGORITHM

from sklearn import tree
dt=tree.DecisionTreeClassifier(criterion='gini')
dt.fit(x_train,y_train)


# In[126]:


pred_cv1=dt.predict(x_cv)


# In[127]:


print(accuracy_score(y_cv,pred_cv1))
matrix1=confusion_matrix(y_cv,pred_cv1)
print(matrix1)


# In[128]:


#(c)RANDOM FOREST ALGORITHM

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)


# In[129]:


pred_cv2=rf.predict(x_cv)


# In[130]:


print(accuracy_score(y_cv,pred_cv2))
matrix2=confusion_matrix(y_cv,pred_cv2)
print(matrix2)


# In[131]:


#(d)SUPPORT VECTOR MACHINE (SVM) ALGORITHM

from sklearn import svm
svm_model=svm.SVC()
svm_model.fit(x_train,y_train)


# In[132]:


pred_cv3=svm_model.predict(x_cv)


# In[133]:


print(accuracy_score(y_cv,pred_cv3))
matrix3=confusion_matrix(y_cv,pred_cv3)
print(matrix3)


# In[134]:


from sklearn.naive_bayes import GaussianNB 
nb=GaussianNB()
nb.fit(x_train,y_train)

pred_cv4=nb.predict(x_cv)

print(accuracy_score(y_cv,pred_cv4))
matrix4=confusion_matrix(y_cv,pred_cv4)
print(matrix4)


# In[135]:


from sklearn.neighbors import KNeighborsClassifier
kNN=KNeighborsClassifier()
kNN.fit(x_train,y_train)

pred_cv5=kNN.predict(x_cv)

print(accuracy_score(y_cv,pred_cv5))
matrix5=confusion_matrix(y_cv,pred_cv5)
print(matrix5)


# In[136]:


print("Logistic Regression:", accuracy_score(y_cv,pred_cv))
print("Decision Tree:", accuracy_score(y_cv,pred_cv1))
print("Random Forest:", accuracy_score(y_cv,pred_cv2))
print("SVM:", accuracy_score(y_cv,pred_cv3))
print("Naive Bayes:", accuracy_score(y_cv,pred_cv4))
print("KNN:", accuracy_score(y_cv,pred_cv5))


# In[137]:


pd.DataFrame(pred_RF, columns=['Predictions']).to_csv('H8_RF_Loan_Prediction.csv')


# In[ ]:




