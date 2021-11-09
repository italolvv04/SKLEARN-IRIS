#!/usr/bin/env python
# coding: utf-8

# In[31]:


from sklearn import datasets
import pandas as pd
import seaborn as sns


# In[32]:


iris = datasets.load_iris()
iris_df = pd.DataFrame(data= iris.data, columns=iris.feature_names)
iris_df['label'] = iris.target
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)


# In[33]:


# É passada um parametro para visualização de todos os dados do dataframe
pd.set_option('display.max_rows', None)


# In[34]:


iris_df


# In[35]:


sns.pairplot(iris_df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 
                     'petal width (cm)', 'species']], hue='species')


# In[36]:


from sklearn import svm
from sklearn.model_selection import train_test_split


# In[37]:


X = iris.data
y = iris.target


# In[38]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=13)


# In[39]:


clf = svm.SVC(C=1.0)


# In[40]:


clf.fit(X_train, y_train)


# In[41]:


clf.predict(X_test)


# In[42]:


y_test


# In[43]:


clf.score(X_test, y_test)

