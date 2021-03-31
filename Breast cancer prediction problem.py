#!/usr/bin/env python
# coding: utf-8

# In[1]:


# classification - decision tree implemetaiton
# aim : to predict the chances of having breast cancer using classification algorithm

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# keeps the plots in one place. calls image as static pngs
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt # side-stepping mpl backend
import matplotlib.gridspec as gridspec # subplots
import mpld3 as mpl

#Import models from scikit learn module:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics


# In[2]:


# Load the Data
df = pd.read_csv("breast_cancer_data.csv",header = 0)
df.head()


# In[3]:


# Cleaning and Preparing the data
df.drop('id',axis=1,inplace=True)
df.drop('Unnamed: 32',axis=1,inplace=True)
# size of the dataframe
len(df)


# In[4]:


df.diagnosis.unique()


# In[5]:


df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
df.head()


# In[6]:


# Explore the data
df.describe()


# In[7]:


df.describe()
plt.hist(df['diagnosis'])
plt.title('Diagnosis (M=1 , B=0)')
plt.show()


# In[8]:


# nucleus features vs diagnosis
features_mean=list(df.columns[1:11])
# split dataframe into two based on diagnosis
dfM=df[df['diagnosis'] ==1]
dfB=df[df['diagnosis'] ==0]


# In[9]:


#Stack the data
plt.rcParams.update({'font.size': 8})
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8,10))
axes = axes.ravel()
for idx,ax in enumerate(axes):
    ax.figure
    binwidth= (max(df[features_mean[idx]]) - min(df[features_mean[idx]]))/50
    ax.hist([dfM[features_mean[idx]],dfB[features_mean[idx]]], bins=np.arange(min(df[features_mean[idx]]), max(df[features_mean[idx]]) + binwidth, binwidth) , alpha=0.8,stacked=True, density = True, label=['M','B'],color=['r','g'])
    ax.legend(loc='upper right')
    ax.set_title(features_mean[idx])
plt.tight_layout()
plt.show()


# In[10]:


# Observations
# 1. mean values of cell radius, perimeter, area, compactness, concavity and concave points can be used in classification 
#   of the cancer. Larger values of these parameters tends to show a correlation with malignant tumors.
# 2. mean values of texture, smoothness, symmetry or fractual dimension does not show a particular preference of one diagnosis 
#    over the other. In any of the histograms there are no noticeable large outliers that warrants further cleanup.


# In[11]:


# Creating a test set and a training set
traindf, testdf = train_test_split(df, test_size = 0.3)


# In[12]:


#Generic function for making a classification model and accessing the performance. 
 
def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  #Make predictions on training set:
  predictions = model.predict(data[predictors])
  
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print("Accuracy : %s" % "{0:.3%}".format(accuracy))
    
  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome]) 


# In[13]:


predictor_var = ['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean']
outcome_var='diagnosis'
model=LogisticRegression()
classification_model(model,traindf,predictor_var,outcome_var)


# In[14]:


# Decision Tree Model

predictor_var = ['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean']
model = DecisionTreeClassifier()
classification_model(model,traindf,predictor_var,outcome_var)


# In[15]:


# Here we are over-fitting the model probably due to the large number of predictors. 
# Let use a single predictor, the obvious one is the radius of the cell.


# In[16]:


predictor_var = ['radius_mean']
model = DecisionTreeClassifier()
classification_model(model,traindf,predictor_var,outcome_var)


# In[17]:


# The accuracy of the prediction is much much better here. But does it depend on the predictor?
# Using a single predictor gives a 97% prediction accuracy for this model.


# In[18]:


# Randome Forest
# Use all the features of the nucleus
predictor_var = features_mean
model = RandomForestClassifier(n_estimators=100,min_samples_split=25, max_depth=7, max_features=2)
classification_model(model, traindf,predictor_var,outcome_var)


# In[19]:


#Create a series with feature importances:
featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print(featimp)


# In[20]:


# Using top 5 features
predictor_var = ['concave points_mean','area_mean','radius_mean','perimeter_mean','concavity_mean',]
model = RandomForestClassifier(n_estimators=100, min_samples_split=25, max_depth=7, max_features=2)
classification_model(model,traindf,predictor_var,outcome_var)


# In[21]:


predictor_var =  ['radius_mean']
model = RandomForestClassifier(n_estimators=100)
classification_model(model, traindf,predictor_var,outcome_var)


# In[22]:


# Using on the test dataset 
# Use all the features of the nucleus
predictor_var = features_mean
model = RandomForestClassifier(n_estimators=100,min_samples_split=25, max_depth=7, max_features=2)
classification_model(model, testdf,predictor_var,outcome_var)


# In[ ]:





# In[ ]:




