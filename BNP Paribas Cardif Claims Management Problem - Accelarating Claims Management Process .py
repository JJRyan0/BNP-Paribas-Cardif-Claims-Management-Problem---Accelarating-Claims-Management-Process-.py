# coding: utf-8

# # Introduction
# This tutorial shows how sklearn [Random Forst Model](https://en.wikipedia.org/wiki/Random_forest) ensemble classification model can be utilized to make a prediction. 
This tutorial covers usage of sklearn machine learning module from Python. Title: Accelerating Insurance Claim Process with Machine Learning using Random Forests

#Kaggle competition: BNP Paribas cardif claims management problem

#Created by John Ryan 22th April 2017

#Data source: https://www.kaggle.com/c/bnp-paribas-cardif-claims-management

# 

# ### Sklearn Python Module
# 
# Load the module & dependencies.

# In[ ]:

import os
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore", category=DeprecationWarning,
                       module="pandas, lineno=570")
from __future__ import print_function
import os
import pandas as pd
import numpy as np
import io
import requests
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
%matplotlib inline
from sklearn import preprocessing
from sklearn.metrics import log_loss, auc, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

#1.0 - Data Cleaning & Preperation

    #Read in the csv file.
    #Describe and analysis the data
    #Randomization of the data, separate out all variables to make sure they have the correct data type i.e. numeric, nominal and categorical.
    #Missing value treatment.
    #Encode labels in the data set with "one hot encoder".
    #Use cross Validation to create a test, training and validation data set.
    #Remove columns from the training data set and select important features.
    #Randomization of the data, separate out all variables to make sure they have the correct data type i.e. numeric, nominal and categorical.
    #Build machine Learning algorithms using pythons sci-kit learn from on the training, test data set constructed and preprocessed.
    #Evaluate the results. 10.Improve Performance.

#Load the data and make dataset a Pandas DataFrame
# In[ ]:
df = pd.read_csv("C:\\data\\claims.csv")
df.head()

#summerize the data to make some initle assessments
# In[ ]:
df.describe()

#View of target varable by labels 0 = not eligible for acceleration,  
#1 = claims suitable for an accelerated approval
# In[ ]:
d = sns.countplot(x ="target", data=df)
d.set(xticklabels=['not eligible for acceleration', 'claims suitable for an accelerated approval'])
plt.xlabel(' ')
d;

#Print out the number of missing values
# In[ ]:
print("Number of NA values : {0}".format((df.shape[0] * df.shape[1]) - df.count().sum()))

#Missing Value PVI using mean value of attributes;
# In[ ]:
x = df.fillna(df.mean())
x.head(7)
#check all values have been filled
# In[ ]:
print("Number of NA values : {0}".format((x.shape[0] * x.shape[1]) - x.count().sum()))

#Always good preactise to summerize the data after cleaning or transforming activities
# In[ ]:
x.describe()

#1.2 - Feature Importance

#Label encoder tranforms any label or attribute for input to the algorithim 
#we can also see some missing values in the top few rows of the data set these will also
#need to be treated in a suitable mannor.
# In[ ]:
for feature in x.columns:
    if x[feature].dtype=='object':
        le = LabelEncoder()
        df[feature] = le.fit_transform(x[feature])
x.tail(3)

#Assign the target variable to Y for later processing and 
#Remove the ID Column that is not needed
# In[ ]: 
y = x.target.values
x = x.drop(['ID'], axis = 1)


#Feature Importance - selecting only highly prdictive features using random forest Model
# In[ ]: 
from sklearn.ensemble import ExtraTreesClassifier
x.shape
# feature extraction
# In[ ]: 
model = ExtraTreesClassifier(n_estimators = 250, max_features = "auto", random_state=0)
model.fit(x, y)
print(model.feature_importances_)

#Ranking the most imporatnt predictive variables potentially build model based on top ranked i.e 1 -16
featureimportance = model.feature_importances_
# In[ ]: 
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(featureimportance)[::-1]
#Print top ranked predictive featurescdata = pd.DataFrame(x, columns = [ 'target','v50','v66','v47','v110','v31','v10','v113','v114'..])
print("Feature ranking:")

for feature in range(x.shape[1]):
    print("%d. feature %d (%f)" % (feature + 1, indices[feature], featureimportance[indices[feature]]))   
	
	
#Subsetting the data set using the top ranked variable produced by the algorithim
# In[ ]: 
cdata = pd.DataFrame(x, columns = [ 'target','v50','v66','v47','v110','v31','v10','v114'])
cdata.tail(4)

#Cross - Validation - split the data into 70% training and the remainder for testing the model
# In[ ]: 
from sklearn.cross_validation import train_test_split
# In[ ]: 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=0)

#Create the Random Forest Classification Model
# In[ ]:
np.random.seed(42)
from sklearn.ensemble import RandomForestClassifier
RFmodel = RandomForestClassifier(n_estimators=150, min_samples_split=50, max_depth=25, max_features='auto')
#Prediction on held for testing
RFpredict = RFmodel.fit(X_train, Y_train).predict(X_test)
RFpredict

#Evaluate Model Performance - Classification Accuracy
# In[ ]:
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
cmatrix = confusion_matrix(Y_test, RFpredict)
print (cmatrix)

#Classification Report
#The f1-score is equal to the weighted average of the precision and recall.
# In[ ]:
creport1 = classification_report(Y_test, RFpredict)
print (creport1)


