# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_markers: '{{{,}}}'
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# {{{ [markdown] button=false new_sheet=false run_control={"read_only": false}
# <a href="https://www.bigdatauniversity.com"><img src = "https://ibm.box.com/shared/static/cw2c7r3o20w9zn8gkecaeyjhgw3xdgbj.png" width = 400, align = "center"></a>
#
# <h1 align=center><font size = 5> Classification with Python</font></h1>
# }}}

# {{{ [markdown] button=false new_sheet=false run_control={"read_only": false}
# In this notebook we try to practice all the classification algorithms that we learned in this course.
#
# We load a dataset using Pandas library, and apply the following algorithms, and find the best one for this specific dataset by accuracy evaluation methods.
#
# Lets first load required libraries:
# }}}

# {{{ button=false new_sheet=false run_control={"read_only": false} jupyter={"outputs_hidden": true}
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
# %matplotlib inline
# }}}

# {{{ [markdown] button=false new_sheet=false run_control={"read_only": false}
# ### About dataset
# }}}

# {{{ [markdown] button=false new_sheet=false run_control={"read_only": false}
# This dataset is about past loans. The __Loan_train.csv__ data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:
#
# | Field          | Description                                                                           |
# |----------------|---------------------------------------------------------------------------------------|
# | Loan_status    | Whether a loan is paid off on in collection                                           |
# | Principal      | Basic principal loan amount at the                                                    |
# | Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
# | Effective_date | When the loan got originated and took effects                                         |
# | Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
# | Age            | Age of applicant                                                                      |
# | Education      | Education of applicant                                                                |
# | Gender         | The gender of applicant                                                               |
# }}}

# {{{ [markdown] button=false new_sheet=false run_control={"read_only": false}
# Lets download the dataset
# }}}

# {{{ button=false new_sheet=false run_control={"read_only": false} jupyter={"outputs_hidden": true}
# !wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv
# }}}

# {{{ [markdown] button=false new_sheet=false run_control={"read_only": false}
# ### Load Data From CSV File  
# }}}

# {{{ button=false new_sheet=false run_control={"read_only": false} jupyter={"outputs_hidden": true}
df = pd.read_csv('loan_train.csv')
df.head()
# }}}

df.shape

# {{{ [markdown] button=false new_sheet=false run_control={"read_only": false}
# ### Convert to date time object 
# }}}

# {{{ button=false new_sheet=false run_control={"read_only": false} jupyter={"outputs_hidden": true}
df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()
# }}}

# {{{ [markdown] button=false new_sheet=false run_control={"read_only": false}
# # Data visualization and pre-processing
#
#
# }}}

# {{{ [markdown] button=false new_sheet=false run_control={"read_only": false}
# Let’s see how many of each class is in our data set 
# }}}

# {{{ button=false new_sheet=false run_control={"read_only": false} jupyter={"outputs_hidden": true}
df['loan_status'].value_counts()
# }}}

# {{{ [markdown] button=false new_sheet=false run_control={"read_only": false}
# 260 people have paid off the loan on time while 86 have gone into collection 
#
# }}}

# Lets plot some columns to underestand data better:

# notice: installing seaborn might takes a few minutes
# !conda install -c anaconda seaborn -y

# {{{
import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()
# }}}

# {{{ button=false new_sheet=false run_control={"read_only": false} jupyter={"outputs_hidden": true}
bins=np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()
# }}}

# {{{ [markdown] button=false new_sheet=false run_control={"read_only": false}
# # Pre-processing:  Feature selection/extraction
# }}}

# {{{ [markdown] button=false new_sheet=false run_control={"read_only": false}
# ### Lets look at the day of the week people get the loan 
# }}}

# {{{ button=false new_sheet=false run_control={"read_only": false} jupyter={"outputs_hidden": true}
df['dayofweek'] = df['effective_date'].dt.dayofweek
bins=np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

# }}}

# {{{ [markdown] button=false new_sheet=false run_control={"read_only": false}
# We see that people who get the loan at the end of the week dont pay it off, so lets use Feature binarization to set a threshold values less then day 4 
# }}}

# {{{ button=false new_sheet=false run_control={"read_only": false} jupyter={"outputs_hidden": true}
df['weekend']= df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()
# }}}

# {{{ [markdown] button=false new_sheet=false run_control={"read_only": false}
# ## Convert Categorical features to numerical values
# }}}

# {{{ [markdown] button=false new_sheet=false run_control={"read_only": false}
# Lets look at gender:
# }}}

# {{{ button=false new_sheet=false run_control={"read_only": false} jupyter={"outputs_hidden": true}
df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)
# }}}

# {{{ [markdown] button=false new_sheet=false run_control={"read_only": false}
# 86 % of female pay there loans while only 73 % of males pay there loan
#
# }}}

# {{{ [markdown] button=false new_sheet=false run_control={"read_only": false}
# Lets convert male to 0 and female to 1:
#
# }}}

# {{{ button=false new_sheet=false run_control={"read_only": false} jupyter={"outputs_hidden": true}
df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()
# }}}

# {{{ [markdown] button=false new_sheet=false run_control={"read_only": false}
# ## One Hot Encoding  
# #### How about education?
# }}}

# {{{ button=false new_sheet=false run_control={"read_only": false} jupyter={"outputs_hidden": true}
df.groupby(['education'])['loan_status'].value_counts(normalize=True)
# }}}

# {{{ [markdown] button=false new_sheet=false run_control={"read_only": false}
# #### Feature befor One Hot Encoding
# }}}

# {{{ button=false new_sheet=false run_control={"read_only": false} jupyter={"outputs_hidden": true}
df[['Principal','terms','age','Gender','education']].head()
# }}}

# {{{ [markdown] button=false new_sheet=false run_control={"read_only": false}
# #### Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame 
# }}}

# {{{ button=false new_sheet=false run_control={"read_only": false} jupyter={"outputs_hidden": true}
Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()

# }}}

# {{{ [markdown] button=false new_sheet=false run_control={"read_only": false}
# ### Feature selection
# }}}

# {{{ [markdown] button=false new_sheet=false run_control={"read_only": false}
# Lets defind feature sets, X:
# }}}

# {{{ button=false new_sheet=false run_control={"read_only": false} jupyter={"outputs_hidden": true}
X = Feature
X[0:5]
# }}}

# {{{ [markdown] button=false new_sheet=false run_control={"read_only": false}
# What are our lables?
# }}}

# {{{ button=false new_sheet=false run_control={"read_only": false} jupyter={"outputs_hidden": true}
y = df['loan_status'].values
y[0:5]
# }}}

# {{{ [markdown] button=false new_sheet=false run_control={"read_only": false}
# ## Normalize Data 
# }}}

# {{{ [markdown] button=false new_sheet=false run_control={"read_only": false}
# Data Standardization give data zero mean and unit variance (technically should be done after train test split )
# }}}

# {{{ button=false new_sheet=false run_control={"read_only": false} jupyter={"outputs_hidden": true}
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]
# }}}

# {{{ [markdown] button=false new_sheet=false run_control={"read_only": false}
# # Classification 
# }}}

# {{{ [markdown] button=false new_sheet=false run_control={"read_only": false}
# Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of the model
# You should use the following algorithm:
# - K Nearest Neighbor(KNN)
# - Decision Tree
# - Support Vector Machine
# - Logistic Regression
#
#
#
# __ Notice:__ 
# - You can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a better model.
# - You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.
# - You should include the code of the algorithm in the following cells.
# }}}

# # K Nearest Neighbor(KNN)
# Notice: You should find the best k to build the model with the best accuracy.  
# **warning:** You should not use the __loan_test.csv__ for finding the best k, however, you can split your train_loan.csv into train and test to find the best __k__.

# We split the X into train and test to find the best k
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# Modeling
from sklearn.neighbors import KNeighborsClassifier
k = 3
#Train Model and Predict  
kNN_model = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
kNN_model

# just for sanity chaeck
yhat = kNN_model.predict(X_test)
yhat[0:5]

# Best k
Ks=15
mean_acc=np.zeros((Ks-1))
std_acc=np.zeros((Ks-1))
ConfustionMx=[];
for n in range(1,Ks):
    
    #Train Model and Predict  
    kNN_model = KNeighborsClassifier(n_neighbors=n).fit(X_train,y_train)
    yhat = kNN_model.predict(X_test)
    
    
    mean_acc[n-1]=np.mean(yhat==y_test);
    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
mean_acc

# Building the model again, using k=7
from sklearn.neighbors import KNeighborsClassifier
k = 7
#Train Model and Predict  
kNN_model = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
kNN_model

# # Decision Tree

from sklearn.tree import DecisionTreeClassifier
DT_model = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
DT_model.fit(X_train,y_train)
DT_model

yhat = DT_model.predict(X_test)
yhat

# # Support Vector Machine

from sklearn import svm
SVM_model = svm.SVC()
SVM_model.fit(X_train, y_train) 

yhat = SVM_model.predict(X_test)
yhat

# # Logistic Regression

from sklearn.linear_model import LogisticRegression
LR_model = LogisticRegression(C=0.01).fit(X_train,y_train)
LR_model

yhat = LR_model.predict(X_test)
yhat

# # Model Evaluation using Test set

from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

# First, download and load the test set:

# !wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv

# {{{ [markdown] button=false new_sheet=false run_control={"read_only": false}
# ### Load Test set for evaluation 
# }}}

# {{{ button=false new_sheet=false run_control={"read_only": false} jupyter={"outputs_hidden": true}
test_df = pd.read_csv('loan_test.csv')
test_df.head()
# }}}

## Preprocessing
test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
test_Feature = test_df[['Principal','terms','age','Gender','weekend']]
test_Feature = pd.concat([test_Feature,pd.get_dummies(test_df['education'])], axis=1)
test_Feature.drop(['Master or Above'], axis = 1,inplace=True)
test_X = preprocessing.StandardScaler().fit(test_Feature).transform(test_Feature)
test_X[0:5]

test_y = test_df['loan_status'].values
test_y[0:5]

knn_yhat = kNN_model.predict(test_X)
print("KNN Jaccard index: %.2f" % jaccard_similarity_score(test_y, knn_yhat))
print("KNN F1-score: %.2f" % f1_score(test_y, knn_yhat, average='weighted') )

DT_yhat = DT_model.predict(test_X)
print("DT Jaccard index: %.2f" % jaccard_similarity_score(test_y, DT_yhat))
print("DT F1-score: %.2f" % f1_score(test_y, DT_yhat, average='weighted') )

SVM_yhat = SVM_model.predict(test_X)
print("SVM Jaccard index: %.2f" % jaccard_similarity_score(test_y, SVM_yhat))
print("SVM F1-score: %.2f" % f1_score(test_y, SVM_yhat, average='weighted') )

LR_yhat = LR_model.predict(test_X)
LR_yhat_prob = LR_model.predict_proba(test_X)
print("LR Jaccard index: %.2f" % jaccard_similarity_score(test_y, LR_yhat))
print("LR F1-score: %.2f" % f1_score(test_y, LR_yhat, average='weighted') )
print("LR LogLoss: %.2f" % log_loss(test_y, LR_yhat_prob))

# # Report
# You should be able to report the accuracy of the built model using different evaluation metrics:

# | Algorithm          | Jaccard | F1-score | LogLoss |
# |--------------------|---------|----------|---------|
# | KNN                | 0.67    | 0.63     | NA      |
# | Decision Tree      | 0.72    | 0.74     | NA      |
# | SVM                | 0.80    | 0.76     | NA      |
# | LogisticRegression | 0.74    | 0.66     | 0.57    |

# {{{ [markdown] button=false new_sheet=false run_control={"read_only": false}
# ## Want to learn more?
#
# IBM SPSS Modeler is a comprehensive analytics platform that has many machine learning algorithms. It has been designed to bring predictive intelligence to decisions made by individuals, by groups, by systems – by your enterprise as a whole. A free trial is available through this course, available here: [SPSS Modeler](http://cocl.us/ML0101EN-SPSSModeler).
#
# Also, you can use Watson Studio to run these notebooks faster with bigger datasets. Watson Studio is IBM's leading cloud solution for data scientists, built by data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, Watson Studio enables data scientists to collaborate on their projects without having to install anything. Join the fast-growing community of Watson Studio users today with a free account at [Watson Studio](https://cocl.us/ML0101EN_DSX)
#
#
# <hr>
# Copyright &copy; 2018 [Cognitive Class](https://cocl.us/DX0108EN_CC). This notebook and its source code are released under the terms of the [MIT License](https://bigdatauniversity.com/mit-license/).​
# }}}

# {{{ [markdown] button=false new_sheet=false run_control={"read_only": false}
# ### Thanks for completing this lesson!
#
# Notebook created by: <a href = "https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a>
# }}}
