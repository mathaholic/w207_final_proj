# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 08:48:41 2017

@author: skarb
"""

# This tells matplotlib not to try opening a new window for each plot.
%matplotlib inline

# General libraries.
import re
import numpy as np
import matplotlib.pyplot as plt
import nltk
import json
import pandas as pd
import os

# SK-learn libraries for learning.
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV

# SK-learn libraries for evaluation.
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report

# SK-learn libraries for feature extraction from text.
from sklearn.feature_extraction.text import *

from subprocess import check_output


os.chdir("C:\\Users\\skarb\\Desktop\\W207_Final\\")


# setup the training and development data
train_json_array = np.array(pd.read_json('train.json', orient='columns'))

data_set = train_json_array[:,6]
data_labels = train_json_array[:,22]

train_data = data_set[:data_set.shape[0]/2]
train_outcomes = data_labels[:data_labels.shape[0]/2]
train_labels = np.where(train_outcomes==True, 1, 0)

dev_data = data_set[(data_set.shape[0]/2)+1:]
dev_outcomes = data_labels[(data_labels.shape[0]/2)+1:]
dev_labels = np.where(dev_outcomes==True, 1, 0)

print('There are ',train_json_array.shape[0], ' observations and ', train_json_array.shape[1], ' features\n')


# shows how many people got pizza?
yescount = 0;
nocount = 0;
for i in range(0, train_json_array.shape[0]):
    if yescount > 3 and nocount > 3:
        break 
    if train_json_array[i][22] == False and nocount < 4:
        print(i, '. Outcome: ',train_json_array[i][22],'\n', train_json_array[i][6],'\n')
        nocount += 1
    elif train_json_array[i][22] == True and yescount < 4:
        print(i, '. ',train_json_array[i][22],'\n', train_json_array[i][6],'\n')
        yescount += 1
    else:
        continue


# Logistic regression on half of train data
tfv = TfidfVectorizer(analyzer='word', max_df=0.5, ngram_range=(1,3)) 

X_train = tfv.fit_transform(train_data)
X_dev = tfv.transform(dev_data)
print(X_train.shape, train_labels.shape)

clf = LogisticRegression(penalty='l2',C=100)
clf.fit(X_train, train_labels)
pred_labels = clf.predict(X_dev)
print(pred_labels.shape)
target_names = ['Got pizza', 'No pizza']
print(classification_report(dev_labels, pred_labels, target_names=target_names))
##################################

***********************
## David's additions **
***********************

# Step 1) extract and setup test data
test_json_array = np.array(pd.read_json('test.json', orient='columns'))
test_data = test_json_array[:,2]

labels_whole_train = np.where(data_labels==True, 1, 0)
X_whole_train = tfv.fit_transform(data_set)
X_whole_test = tfv.transform(test_data)

# Step 2) run logistic regression on ALL training data
C_vals = {'C': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0]}
model = LogisticRegression()
regression_model = GridSearchCV(model, C_vals, scoring="f1_macro")
regression_model.fit(X_whole_train, labels_whole_train)

# Step 3) predict test data
pred_test = regression_model.predict(X_whole_test)  

# Step 4) output results in the proper format

request_id = test_json_array[:,1]

output_submission = {"request_id": request_id , "requester_received_pizza":pred_test}

my_submission = pd.DataFrame(data=output_submission)
my_submission.to_csv('first_submission.csv', index = False)
