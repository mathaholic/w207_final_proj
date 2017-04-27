
# This tells matplotlib not to try opening a new window for each plot.
%matplotlib inline

# Allows for proper import of xgboost package
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-6.3.0-posix-seh-rt_v5-rev2\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

# General libraries.
import re
import numpy as np
import matplotlib.pyplot as plt
import nltk
import json
import pandas as pd
import os
import xgboost as xgb

# SK-learn libraries for learning.
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# SK-learn libraries for evaluation.
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report

# SK-learn libraries for feature extraction from text.
from sklearn.feature_extraction.text import *

from subprocess import check_output



# For when I'm working on my desktop
os.chdir("C:\\Users\\skarb\\Desktop\\Github\\w207_final_proj\\")


####################################################################
# A place to test things or do EDA
####################################################################
# setup the training and development data
df = pd.read_json("train.json", orient="columns")
train_data = np.array(pd.read_json("train.json", orient="columns"))
train_labels = np.where(train_data[:,22]==True,1,0)
text_data = train_data[:,6]
##############################################

# Playing with the DataFrame

# Let's verify our features' object types
df.dtypes

df.head()

df.columns

df['test'] = pd.to_datetime(df['unix_timestamp_of_request_utc'], unit='s')


test = df[['test','number_of_upvotes_of_request_at_retrieval','requester_received_pizza']]

test.head()

test[]


df[]
df.describe()

df.giver_username_if_known

plt.figure()

plt.scatter(sample.words, sample.requester_received_pizza)


len(df["request_text"][0].split())
df["words"]

sample = df[['words','requester_received_pizza']]




df["words"] = map(df)

df["words"] = [len(x.split()) for x in df["request_text"]]

test1 = "this is a string"
type(test1)
test

len(test1.split())



train_data.shape





# some posts have crazy high vote counts
np.sort(train_data[:,26])[:-11:-1]


for x,y in enumerate(train_data.columns):
    print(x,y)

for j in [1,5,6,60,600,659,3000]:
    for i in all_cont_num_vars:
        print(train_data[j,i])

np.unique(train_data[:,25])

train_data['bin_votes'] = bin_votes

np.max(train_data[:,26])/1000
   
train_data[0:10,28]

train_data[3]

####################################################################
####################################################################

###########################################
# Just numeric features
####################################

np.random.seed(1)

# indices of all numeric variables
all_cont_num_vars = [1,2,5,9,10,11,12,13,14,15,16,17,18,19,20,21,24,25,26,27]

# only "at time of request" numeric variables
cont_num_vars_at_request = [1,2,5,9,11,13,15,17,19,21,24,26]


# Preparing train/test data
num_data = train_data[:,all_cont_num_vars]
num_labels = train_labels



# randomizing the data
shuffle = np.random.permutation(np.arange(num_data.shape[0]))
num_train_data, num_train_labels = num_data[shuffle], num_labels[shuffle]

num_train_data = num_data[:3000,]
num_train_labels = num_labels[:3000]

num_test_data = num_data[3001:,]
num_test_labels = num_labels[3001:]



# simple logistic regression model
clf = LogisticRegression(C = 100)
clf.fit(num_train_data, num_train_labels)

preds = clf.predict(num_test_data)

print(classification_report(num_test_labels,preds))

print(metrics.accuracy_score(num_test_labels,preds))



# simple Random Forest model (default)
clf2 = RandomForestClassifier(n_estimators=12)
clf2.fit(num_train_data, num_train_labels)

preds2 = clf2.predict(num_test_data)

print(classification_report(num_test_labels,preds2))

print(metrics.accuracy_score(num_test_labels,preds2))


gbm = xgb.XGBClassifier()
gbm.fit(num_train_data, num_train_labels)
preds3 = gbm.predict(num_test_data)

print(classification_report(num_test_labels,preds3))

print(metrics.accuracy_score(num_test_labels,preds3))

##########################################################################
##########################################################################

##############################
# Working with the text data #
##############################


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



#  extract and setup test data
test_json_array = np.array(pd.read_json('test.json', orient='columns'))
test_data = test_json_array[:,2]

labels_whole_train = np.where(data_labels==True, 1, 0)
X_whole_train = tfv.fit_transform(data_set)
X_whole_test = tfv.transform(test_data)

# run logistic regression on ALL training data
C_vals = {'C': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0]}
model = LogisticRegression()
regression_model = GridSearchCV(model, C_vals, scoring="f1_macro")
regression_model.fit(X_whole_train, labels_whole_train)

# predict test data
pred_test = regression_model.predict(X_whole_test)  

##########################################################################
##########################################################################

####################################
# OUTPUTING RESULTS FOR SUBMISSION #
####################################
request_id = test_json_array[:,1]

output_submission = {"request_id": request_id , "requester_received_pizza":pred_test}

my_submission = pd.DataFrame(data=output_submission)
my_submission.to_csv('first_submission.csv', index = False)
