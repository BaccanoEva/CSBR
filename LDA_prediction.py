import os
import re
import csv
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import random

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, metrics
from sklearn.cross_validation import train_test_split

import matplotlib.pylab as plt

#data process
groundtruth = []
csvfile = open('groundtruth.csv','r')
reader = csv.reader(csvfile)

for item in reader:
    groundtruth.append(len(item))
csvfile.close()

print "len(groundtruth):",len(groundtruth)

#{0: 144, 1: 3706, 2: 1634, 3: 664, 4: 333, 5: 198, 6: 82, 7: 46, 8: 42, 9: 36, 10: 91}
#len(groundtruth): 6976
for i in range(len(groundtruth)):
    if groundtruth[i] == 1:
        groundtruth[i] = 0
    else:
        groundtruth[i] = 1


X = []
y = []
predict_set = []
predict_data = []
predict_gt = []

f = open('/Users/liuyancen/Desktop/stanford_tmt/stanford_tmt_input/document-topic-distributions.csv',"r")

reader = csv.reader(f)

mashup_lda_vector = {}
for item in reader:
    if "mashup" in item[0]:
        t = item[0].split('_')
        k = int(t[1])

        vector = []
        for i in range(1,51):
            vector.append(item[i])

        #print "k = ",k
        if mashup_lda_vector.has_key(k):
            mashup_lda_vector[k].append(vector)
        else:
            mashup_lda_vector[k] = []
            mashup_lda_vector[k].append(vector)

print "len(mashup_lda_vector)", len(mashup_lda_vector)

#print mashup_service_lda_vector["service_20"]
f.close()

f1 = open('/Users/liuyancen/Desktop/nlp_lxtong/nlplab/Test_set_100','r')
line = f1.readline()
t = line.split('|')
for i in range(len(t)-1):
    predict_set.append(int(t[i]))

items = mashup_lda_vector.items()
for key,vector in items:
    if groundtruth[key] > 0 or groundtruth[key] == 0:
        v = 50*[0]
        for i in vector:
            for j in range(50):
                v[j] += float(i[j])
        for j in range(50):
            v[j] /= len(vector)

        #y.append(groundtruth[key])
        #X.append(v)

        if key in predict_set:
            predict_data.append(v)
            predict_gt.append(groundtruth[key])
        else:
            y.append(groundtruth[key])
            X.append(v)


# generate synthetic data from ESLII - Example 10.2
#X, y = make_hastie_10_2(n_samples=5000)
#print X.shape
#print y.shape
#print type(X)
#print type(y)
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# fit estimator
'''
param_test1 = {'n_estimators':range(10,101,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, max_depth=9, min_samples_split=1200,min_samples_leaf=60, subsample=0.85, random_state=10, max_features=7),param_grid = param_test1, scoring='roc_auc',iid=False,cv=5)
gsearch1.fit(X, y)

print gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
'''
#parameter log 1
#for param_test1 = {'n_estimators':range(100,801,100)}
#the best result is when n_estimators is 100
#so try param_test1 = {'n_estimators':range(10,101,10)}
#the best result is when n_estimators is 80
'''
param_test2 = {'max_depth':range(3,14,2),'min_samples_split':range(100,1001,200)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1,min_samples_leaf=60, subsample=0.85, random_state=10, max_features='sqrt'),param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
gsearch2.fit(X, y)

print gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
'''
#parameter log 2
#param_test2 = {'max_depth':range(3,14,2),'min_samples_split':range(100,1001,200)}
#the best result is when max_depth is 7 and min_samples_split is 700
#and the max_depth we can defined to be 7
'''
param_test3 = {'min_samples_leaf':range(10,101,10),'min_samples_split':range(500,1901,200)}
gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators = 80, max_depth = 7, subsample=0.85, random_state=10, max_features='sqrt'),param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
gsearch3.fit(X, y)

print gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
'''
#parameter log 3
#param_test3 = {'min_samples_leaf':range(10,101,10),'min_samples_split':range(500,1901,200)}
#the best result is when min_samples_leaf is 60 and min_samples_split is 700
'''
gbm1 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,max_depth=7, min_samples_leaf =60,
               min_samples_split =700, max_features='sqrt', subsample=0.8, random_state=10)
gbm1.fit(X,y)
y_pred = gbm1.predict(X)
y_predprob = gbm1.predict_proba(X)[:,1]
print "Accuracy : %.4g" % metrics.accuracy_score(y, y_pred)
print "AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob)
'''
#Accuracy : 0.7349
#AUC Score (Train): 0.813601
'''
param_test4 = {'max_features':range(7,20,2)}
gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,max_depth=7, min_samples_leaf =60,
               min_samples_split =700, subsample=0.8, random_state=10),
                       param_grid = param_test4, scoring='roc_auc',iid=False, cv=5)
gsearch4.fit(X,y)
print gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
'''
#parameter log 4
#param_test4 = {'max_features':range(7,20,2)}
#the best result is when max_features is 11
'''
param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,max_depth=7, min_samples_leaf =60,
               min_samples_split =700, max_features=11, random_state=10),
                       param_grid = param_test5, scoring='roc_auc',iid=False, cv=5)
gsearch5.fit(X,y)
print gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_
'''
#parameter log 5
#param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
#the best result is when subsample is 0.8
'''
gbm2 = GradientBoostingClassifier(learning_rate=0.05, n_estimators=80,max_depth=7, min_samples_leaf =60,
               min_samples_split =700, max_features=11, subsample=0.8, random_state=10)
gbm2.fit(X,y)
y_pred = gbm2.predict(X)
y_predprob = gbm2.predict_proba(X)[:,1]
print "Accuracy : %.4g" % metrics.accuracy_score(y, y_pred)
print "AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob)
'''
#Accuracy : 0.6912
#AUC Score (Train): 0.763008

est = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,max_depth=7, min_samples_leaf =60,
               min_samples_split =700, max_features='sqrt', subsample=0.8, random_state=10)
est.fit(X,y)

# score on test data (accuracy)
y_pred = est.predict(predict_data)
y_predprob = est.predict_proba(predict_data)[:,1]
print "Accuracy : %.4g" % metrics.accuracy_score(predict_gt, y_pred)
print "AUC Score (Train): %f" % metrics.roc_auc_score(predict_gt, y_predprob)


# predict class probabilities
#est.predict_proba(X_test)[0]
#ACC: 0.9240
#Out[4]:
#array([ 0.26442503,  0.73557497])
c = 0
for i in range(97):
    print predict_gt[i],y_pred[i]
    if predict_gt[i] == y_pred[i]:
        c+=1
print "c=",c
