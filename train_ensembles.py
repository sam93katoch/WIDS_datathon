#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 2020

@author: Deepta Rajan
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold

training_df = pd.read_csv('train_processed.csv')
features = training_df.columns
X = training_df.loc[:, training_df.columns != 'hospital_death'].values
y = training_df['hospital_death'].values
X_unlabeled = pd.read_csv('unlabeled_processed.csv').values
res = pd.read_csv('./resids.csv', names=['encounter_id', 'hospital_death'])
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

names = ['Bagging', 'Random Forest', 'AdaBoost', 'XGB']
classifiers = [
    BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=5), n_estimators=100, random_state=42, n_jobs=4),
    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=4),
    AdaBoostClassifier(n_estimators=100, base_estimator=DecisionTreeClassifier(max_depth=5), algorithm='SAMME', random_state=42),
    # GradientBoostingClassifier(n_estimators=100, random_state=42, n_jobs=4),
    XGBClassifier(n_estimators=100, objective='binary:logistic', learning_rate=0.1, booster='gbtree',max_depth=5, random_state=42, n_jobs=4)
]

fold = 1
for train_index, test_index in skf.split(X, y):
    print("K-Fold: ", fold)
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    for name,clf in zip(names,classifiers):
        print("Training classifier: ", name)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:,1]
        # Metrics
        print("Classification Report for %s classifier"%(name))
        print("Val AUC Score", metrics.roc_auc_score(y_test, y_prob))
        print(metrics.confusion_matrix(y_test, y_pred))
        # Save model
        joblib_file = os.path.join('ckpts',name+"_"+str(fold)+"_baseline.pkl")
        joblib.dump(clf, joblib_file)
        # Evaluate on unlabeled set
        unlab_pred = clf.predict(X_unlabeled)
        unlab_prob = clf.predict_proba(X_unlabeled)[:,1]
        res['hospital_death'] = unlab_prob
        res.to_csv('./res_'+name+"_"+str(fold)+'_baseline.csv')
    fold += 1


# logistic = LogisticRegression(solver='saga', tol=1e-2, max_iter=200, random_state=42)
# distributions = dict(C=uniform(loc=0, scale=4), penalty=['l2', 'l1'])
# clf = RandomizedSearchCV(logistic, distributions, random_state=42, cv = 3, verbose=True, n_jobs=4)

# # Fit on data
# clf = LogisticRegression(solver='saga', penalty='l2', tol=1e-2, max_iter=200, random_state=42)
# best_clf = clf.fit(X_train, y_train)
# y_pred = best_clf.predict(X_test)
# print("Val AUC Score", metrics.roc_auc_score(y_test.values, clf.predict_proba(X_test)[:,1]))
# print(metrics.confusion_matrix(y_test, y_pred))
# #optimised_random_forest = clf.best_estimator_
# joblib_file = os.path.join('ckpts',"LR_baseline.pkl")
# joblib.dump(best_clf, joblib_file)

# X_unlabeled = pd.read_csv('unlabeled_processed.csv').values
# unlab_pred = best_clf.predict(X_unlabeled)
# res = pd.read_csv('./resids.csv', names=['encounter_id', 'hospital_death'])
# res['hospital_death'] = unlab_pred
# res.to_csv('./res_LR_baseline.csv')
