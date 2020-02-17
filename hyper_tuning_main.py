#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 20:36:04 2020

@author: sameeksha
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


training_df = pd.read_csv('nan_empty_allzero.csv')
features = training_df.columns
X = training_df.loc[:, training_df.columns != 'hospital_death']
y = training_df['hospital_death']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
pca = False
# Dimensionality Reduction
if pca:
        sc = StandardScaler()
        X_train_std = sc.fit_transform(X_train)
        X_test_std = sc.transform(X_test)
        pca = PCA().fit(X_train_std)
        plt.figure()
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Variance (%)') #for each component
        plt.title('WIDS Dataset Explained Variance')
        plt.show()
        pca = PCA(n_components=5)
        X_train = pca.fit_transform(X_train_std)
        X_test = pca.transform(X_test_std)
        
        
pipe = Pipeline([('classifier' , RandomForestClassifier())])

# Create param grid.

param_grid = [
    {'classifier' : [LogisticRegression()],
     'classifier__penalty' : ['l1', 'l2'],
    'classifier__C' : np.logspace(-4, 4, 20), #Inverse of regularization strength
    'classifier__solver' : ['liblinear']},
    {'classifier' : [RandomForestClassifier()],
    'classifier__n_estimators' : list(range(10,101,10)),
    'classifier__criterion': ['entropy', 'gini'],
    'classifier__max_features' : list(range(10,len(features),20))},
    {'classifier' : [XGBClassifier()],
    'classifier__objective' : ['reg:logistic'],
    'classifier__learning_rate': list(np.arange(0.,0.1,0.05)),
    'classifier__max_features' : list(range(6,len(features),5)),
    'classifier__n_estimators' : list(range(10,101,10))}
]

# Create grid search object

clf = GridSearchCV(pipe, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)

# Fit on data

best_clf = clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)
print("AUC Score", metrics.roc_auc_score(y_test.values, clf.predict_proba(X_test)[:,1]))
metrics.plot_confusion_matrix(clf, X_test, y_test)