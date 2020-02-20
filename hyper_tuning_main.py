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
import matplotlib.pyplot as plt

def main():
    training_file = args.data
    training_df = pd.read_csv(training_file)
    features = training_df.columns
    X = training_df.loc[:, training_df.columns != 'hospital_death']
    y = training_df['hospital_death']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    
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
            
    if args.model == 'Random Forest':
        pipe = Pipeline([('classifier' , RandomForestClassifier())])
        
    if args.model == 'XGBoost':
        pipe = Pipeline([('classifier' , XGBClassifier())])
        
    if args.model == 'Logistic Regression':
        pipe = Pipeline([('classifier' , LogisticRegression())])
    
    # Create param grid.
    
    param_grid = [
        {'classifier' : [LogisticRegression()],
         'classifier__penalty' : ['l1', 'l2'],
        'classifier__C' : np.logspace(-4, 4, 20), #Inverse of regularization strength
        'classifier__solver' : ['liblinear']},
        {'classifier' : [RandomForestClassifier()],
        'classifier__n_estimators' : list(range(10,101,20)),
        'classifier__criterion': ['entropy', 'gini'],
        'classifier__max_features' : list(range(10,len(features),30))},
        {'classifier' : [XGBClassifier()],
        'classifier__objective' : ['reg:logistic'],
        'classifier__learning_rate': list(np.arange(0.,0.1,0.05)),
        #'classifier__max_features' : list(range(6,len(features),5)),
        'classifier__n_estimators' : list(range(10,101,20))}
    ]
    
    # Create grid search object
    
    clf = GridSearchCV(pipe, param_grid = param_grid, cv = 3, verbose=True, n_jobs=4)
    
    # Fit on data
            
    best_clf = clf.fit(X_train, y_train)
    y_pred = best_clf.predict(X_test)
    print("AUC Score", metrics.roc_auc_score(y_test.values, clf.predict_proba(X_test)[:,1]))
    metrics.plot_confusion_matrix(clf, X_test, y_test)
    #optimised_random_forest = clf.best_estimator_
    joblib_file = os.path.join('XGBoost',"XG.pkl")
    joblib.dump(best_clf, joblib_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default = 'training.csv', help = 'Preprocessed data')
    parser.add_argument('--pca', type=str, default = True, help = 'Dimensionality Reduction')
    parser.add_argument('--model', type=str, default = 'Random Forest', help = 'Model Choice') # Random Forest, XGBoost, Logistic Regression
    main(parser.parse_args())