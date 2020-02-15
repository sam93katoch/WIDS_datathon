#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 02:35:58 2020

@author: sameeksha
"""
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn import metrics
from xgboost import XGBClassifier

def main(args):
    training_df = pd.read_csv(args.data)
    features = training_df.columns
    X = training_df.loc[:, training_df.columns != 'hospital_death']
    y = training_df['hospital_death']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    
    if args.model == 'Random Forest':
        clf=RandomForestClassifier(n_estimators=args.num_estimators, criterion=args.split_criteria, max_depth = None)
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        auc_score = metrics.roc_auc_score(y_test.values, clf.predict_proba(X_test)[:,1])
        print("AUC Score", auc_score)
        metrics.plot_confusion_matrix(clf, X_test, y_test)
        joblib_file = os.path.join('RF_model_',str(args.num_estimators),'_estimators',str(args.split_criteria),\
                                   '_criterion',".pkl")
        joblib.dump(clf, joblib_file)

    
    if args.model == 'xgboost':
        clf=XGBClassifier(objective ='reg:linear', learning_rate = args.xgboost_lr,
                max_depth = args.xgboost_max_depth, alpha = args.xgboost_alpha, n_estimators = args.num_estimators)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("AUC Score", metrics.roc_auc_score(y_test.values, clf.predict_proba(X_test)[:,1]))
        metrics.plot_confusion_matrix(clf, X_test, y_test)
        joblib_file = os.path.join('XG_model_',str(args.num_estimators),'_estimators',str(args.xgboost_lr),\
                                   '_lr',str(args.xgboost_max_depth),'_max_depth',str(args.xgboost_alpha),'_normalize',".pkl")
        joblib.dump(clf, joblib_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default = 'training.csv', help = 'Preprocessed data')
    parser.add_argument('--model', type=str, default = 'Random Forest', help = 'Model Choice') # Random Forest, XGBoost, Autoencoder
    parser.add_argument('--num_estimators', type=int, default = 1000, help = 'Num trees in Random Forest/XGBoost')
    parser.add_argument('--split_criteria', type=str, default = 'entropy', help = 'Measure the quality of a split.') # entropy, gini_index
    parser.add_argument('--xgboost_lr', type=float, default = 0.01, help = 'Step Size shrinkage to prevent overfitting')
    parser.add_argument('--xgboost_max_depth', type=int, default = 5, help = 'Tree Depth during each boosting round')
    parser.add_argument('--xgboost_alpha', type=float, default = 2, help = 'L1 regularization on leaf weights')
    main(parser.parse_args())