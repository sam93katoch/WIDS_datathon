#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 16:19:09 2020

@author: sameeksha
"""


import pandas as pd
import numpy as np


df = pd.read_csv('training_v2.csv', skipinitialspace=True)
features = df.columns

i = 2

def categorical_2_onehot(df):
    for i in df.columns:
        col = df[i]
        if type(col.values[0]) == str:
            df = pd.concat([df,pd.get_dummies(col, prefix=i)],axis=1)
            df.drop([i],axis=1, inplace=True)
    return df

#Make null values/empt values 0
def all_zero(df):
    categorical_2_onehot(df)
    df.fillna(0, inplace = True)
    df.to_csv('nan_empty_allzero.csv')

#Drop more missing values columns
def drop_missing(df):
    df = df[features]
    df.isna().sum()
    # Drop columns based on threshold limit
    threshold = len(df) * 0.60
    df_thresh=df.dropna(axis=1, thresh=threshold)
    #df_thresh=df_thresh.fillna(0)
    df_thresh =  categorical_2_onehot(df_thresh)
    df_thresh.to_csv('missing_values_thresholded.csv')

#Drop highly correlated columns
def drop_correlated(df):
    # Threshold for removing correlated variables
    threshold = 0.9
    # Absolute value correlation matrix
    corr_matrix = df.corr().abs()
    #corr_matrix.head()
    # Upper triangle of correlations
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    #upper.head()
    # Select columns with correlations above threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print('There are %d columns to remove.' % (len(to_drop)))
    #Drop the columns with high correlations
    df_thresh = df.drop(columns = to_drop)
    df_thresh.to_csv('removed_correlated_features.csv')
    
drop_correlated(df)
#dimensionality reduction
#df.to_csv('out.csv')