# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 21:09:10 2017
@author: Felipe
"""

'''
KNN:
    CRIA MODELO FINAL COM AS FEATURES JÁ ESCOLHIDAS POR select-features
'''

import pandas as pd
from sklearn import neighbors
import json
from sklearn.model_selection import KFold
from basic import cv_scores

with open('../config.json') as json_data:
    config = json.load(json_data)
   
def main():
    dataset = pd.read_csv(config['DATASET'])
    dataset = dataset.drop(['file', 'frame'], axis=1)
    
    clf = neighbors.KNeighborsClassifier()
    kf = KFold(n_splits=2, shuffle=True)
    features = ['low_energy_proportion', 'mfcc_2_var', 'autocorrelation', 
                'spectralShapeStatistics_3_mean', 'perceptualSharpness_0_mean', 
                'mfcc_3_var', 'mfcc_0_var', 'energy_0_var', 'lpc_0_mean', 
                'mfcc_0_mean'	, 'energy_0_mean', 'mfcc_2_mean', 'mfcc_1_mean', 
                'mfcc_6_var', 'spectralShapeStatistics_2_mean']

    
    df_values = dataset.values
    y = df_values[:,0]

    X = dataset[features]
    X = X.values
    cv_scores(X, y, clf, kf, 'knn')

main()
