# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 21:09:10 2017
@author: Nazli
"""

'''
SVM:
    CRIA MODELO FINAL COM AS FEATURES JÁ ESCOLHIDAS POR select-features
'''

#Import Library
import pandas as pd
from sklearn import svm
import json
from sklearn.model_selection import KFold
from basic import cv_scores

with open('../config.json') as json_data:
    config = json.load(json_data)
   
def main():
    dataset = pd.read_csv(config['DATASET'])
    dataset = dataset.drop(['file', 'frame'], axis=1)
    
    clf = svm.SVC(kernel='rbf', C=5, gamma=0.3)
    kf = KFold(n_splits=2, shuffle=True)
    features = ['low_energy_proportion', 'mfcc_2_var', 'mfcc_3_var', 
                'autocorrelation', 'spectralShapeStatistics_3_mean', 
                'perceptualSharpness_0_mean', 'mfcc_0_var', 'mfcc_4_mean', 
                'spectralShapeStatistics_0_mean', 'spectralFlux_0_mean', 
                'mfcc_1_mean']
    
    df_values = dataset.values
    y = df_values[:,0]

    X = dataset[features]
    X = X.values
    cv_scores(X, y, clf, kf, 'svm')

main()
