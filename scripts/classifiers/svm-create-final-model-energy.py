# -*- coding: utf-8 -*-

"""
Created on Sat Jul 15 21:09:10 2017
@author: Nazli
"""

#Import Library
import pandas as pd
from sklearn import svm
import json
from sklearn.model_selection import KFold
from basic import cv_scores

'''
SVM:
    CRIA MODELO FINAL COM AS FEATURES APENAS DE ENERGIA:
'''

with open('../config.json') as json_data:
    config = json.load(json_data)
   
def main():
    dataset = pd.read_csv(config['DATASET'])
    dataset = dataset.drop(['file', 'frame'], axis=1)
    
    df_values = dataset.values
    y = df_values[:,0]
        
    clf = svm.SVC(kernel='linear')
    kf = KFold(n_splits=2, shuffle=True)
    
    features = ['low_energy_proportion', 'autocorrelation', 'rms']
    
    X = dataset[features]
    X = X.values
    cv_scores(X, y, clf, kf, 'svme')
    return

main()
