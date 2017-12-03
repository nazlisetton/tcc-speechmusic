# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 12:13:24 2017
@author: Nazli
"""

'''
SVM:
    SELECIONA AS FEATURES QUE SERÃO USADAS NO MODELO
    A PARTIR DE UM RANKING DE PERFORMANCE DAS FEATURES
'''


#Import Library
import pandas as pd
from sklearn import svm
import json
from sklearn.model_selection import KFold
from basic import test_features_by_ranking, validate

with open('../config.json') as json_data:
    config = json.load(json_data)

def main():
    dataset = pd.read_csv(config['DATASET'])
    dataset = dataset.drop(['file', 'frame'], axis=1)
    
    df_values = dataset.values
    y = df_values[:,0]
    
    dataset = dataset.drop(['music'], axis=1)

    clf = svm.SVC(kernel='linear')
    kf = KFold(n_splits=2, shuffle=True)

    ''' 
    Let's try to find the best set of features for a model.
    We use test_features_by_ranking and valide methods for that.
    
    We start by testing each feature and calculating the 
    performace metrics (f1, accuracy, etc).
    The best feature is then added to FEATURES global variable (in basic.py) and 
    all unused features will be tested with the one that has already been chosen.
    It goes on until all features have been chosen.
    It is basically a ranking algorithm.
    ''' 
    for i in range(0, len(dataset.columns)):
       results = test_features_by_ranking(dataset, y, clf, kf, 'svm')
       validate(results)

main()
