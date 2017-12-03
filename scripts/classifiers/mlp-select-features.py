# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 12:13:24 2017
@author: Rodrigo
"""

'''
MLP:
    SELECIONA AS FEATURES QUE SERÃO USADAS NO MODELO
    A PARTIR DE UM RANKING DE PERFORMANCE DAS FEATURES
'''

#Import Library
import pandas as pd
from sklearn import neural_network as nn
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

    kf = KFold(n_splits=2, shuffle=True)
    
    for i in range(0, len(dataset.columns)):
        '''
        We have to create the classifier again at each iteration
        to prevent a strange error. 
        May it be due to the "adaptive" learning rate?
        '''
        clf = nn.MLPClassifier(
            batch_size=132, 
            warm_start=True, 
            learning_rate='adaptive', 
            hidden_layer_sizes=(30,30,45))
        
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
        results = test_features_by_ranking(dataset, y, clf, kf, 'nn')
        validate(results)

g = main()
