# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 21:09:10 2017
@author: Rodrigo
"""

#Import Library
import pandas as pd
import json
from sklearn.neural_network import MLPClassifier
from basic import cv_scores
from sklearn.model_selection import KFold

'''
MLP:
    CRIA MODELO FINAL COM AS FEATURES JÁ ESCOLHIDAS POR select-features
'''

with open('../config.json') as json_data:
    config = json.load(json_data)
   
def main():
    dataset = pd.read_csv(config['DATASET'])
    dataset = dataset.drop(['file', 'frame'], axis=1)
    
    clf = MLPClassifier(
            activation = 'tanh',
            batch_size=135,
            hidden_layer_sizes=(30,30,45))
    kf = KFold(n_splits=2, shuffle=True)
    
    features = ['low_energy_proportion', 'mfcc_2_var', 'autocorrelation', 
                'spectralShapeStatistics_3_mean', 'perceptualSharpness_0_mean', 
                'mfcc_3_var', 'spectralShapeStatistics_2_var', 'energy_0_mean', 
                'lpc_0_mean', 'rms', 'spectralShapeStatistics_2_mean', 
                'perceptualSharpness_0_var','spectralShapeStatistics_3_var', 
                'spectralShapeStatistics_1_mean', 'spectralShapeStatistics_1_var', 
                'lpc_0_var']
    
    df_values = dataset.values
    y = df_values[:,0]
    
    X = dataset[features]
    X = X.values
    cv_scores(X, y, clf, kf, 'nn')
    
main()
