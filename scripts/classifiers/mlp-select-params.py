# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 11:46:29 2017
@author: Rodrigo
"""

'''
MLP:
    ANALISA PARÊMETROS PARA O MODELO:
'''

import pandas as pd
import json
from sklearn.model_selection import KFold
import numpy as np
from sklearn.neural_network import MLPClassifier
from basic import cv_scores, rank_params

with open('../config.json') as json_data:
    config = json.load(json_data)

   
def mlp_param_selection(X, y, kf):
    batch_size = 135        #best size found
    activation = 'tanh'     #best activation function found
    neurons = [40, 60, 80]
    
    results = []
    dic_2 = {}
    count = 1
    
    for neuron1 in neurons:
        for neuron2 in neurons:
            for neuron3 in neurons:
                print('Iteration: ' + str(count))
                count += 1
                
                clf = MLPClassifier(
                        hidden_layer_sizes = (neuron1, neuron2, neuron3),
                        batch_size = batch_size,
                        activation = activation)
                
                f1, p, r, a, m = cv_scores(X, y, clf, kf, '')
                dic_2["{}_{}_{}".format(neuron1, neuron2, neuron3)] = np.mean(f1)
            
    return results, dic_2

dataset = pd.read_csv(config['DATASET'])
dataset = dataset.drop(['file', 'frame'], axis=1)
dataset = dataset.sample(frac=1).reset_index(drop=True)
kf = KFold(n_splits=2, shuffle=True)

df_values = dataset.values
y = df_values[:,0]
X = dataset[['low_energy_proportion', 'mfcc_2_var', 'autocorrelation', 
             'spectralShapeStatistics_3_mean', 'perceptualSharpness_0_mean', 
             'mfcc_3_var', 'spectralShapeStatistics_2_var', 'energy_0_mean', 
             'lpc_0_mean', 'rms', 'spectralShapeStatistics_2_mean', 
             'perceptualSharpness_0_var','spectralShapeStatistics_3_var', 
             'spectralShapeStatistics_1_mean',
             'spectralShapeStatistics_1_var', 'lpc_0_var']]

X = X.values
results, dic_2 = mlp_param_selection(X, y, kf)
sorted_params = rank_params(dic_2)

with open('./results_nn/params_results.json', 'w') as file:
    config = json.dump(dic_2, file)
