# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 11:46:29 2017
@author: Nazli
"""

#Import Library
import pandas as pd
from sklearn import svm
import json
from sklearn.model_selection import KFold
import numpy as np
from basic import cv_scores

'''
SVM:
    ANALISA PARÊMETROS PARA KERNEL RBF:
'''

with open('../config.json') as json_data:
    config = json.load(json_data)
   
def svc_param_selection(X, y, kf):
    '''
        Test all combinations of Cs and gammas chosen and 
        add results to a dictionary.
    '''
    Cs = [1, 3, 5, 7]
    gammas = [0.1, 0.3, 0.5, 0.7]
    
    results = []
    dic_2 = {}
    
    clf = svm.SVC(kernel='linear')
    f1, p, r, a, m = cv_scores(X, y, clf, kf, '')
    dic = {}
    dic['kernel'] = 'linear'
    dic['f1'] = f1
    dic['acc'] = a
    dic['f1_mean'] = np.mean(f1)
    dic['acc_mean'] = np.mean(a)
    results.append(dic)
    
    dic_2['linear'] = np.mean(f1)
    print("Linear: {} \n".format(dic['f1_mean']))
    
    for c in Cs:
        for gamma in gammas:
            clf = svm.SVC(kernel='rbf', C=c, gamma=gamma)
            f1, p, r, a, m = cv_scores(X, y, clf, kf, '')
            dic = {}
            dic['c'] = c
            dic['gamma'] = gamma
            dic['f1'] = f1
            dic['acc'] = a
            dic['f1_mean'] = np.mean(f1)
            print("C = {}, gamma = {}: {} \n".format(c, gamma, dic['f1_mean']))
            dic['acc_mean'] = np.mean(a)
            results.append(dic)
            dic_2[str(c) + '_' + str(gamma)] = np.mean(f1)
            
    return results, dic_2

def rank_params(results):
    '''
        Rank parameters by f-measure
    '''
    return [(k, results[k]) for k in sorted(results, key=results.get, reverse=True)]

dataset = pd.read_csv(config['DATASET'])
dataset = dataset.drop(['file', 'frame'], axis=1)
dataset = dataset.sample(frac=1).reset_index(drop=True)
kf = KFold(n_splits=2, shuffle=True)

df_values = dataset.values
y = df_values[:,0]
X = dataset[['low_energy_proportion', 'mfcc_2_var', 'mfcc_3_var', 
              'autocorrelation', 'spectralShapeStatistics_3_mean', 
              'perceptualSharpness_0_mean', 'mfcc_0_var', 'mfcc_4_mean', 
              'spectralShapeStatistics_0_mean', 'spectralFlux_0_mean', 
              'mfcc_1_mean']]

X = X.values
results, dic_2 = svc_param_selection(X, y, kf)
sorted_params = rank_params(dic_2)

with open('./results_svm/params_results_2.json', 'w') as file:
    config = json.dump(dic_2, file)
