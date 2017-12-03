# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 21:09:10 2017
@author: Nazli
"""

'''
CLASSIFICADORES:
   TESTA PERFORMANCE DE UM VOTADOR DOS CLASSIFICADORES:
       - Output do votador é o output da maioria dos 3 classificadores
       - Valida com k-fold
'''

import pandas as pd
import json
from sklearn import metrics
from sklearn.model_selection import KFold
import pickle
import numpy as np

with open('../config.json') as json_data:
    config = json.load(json_data)

def predict(X, y_test, test_index, scaler, clf):
     X_test = X[test_index]        
     X_test_scaled = scaler.transform(X_test)
     predicted = clf.predict(X_test_scaled)
     f1 = metrics.f1_score(y_test, predicted)
     acc = metrics.accuracy_score(y_test, predicted)
     print('F1: ' + str(f1))
     print('ACC: ' + str(acc))
     print('')
     return predicted
        
def calculate_voter_results(soma, y_test):    
    for i, s in enumerate(soma):
        if soma[i] > 1:
            soma[i] = 1

    f1 = metrics.f1_score(y_test, soma)
    acc = metrics.accuracy_score(y_test, soma)
        
    print('VOTER')
    print('F1: ' + str(f1))
    print('ACC: ' + str(acc))
    print('')
    return f1, acc

def main():
    clf_svm = pickle.load(open('./models/svm.sav', 'rb'))
    clf_nn = pickle.load(open('./models/nn.sav', 'rb'))
    clf_knn = pickle.load(open('./models/knn.sav', 'rb'))
    
    scaler_svm = pickle.load(open('./models/scaler_svm.sav', 'rb'))
    scaler_nn = pickle.load(open('./models/scaler_nn.sav', 'rb'))
    scaler_knn = pickle.load(open('./models/scaler_knn.sav', 'rb'))
    
    kf = KFold(n_splits=2, shuffle=True)
    
    dataset = pd.read_csv(config['DATASET'])
    dataset = dataset.drop(['file', 'frame'], axis=1)
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    df_values = dataset.values
    y = df_values[:,0]
    
    X_svm = dataset[svm_features].values
    X_knn = dataset[knn_features].values
    X_nn = dataset[nn_features].values
    
    f1_scores = []
    acc_scores = []
    for train_index, test_index in kf.split(X_svm):
        y_test = y[test_index]

        print('SVM')
        predicted_1 = predict(X_svm, y_test, test_index, scaler_svm, clf_svm)
        print('KNN')
        predicted_2 = predict(X_knn, y_test, test_index, scaler_knn, clf_knn)
        print('NN')
        predicted_3 = predict(X_nn, y_test, test_index, scaler_nn, clf_nn)
        
        soma = predicted_1 + predicted_2 + predicted_3
        f1, acc = calculate_voter_results(soma, y_test)
        f1_scores.append(f1)
        acc_scores.append(acc)
        

    print('FINAL RESULTS FOR VOTER:')
    print(np.mean(f1_scores))
    print(np.mean(acc_scores))       

svm_features = ['low_energy_proportion', 'mfcc_2_var', 'mfcc_3_var', 
                'autocorrelation', 'spectralShapeStatistics_3_mean', 
                'perceptualSharpness_0_mean', 'mfcc_0_var', 'mfcc_4_mean', 
                'spectralShapeStatistics_0_mean', 'spectralFlux_0_mean', 
                'mfcc_1_mean']
    
nn_features = ['low_energy_proportion', 'mfcc_2_var', 'autocorrelation', 
               'spectralShapeStatistics_3_mean', 'perceptualSharpness_0_mean', 
               'mfcc_3_var', 'spectralShapeStatistics_2_var', 'energy_0_mean', 
               'lpc_0_mean', 'rms', 'spectralShapeStatistics_2_mean', 
               'perceptualSharpness_0_var','spectralShapeStatistics_3_var', 
               'spectralShapeStatistics_1_mean', 'spectralShapeStatistics_1_var', 
               'lpc_0_var']
    
knn_features = ['low_energy_proportion', 'mfcc_2_var', 'autocorrelation', 
                'spectralShapeStatistics_3_mean', 'perceptualSharpness_0_mean', 
                'mfcc_3_var','mfcc_0_var', 'energy_0_var', 'lpc_0_mean', 
                'mfcc_0_mean', 'energy_0_mean', 'mfcc_2_mean', 'mfcc_1_mean', 
                'mfcc_6_var', 'spectralShapeStatistics_2_mean']

main()
