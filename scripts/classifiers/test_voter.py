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

#Import Library
import pandas as pd
import json
from sklearn import metrics
from sklearn.model_selection import KFold
import pickle
import numpy as np

with open('../config.json') as json_data:
    config = json.load(json_data)

def cv_scores(data, y, scaler, clf, kf):
    X = data.values
    predictions = []
    y_tests = []
    
    for train_index, test_index in kf.split(X):
        X_test = X[test_index]
        y_test = y[test_index]
        y_tests.append(y_test)
        
        X_test_scaled = scaler.transform(X_test)
        predicted = clf.predict(X_test_scaled)
        f1 = metrics.f1_score(y_test, predicted)
        predictions.append(predicted)
        
        acc = metrics.accuracy_score(y_test, predicted)
        print('F1: ' + str(f1))
        print('ACC: ' + str(acc))
        print('')
        
    return predictions, y_tests

def calculate_voter_results(svm_res, knn_res, nn_res, y_test):
    soma = svm_res + knn_res + nn_res
    
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
    dataset = pd.read_csv(config['DATASET'])
    dataset = dataset.drop(['file', 'frame'], axis=1)
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    
    clf_svm = pickle.load(open('./models/svm.sav', 'rb'))
    clf_nn = pickle.load(open('./models/nn.sav', 'rb'))
    clf_knn = pickle.load(open('./models/knn.sav', 'rb'))
    
    scaler_svm = pickle.load(open('./models/scaler_svm.sav', 'rb'))
    scaler_nn = pickle.load(open('./models/scaler_nn.sav', 'rb'))
    scaler_knn = pickle.load(open('./models/scaler_knn.sav', 'rb'))
    
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
    
    knn_features = ["low_energy_proportion", "mfcc_2_var", 'autocorrelation', 
                    'spectralShapeStatistics_3_mean', 'perceptualSharpness_0_mean', 
                    'mfcc_3_var','mfcc_0_var', 'energy_0_var', 'lpc_0_mean', 
                    'mfcc_0_mean', 'energy_0_mean', 'mfcc_2_mean', 'mfcc_1_mean', 
                    'mfcc_6_var', 'spectralShapeStatistics_2_mean']
    
    kf = KFold(n_splits=2, shuffle=True)

    df_values = dataset.values
    y = df_values[:,0]
    
    print('SVM')
    svm_pred, y_tests = cv_scores(dataset[svm_features], y, scaler_svm, clf_svm, kf)
    print('KNN')
    knn_pred, y_tests = cv_scores(dataset[knn_features], y, scaler_knn, clf_knn, kf)
    print('NN')
    nn_pred, y_tests = cv_scores(dataset[nn_features], y, scaler_nn, clf_nn, kf)
     
    #results for first iteration
    f1_0, acc_0 = calculate_voter_results(svm_pred[0], knn_pred[0], nn_pred[0], y_tests[0])
    #results for second iteration
    f1_1, acc_1 = calculate_voter_results(svm_pred[1], knn_pred[1], nn_pred[1], y_tests[1])

    print('FINAL RESULTS FOR VOTER:')
    print(np.mean([f1_0, f1_1]))
    print(np.mean([acc_0, acc_1]))       

main()
