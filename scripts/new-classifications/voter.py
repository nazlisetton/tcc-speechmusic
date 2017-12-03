# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 14:55:22 2017
@author: nazli
"""

'''
NOVAS CLASSIFICAÇÕES:
        CALCULA PERFORMANCE DE UM MODELO ESCOLHIDO EM NOVOS DADOS COM VOTADOR:
            - mesmo dados de check-model-with-music.py
            - performance é calculada para cada um dos modelos e depois para 
            o votador
'''

import pickle
import json
import pandas as pd
import numpy as np
from basic import get_files, get_1sec_frames, autocorr, remove_outliers, check_results

with open('../config.json') as json_data:
    config = json.load(json_data)

##############################################################################
knn_features = ['low_energy_proportion', 'mfcc_2_var', 'autocorrelation', 
                'spectralShapeStatistics_3_mean', 'perceptualSharpness_0_mean', 
                'mfcc_3_var', 'mfcc_0_var', 'energy_0_var', 'lpc_0_mean', 
                'mfcc_0_mean', 'energy_0_mean', 'mfcc_2_mean', 'mfcc_1_mean', 
                'mfcc_6_var', 'spectralShapeStatistics_2_mean']

nn_features = ['low_energy_proportion', 'mfcc_2_var', 'autocorrelation', 
               'spectralShapeStatistics_3_mean', 'perceptualSharpness_0_mean',  
               'mfcc_3_var', 'spectralShapeStatistics_2_var', 'energy_0_mean', 
               'lpc_0_mean', 'rms', 'spectralShapeStatistics_2_mean',  
               'perceptualSharpness_0_var', 'spectralShapeStatistics_3_var', 
               'spectralShapeStatistics_1_mean', 'spectralShapeStatistics_1_var', 
               'lpc_0_var']

svm_features = ['low_energy_proportion', 'mfcc_2_var', 'mfcc_3_var', 
                'autocorrelation', 'spectralShapeStatistics_3_mean', 
                'perceptualSharpness_0_mean', 'mfcc_0_var', 'mfcc_4_mean', 
                'spectralShapeStatistics_0_mean', 'spectralFlux_0_mean', 
                'mfcc_1_mean']

base_features = ['.energy.csv', '.mfcc.csv', '.spectralFlux.csv', 
                 '.spectralShapeStatistics.csv', '.perceptualSharpness.csv', 
                 '.lpc.csv']
##############################################################################

def predict(data, algorithm, features):
    data = data[features]
    
    scaler_file = '../classifiers/models/scaler_{}.sav'.format(algorithm)
    scaler = pickle.load(open(scaler_file, 'rb'))
    clf_file = '../classifiers/models/{}.sav'.format(algorithm)
    clf = pickle.load(open(clf_file, 'rb'))

    scaled_data = scaler.transform(data)
    predicted = clf.predict(scaled_data)
    
    new_predicted = remove_outliers(predicted)
    df = pd.DataFrame(new_predicted)
    df.columns = ['predicted']
    
    return df
    
def check_results_soma(df, pcts, file):
    ones = np.count_nonzero(df['predicted'][1:] >= 2)
    pct = ones/len(df['predicted'][1:])
    pcts[file] = pct
    print(pct)

def get_results(pcts):
    total = 0
    count = 0
    for key, value in pcts.items():
        total += value
        count += 1
    print(total/count)
    
def main():
    pcts_svm = {}
    pcts_nn = {}
    pcts_knn = {}
    pcts_voter = {}

    #get CSV with files names
    music_files = pd.read_csv('./music.csv')
    soma = pd.DataFrame(columns=['soma'])

    knn_set = set(knn_features)
    nn_set = set(nn_features)
    svm_set = set(svm_features)

    all_features = knn_features + list(nn_set - knn_set) + list(svm_set - knn_set)

    for file in music_files['file']:
        print(file)
        files = []
        for feature in base_features:
            files.append('../../music_csv/' + file + feature)
            
        data_f = get_files(files)
        data = get_1sec_frames(data_f, all_features)
        data['autocorrelation'] = autocorr(data['autocorrelation'], 'full')
    
        #data = data.dropna()
    
        print("SVM")
        df_1 = predict(data, 'svm', svm_features)
        check_results(df_1, pcts_svm, file)
    
        print("KNN")
        df_2 = predict(data, 'knn', knn_features)
        check_results(df_2, pcts_knn, file)
    
        print("NN")
        df_3 = predict(data, 'nn', nn_features)
        check_results(df_3, pcts_nn, file)
    
        soma = df_1 + df_2 + df_3
        print("VOTER")
        check_results_soma(soma, pcts_voter, file)
        
        print("\n")
        
    return pcts_svm, pcts_knn, pcts_nn, pcts_voter
   
svm, knn, nn, voter = main()
print('\n Final results \n')
print('SVM')
get_results(svm)
print('KNN')
get_results(knn)
print('NN')
get_results(nn)
print('VOTER')
get_results(voter)
