# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:03:36 2017
@author: nazli
"""

'''
NOVAS CLASSIFICAÇÕES:
    CALCULA PERFORMANCE DE UM MODELO ESCOLHIDO EM NOVOS DADOS:
        - Esses dados não foram utilizados para treinar os modelos
        - São provenientes de rádios diferentes das rádios de treinamento
        - Só há músicas nos dados. Assim, sabe-se que toda predição deveria ser 1.
'''

import pickle
import json
import pandas as pd
from basic import get_files, get_1sec_frames, autocorr, remove_outliers, check_results

with open('../config.json') as json_data:
    config = json.load(json_data)

######################################################################################
#Change algorithm here. Options: knn, nn, svm, knne, nne, svme
algorithm = 'svm'

#change files that will be used here
base_features = ['.energy.csv', '.mfcc.csv', '.perceptualSharpness.csv',
                '.spectralShapeStatistics.csv', '.spectralFlux.csv']

#Add features for the chosen model
features =  ['low_energy_proportion', 'mfcc_2_var', 'mfcc_3_var', 
             'autocorrelation', 'spectralShapeStatistics_3_mean', 
             'perceptualSharpness_0_mean', 'mfcc_0_var', 'mfcc_4_mean', 
             'spectralShapeStatistics_0_mean', 'spectralFlux_0_mean', 
             'mfcc_1_mean']
    
######################################################################################    

def main():
    #get CSV with files names
    music_files = pd.read_csv('./music.csv')

    #get scaler and classifier
    scaler_file = '../classifiers/models/scaler_{}.sav'.format(algorithm)
    scaler = pickle.load(open(scaler_file, 'rb'))
    clf_file = '../classifiers/models/{}.sav'.format(algorithm)
    clf = pickle.load(open(clf_file, 'rb'))

    pcts = {}
    for file in music_files['file']:
        print(file)
        files = []
        for feature in base_features:
            files.append('../../music_csv/' + file + feature)
        data_f = get_files(files)
        data = get_1sec_frames(data_f, features)        
        data['autocorrelation'] = autocorr(data['autocorrelation'], 'full')
        scaled_data = scaler.transform(data)
        predicted = clf.predict(scaled_data)

        new_predicted = remove_outliers(predicted)
        df = pd.DataFrame(new_predicted)
        df.columns = ['predicted']
        
        #check results with what was expected (1)
        check_results(df, pcts, file)

    total = 0
    count = 0
    
    for key, value in pcts.items():
        total += value
        count += 1
    
    print('OVERALL')
    print(total/count)
    
data = main()
