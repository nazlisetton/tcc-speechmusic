# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 22:16:59 2017
@author: nazli
"""

'''
NOVAS CLASSIFICAÇÕES:
    CLASSIFICA UM NOVO ARQUIVO EM MÚSICA OU FALA:
        - é o que será feito na API.
'''

import pickle
import json
from os import listdir
from os.path import isfile, join
from basic import get_files, get_1sec_frames, autocorr, remove_outliers

######################################################################
#Change algorithm here. Options: knn, nn, svm, knne, nne, svme
algorithm = 'svm'

#Add features for the chosen model
features =  ['low_energy_proportion', 'mfcc_2_var', 'mfcc_3_var', 
             'autocorrelation', 'spectralShapeStatistics_3_mean', 
             'perceptualSharpness_0_mean', 'mfcc_0_var', 'mfcc_4_mean', 
             'spectralShapeStatistics_0_mean', 'spectralFlux_0_mean', 
             'mfcc_1_mean']
######################################################################

with open('../config.json') as json_data:
    config = json.load(json_data)
    
def main(algorithm, features):
    path = './input/'
    files = ['./input/' + f for f in listdir(path) if isfile(join(path, f))]

    data_f = get_files(files)
    data = get_1sec_frames(data_f, features)

    data['autocorrelation'] = autocorr(data['autocorrelation'], 'full')

    #get scaler and classifier
    scaler_file = '../classifiers/models/scaler_{}.sav'.format(algorithm)
    scaler = pickle.load(open(scaler_file, 'rb'))
    clf_file = '../classifiers/models/{}.sav'.format(algorithm)
    clf = pickle.load(open(clf_file, 'rb'))

    scaled_data = scaler.transform(data)

    predicted = clf.predict(scaled_data)
    new_predicted = remove_outliers(predicted)
    return new_predicted

predicted = main(algorithm, features)
