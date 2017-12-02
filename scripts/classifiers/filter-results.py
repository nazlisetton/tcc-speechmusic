# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 15:57:11 2017
@author: Nazli
"""

'''
CLASSIFICADORES:
    CRIA MODELO FINAL COM AS FEATURES JÁ ESCOLHIDAS POR select-features:
'''

import json
import matplotlib.pyplot as plt
from basic import filter_file

#############################################################
#Number of the last results_file in folder './results_[algo]/
last_file = 38
#it may be: knn, svm, nn
algorithm = "svm" 
#############################################################

def main(last_file, algorithm):
    f1 = []
    acc = []
    x = []
    features_f1 = []
    features_acc = []

    for i in range(0, last_file):
        with open('./results_{}/results_{}.txt'.format(algorithm, i)) as json_data:
            dic = json.load(json_data)
        feat_f1, feat_acc, f1_result, acc_result = filter_file(dic, i)
        
        features_f1.append(feat_f1)
        features_acc.append(feat_acc)
        
        f1.append(f1_result)
        acc.append(acc_result)
        
        x.append(i)

    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(9,6))
    plt.plot(x, f1)
    plt.plot(x, acc)
    plt.title('F1 e acurácia em função do número de atributos escolhidos')
    plt.ylabel('Resultados')
    plt.xlabel('Número de atributos')
    plt.legend(['F1', 'Acurácia'])

    plt.savefig('./{}-feature-selection-results'.format(algorithm))
    
    return features_f1, features_acc

features_f1, features_acc = main(last_file, algorithm)
