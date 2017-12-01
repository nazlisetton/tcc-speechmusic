# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 15:19:05 2017
@author: Nazli
"""

'''
ANÁLISE EXPLORATÓRIA - PARTE 2
    CRIA GRÁFICOS DE CORRELAÇÃO ENTRE COLUNAS DO DATASET

    INPUT: ".csv" do dataset
    OUTPUT: gráfico bivariado de cada feature com as
            classes alvo (música ou fala) na pasta plots/target;
            gráfico bivariado feature_1 X feature_2 para um array de features 
            escolhidas (array "bivariate_features") na pasta plots/bivar.
'''

import pandas as pd
import matplotlib.pyplot as plt
import json

with open('../config.json') as json_data:
    config = json.load(json_data)
    
def plot_bivariate(df, features_list):
    for x in features_list:
        for y in features_list:
            if(x != y):
                print(x)
                print(y)
                data = df[[x, y]]
                
                data = data.sort_values(x)
                plt.scatter(data[x], data[y], s=0.1)
                
                plt.xlabel(x)
                plt.ylabel(y)
                plt.title(x + ' X ' + y) 
                
                plt.savefig("{}{}_{}.jpeg".format(config['PLOTS_PATH'] + 'bivar/', x, y))
                plt.show()

def plot_feature_target(df):
    columns = [col for col in df.columns if col not in ['file', 'music', 'frame']]
   
    for column in columns:
        df = df.sort_values(column)
        plt.scatter(df[column], df['music'], s=0.1)
        
        plt.xlabel(column)
        plt.ylabel('music')
        plt.title(column + ' X ' + 'music') 
        
        plt.savefig("{}{}.jpeg".format(config['PLOTS_PATH'] + 'target/', column))
        plt.show()

dataset = pd.read_csv(config['DATASET'])

#Features that will be passes as an argument to plot_bivariate
bivariate_features = ['rms', 'low_energy_proportion', 'zeroCrossing_0_var', 
                      'spectralShapeStatistics_2_mean', 'mfcc_0_var']

plot_feature_target(dataset)
plot_bivariate(dataset, bivariate_features)
