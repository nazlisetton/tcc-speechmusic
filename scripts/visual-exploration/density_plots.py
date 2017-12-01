# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 19:45:25 2017
@author: Nazli
"""

'''
ANÁLISE EXPLORATÓRIA - PARTE 1
    FAZ GRÁFICOS DE DISTRIBUIÇÃO DE CADA FEATURE DO DATASET
    
    INPUT: ".csv" do dataset
    OUTPUT: um gráfico de distribuição (ou histograma) por feature na pasta plots/density.
'''

import json
import pandas as pd
import matplotlib.pyplot as plt

with open('../config.json') as json_data:
    config = json.load(json_data)

def main():
    df = pd.read_csv(config['DATASET'])
    
    for col in df.columns:
        if(col not in ['music', 'file', 'frame']):
            df_2 = df[['music', col]]
            plot(df_2, col)

def plot(df, name):
    music = df[(df['music'] == 1)]
    speech = df[(df['music'] == 0)]
    
    music[name].plot(kind='density')
    speech[name].plot(kind='density')
    
    #plt.hist(music[name], bins=50)
    #plt.hist(speech[name], bins=50)
    plt.legend(['Music', 'Speech'])
    
    plt.title(name) 
    plt.ylabel('Densidade')
    plt.xlabel('Valor')
    plt.savefig("{}{}.jpeg".format(config['PLOTS_PATH'] + 'density/', name))

    plt.show()

main()
