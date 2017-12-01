# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 11:26:12 2017
@author: Nazli
"""

'''
PRÉ-PROCESSAMENTO - Parte 2:
    ESCOPO: Criação do dataset

    Junta dados de todas as features. Lê ".csv" de cada feature e monta um
    dataframe único. Salva um ".csv" consolidado. 
'''

import pandas as pd
import json
import prepare_data_feature as prep

with open('../config.json') as json_data:
    config = json.load(json_data)

'''
PRÉ-PROCESSAMENTO - Partes 1 e 2:
    ESCOPO: Criação do dataset
    
    Chama main de prepare_data_feature para gerar csv consolidados para cada feature
    Agrupa dados de todas as features, criando dataset final.
'''

def join_data(files):
    dataset = pd.DataFrame()
    
    for file in files:
        if(file not in ['dataset.csv']):
            path = config['CSV_OUT'] + file
            df = pd.read_csv(path)
            
            print(file)
            print(len(df))
            
            if(len(dataset) == 0):
                dataset = df
            else:
                df = df.drop('music', axis = 1)
                dataset = pd.merge(dataset, df, on = ['file', 'frame'], how = 'inner')

    dataset.to_csv(config['DATASET'], index=False)
    return dataset

files = []
for feature in config['FEATURES']:
    files.append(feature + ".csv")

prep.main()
dataset = join_data(files)
