# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 15:46:38 2017
@author: Nazli
"""

'''
ANÁLISE EXPLORATÓRIA - Parte 3:
    CRIA MATRIZES DE CORRELAÇÃO

    INPUT: arquivo ".csv" do dataset
    OUTPUT: matriz de correlação do dataset todo; 
            matriz de correlação de cada FEATURE com a classe alvo.
'''

# Correction Matrix Plot
import matplotlib.pyplot as plt
import pandas as pd
import numpy
import json

with open('../config.json') as json_data:
    config = json.load(json_data)
    
def plot_corr_music(corr):
    music_corr = corr[['music']]
    half = round(len(music_corr) / 2)
    
    c_1 = music_corr.iloc[0:half, :]
    c_2 = music_corr.iloc[half:len(music_corr), :]
    cols = corr.columns
    
    fig = plt.figure(figsize=(10,12))
    my_cmap = plt.cm.get_cmap('RdBu')
    
    ax = fig.add_subplot(111)
    cax = ax.imshow(c_2, my_cmap, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = numpy.arange(0, len(c_2), 1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(cols[half:len(cols)])
    ax.set_xticks([])
    
    
    ax = fig.add_subplot(121)
    cax = ax.imshow(c_1, my_cmap, vmin=-1, vmax=1)
    ticks = numpy.arange(0, len(c_1), 1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(cols[0:half])
    ax.set_xticks([])
    
    plt.savefig(config['PLOTS_PATH'] + 'correlation/corr_music.jpeg')
    plt.show()

def plot_corr_matrix(df, corr):
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    my_cmap = plt.cm.get_cmap('RdBu')
    cax = ax.imshow(corr, my_cmap, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = numpy.arange(0, len(corr.columns), 2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    
    for i in range(0, len(df.columns)):
        print("{} - {}".format(i, df.columns[i]))
        
    plt.savefig(config['PLOTS_PATH'] + 'correlation/corr_matrix.jpeg')
    plt.show()     
    
def calc_correlation(df):
    correlation = df.corr()
    
    #drop weak correlations
    correlation[(correlation>0) & (correlation<0.4)] = 0
    correlation[(correlation<0) & (correlation>-0.4)] = 0
    
    return correlation


dataset = pd.read_csv(config['DATASET'])
dataset = dataset.drop(['file', 'frame'], axis = 1)

corr = calc_correlation(dataset)

plt.rcParams.update({'font.size': 14})
plot_corr_music(corr)
plot_corr_matrix(dataset, corr)
