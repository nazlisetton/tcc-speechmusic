# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 11:01:52 2017
@author: nazli
"""

'''
Create additional plots for documentation purposes:
- Pie Chart showing data distribution
'''

import pandas as pd
import json
import matplotlib.pyplot as plt

with open('../config.json') as json_data:
    config = json.load(json_data)
    
'''
Pie Chart
'''
dataset = pd.read_csv(config['DATASET'])
distribution = dataset.groupby('music').count()['file']

labels = ['Fala', 'Música']
patches, text, autotext = plt.pie(distribution, colors = ['lightcoral', 'lightskyblue'], 
                                  autopct='%1.1f%%')
plt.legend(patches, labels, loc="best", prop={'size': 16})
autotext[0].set_fontsize(16)
autotext[1].set_fontsize(16)
plt.axis('equal')
plt.tight_layout()
plt.savefig('../../plots/additional/music_speech_distr.jpg')
plt.show()

'''
Speech X Music - Distribution for RMS
'''
names = ['uspSP_2017_08_06__11_32_00.wav.energy.csv', 
         'evangelizar2017_08_05__18_51_00.wav.energy.csv',
         'alpha_2017_05_13__20_44_00.wav.energy.csv']
labels = ['Fala - USP SP', 'Fala - Evangelizar FM', 'Música - Alpha FM']

for i in range(0, 3):
    df = pd.read_csv('./files/' + names[i], 
                     skiprows=range(0, 4), header = 0).reset_index()
    df.columns = ['index', 'en']
    df['index'] = df['index']/100
    plt.figure(figsize=(6,4))
    plt.ylim(0, 0.45)
    plt.plot(df['index'], df['en'])
    plt.title(labels[i], fontsize = 16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylabel('RMS', fontsize=16)
    plt.xlabel('Tempo (s)', fontsize=16)
    plt.savefig('../../plots/additional/speechVSmusic_' + str(i))
    plt.show()
