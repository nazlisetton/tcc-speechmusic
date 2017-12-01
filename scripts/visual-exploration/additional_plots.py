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
