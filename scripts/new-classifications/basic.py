# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 11:08:49 2017

@author: nazli
"""

'''
NOVAS CLASSIFICAÇÕES:
    FUNÇOES COMUNS:
        - get_files
        - get_1sec_frames
        - get_sample_avg_var
        - remove_outliers
        - autocorr
        - check_results
'''

import pandas as pd
import statistics as st
import numpy as np

def get_files(files):
    '''
    Get files:
        - open features files, skipping YAAFE header
        - add columns names
        - concatenate all files
    '''
    data = pd.DataFrame()
    
    for file in files:
        df = pd.read_csv(file, skiprows=range(0, 4), header = 0).reset_index()
        feature = (file.split('.wav.')[1].split('.csv')[0])
    
        if('index' in df.columns and feature not in ['mfcc', 'lpc', 'spectralShapeStatistics']):
            df = df.drop('index', 1)
        
        #Add columns names
        columns = []

        for i in range(0, len(df.columns)):
            columns.append('{}_{}'.format(feature, i))
        df.columns = columns
            
        data = pd.concat([data, df], axis=1)

    #Make sure all values are numeric
    columns = data.columns
    for i in range(0, len(columns)):
        data[columns[i]] = data[columns[i]].apply(pd.to_numeric)

    data = data.dropna(0)
    return data

def get_1sec_frames(data, features):
    '''
    Get 1 second frames by calculating average and variance
    '''
    frame_size = 100
    rows = int(len(data) / frame_size)
    processed_df = pd.DataFrame(index=range(0, rows), columns = features)
    point = 0

    while point <= rows:
        frame = pd.DataFrame()
        frame = data.iloc[point * frame_size:(point + 1) * frame_size, :]
        
        if len(frame) < 10:
            return processed_df
        
        results = {}
        for column in frame.columns:
            get_sample_avg_var(frame[column], column, results)
        
        processed_df.loc[point] = list(map(lambda x: results[x] if x != 'autocorrelation' 
                                           else results['rms'], features))
        point += 1

    return processed_df

def get_sample_avg_var(col, name, results):  
    '''
    Calculate average and variance of a given column
    If this column is related to energy, calculate 
    low energy proportion and rms as well.
    '''
    mean = st.mean(col)
    var = st.variance(col)
    results[name + '_mean'] = mean
    results[name + '_var'] = var
    
    if(name == 'energy_0'):
        count = 0
        
        col = col.values
        for i in range(0, len(col)):
            if(col[i] < 0.5 * mean):
                count += 1
                
        results['rms'] = var / (mean * mean)
        results['low_energy_proportion'] = count / len(col)
    return results

def remove_outliers(prediction):
    '''
    Check each point (P') of the classification results: 
        - if neighbors (P and P'' ) have the same classification and it is different
          from the classification of P', change P' 
    '''
    for i in range(1, len(prediction) - 1):
        if(prediction[i - 1] != prediction[i] and prediction[i - 1] == prediction[i + 1]):
            prediction[i] = prediction[i - 1]
    return prediction

def autocorr(x, mode):
    result = np.correlate(x, x, mode=mode)
    return result[int(len(result)/2):]

def check_results(df, pcts, file):
    '''
    Check results for "check-model-with-music".
    The expected results are 1. 
    Compare real results with what was expected.
    '''
    ones = np.count_nonzero(df['predicted'][1:] == 1)
    pct = ones/len(df['predicted'][1:])
    pcts[file] = pct
    print(pct)