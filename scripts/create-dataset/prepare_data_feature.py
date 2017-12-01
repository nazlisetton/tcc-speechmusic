# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 17:56:30 2017
@author: Nazli
"""

'''
PRÉ-PROCESSAMENTO - Parte 1:
    ESCOPO: Criação do dataset

    Agrupa dados de áudio em segmentos de 1 segundo, calculando média e variância.
    INPUT: arquivos ".csv" gerados pelo YAAFE para cada gravação
    OUTPUT: arquivo ".csv" com dados unificados (UM ARQUIVO POR FEATURE)
'''

import pandas as pd
import statistics as st
import json
import numpy as np

with open('../config.json') as json_data:
    config = json.load(json_data)

def main():
    '''
    Process each file of a feature and join them in one final dataframe
    '''
        
    for feature in config['FEATURES']:
        print("Starting feature: {}".format(feature))
        
        files = pd.read_csv(config['AUDIO_DATA_PATH'])
        final_data = pd.DataFrame()
        
        for index, row in files.iterrows():
            print(index)
            df = process_file(row['file'], row['music'], feature)
            
            if(len(df) != 0):
                if(len(final_data.index) > 0):
                    final_data = pd.concat([final_data, df])
                else:
                    final_data = df
        
        final_data = final_data.dropna(how = 'all').reset_index(drop=True)
        
        if('index' in final_data.columns):
            final_data = final_data.drop('index', 1)
        
        final_data.to_csv(config['CSV_OUT'] + feature + ".csv", index=False)
    
        print("Finished feature: {}".format(feature))
            
def process_file(file, music, feature):
    #open file, add columns names and make sure all columns are numeric
    df = get_file(file, feature)
    
    if(len(df) not in range(1500, 6000)):
        return pd.DataFrame()
    
    processed_df = pd.DataFrame()

    for column in df.columns:
        #column = df.columns[i]
        new_df = get_1sec_frames(pd.Series.tolist(df[column]), column)
        processed_df[column + "_mean"] = new_df["mean"]
        processed_df[column + "_var"] = new_df["var"]
        
        if column == 'energy_0':
            processed_df["low_energy_proportion"] = new_df["low_energy_proportion"]
            processed_df["rms"] = new_df["rms"]
            processed_df["autocorrelation_1"] = new_df["autocorrelation_1"]
            processed_df["autocorrelation_2"] = new_df["autocorrelation_2"]
    
    #Create column with file name and music x speech classification
    file = [file] * len(processed_df.index);
    music = [music] * len(processed_df.index);
    processed_df.insert(0, 'frame', list(range(0,len(processed_df.index))))
    processed_df.insert(0, 'file', file)
    processed_df.insert(0, 'music', music)
    
    return processed_df    
    
def get_file(file, feature):
    '''
    Get an audio sample of a given feature
    '''
    path = config['CSV_IN'].format(file, feature)

    #Skip Yaafe initial rows
    df = pd.read_csv(path, skiprows=range(0, 4), header = 0).reset_index()

    if('index' in df.columns and feature not in ['mfcc', 'lpc', 'spectralShapeStatistics']):
        df = df.drop('index', 1)

    df = df.dropna(0)

    #Add columns names
    columns = []
    for i in range(0, len(df.columns)):
        columns.append('{}_{}'.format(feature, i))
        
    df.columns = columns
    
    #Make sure all values are numeric
    for i in range(0, len(df.columns)):
        df[columns[i]] = df[columns[i]].apply(pd.to_numeric)
    
    return df

def get_1sec_frames(col, column_name):
    '''
    Calculate 1 second frames:
    Group 2 * config['FRAME_SIZE'] / config['SEG_SIZE'] points (currently 100 points)
    by calculating the group mean and variance.
    ''' 
    filtered_df_index = round(len(col) / (2 * config['FRAME_SIZE'] / config['SEG_SIZE']))  
    columns = ["mean", "var"]
    
    if column_name == 'energy_0':
        columns += ['low_energy_proportion', 'rms']
        
    filtered_df = pd.DataFrame(0, index=range(0, filtered_df_index), columns=columns)
    
    point = 0
    frame = []
    
    for value in col:
        frame.append(value)
        
        if len(frame) == (2 * config["FRAME_SIZE"] / config["SEG_SIZE"]):
            mean = st.mean(frame)
            variance = st.variance(frame)
            
            if column_name == 'energy_0':
                rms = variance / (mean * mean)
                proportion = check_energy(frame, mean)
                values = [mean, variance, proportion, rms]
            else:
                values = [mean, variance]
                
            filtered_df.loc[point] = values
            
            frame = []
            point += 1
            frame.append(value)
    
    if column_name == 'energy_0':
        a = autocorr(filtered_df['rms'], 'full')
        filtered_df['autocorrelation_1'] = a
        a = autocorr(filtered_df['rms'], 'same')
        filtered_df['autocorrelation_2'] = a
        
    return filtered_df

def check_energy(frame, mean):
    '''
    Calculate low energy proportion.
    For each group of 100 points (frame):
        - compare each point with the group's mean
        - if energy[point] < 0.5 * mean:count this point as a "low energy" point
        - else: do not count it
    Then, calculate the % of low energy points in the group.
    '''
    count = 0
    
    for i in range(0, len(frame)):
        if(frame[i] < 0.5 * mean):
            count += 1
            
    return count / len(frame)

def autocorr(x, mode):
    '''
    Calculate autocorrelation of a given array x
    Returns an array of same size
    '''
    result = np.correlate(x, x, mode=mode)
    
    if mode == 'same':
        return result
    return result[int(len(result)/2):]
