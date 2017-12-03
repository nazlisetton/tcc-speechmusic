# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 21:09:10 2017
@author: Nazli
"""

'''
CLASSIFICADORES:
    FUNÇOES COMUNS:
        - cv_scores
        - test_features_by_ranking
        - validate
        - filter_file
        - rank params
'''

import json
from sklearn import metrics, preprocessing
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix

with open('../config.json') as json_data:
    config = json.load(json_data)

#### Global variables ####
FEATURES = []
##########################

def cv_scores(X, y, clf, kf, clf_name):    
    '''
    dataset: dataset dataframe
    features: features chosen to be part of the model
    clf: classifier (from scikit-learn)
    kf: k-fold indices
    clf_name: classifier name to save the models; 
              if you do not want to save the models, pass an empty string ('')
    '''    
    f1 = []
    precision = []
    recall = []
    matthews = []
    accuracy = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = clf.fit(X_train_scaled, y_train)
        predicted = clf.predict(X_test_scaled)
        
        f1.append(metrics.f1_score(y_test, predicted))
        precision.append(metrics.precision_score(y_test, predicted))
        recall.append(metrics.recall_score(y_test, predicted))
        matthews.append(metrics.matthews_corrcoef(y_test, predicted))
        accuracy.append(metrics.accuracy_score(y_test, predicted))
        
        if clf_name != '':
            scaler_file = './models/scaler_{}.sav'.format(clf_name)
            pickle.dump(scaler, open(scaler_file, 'wb'))
            clf_file = './models/{}.sav'.format(clf_name)
            pickle.dump(clf, open(clf_file, 'wb'))
            cmat = confusion_matrix(y_test, predicted)
            print(cmat)
    
    if clf_name != '':
        '''
        Print results...
        '''
        print('\nF1')
        print(f1)
        print(np.mean(f1))
        print('\nACCURACY')
        print(accuracy)
        print(np.mean(accuracy))
        print('\nMATTHEWS')
        print(matthews)
        print(np.mean(matthews))
            
    return f1, precision, recall, accuracy, matthews

def test_features_by_ranking(dataset, y, clf, kf, clf_name):
    ''' 
    This method is responsible for one iteration of the ranking algorithm.
    It tests all unused features combined with the already chosen features, 
    one at a time.
    
    The output is a i.txt file with the results for the ith iteration.
    ''' 
    global_results = {}
    
    for feature in dataset.columns:        
        if(feature not in FEATURES):
            print(feature)
            test = FEATURES[:]
            test.append(feature)
            X = dataset[test]
            X = X.values
            
            f1, p, r, a, m = cv_scores(X, y, clf, kf, '')
            global_results[feature] = {}
            global_results[feature]['f1'] = f1
            global_results[feature]['precision'] = p
            global_results[feature]['recall'] = r
            global_results[feature]['accuracy'] = a
            global_results[feature]['matt'] = m
                      
    file_name = './results_{}/results_{}.txt'.format(clf_name, len(FEATURES))
            
    with open(file_name, 'w') as outfile:
        json.dump(global_results, outfile)
        
    return global_results

def validate(dic):
    ''' 
    This method analyzes the results from the previous iteration of test_features_by_ranking.
    It chooses the best feature (by checking f1 and accuracy values) and add it to FEATURES.
    ''' 
    results = {}
    results_acc = {}

    for i in dic.keys():
        mean = np.mean(dic[i]['f1'])
        results[i] = mean
        results_acc[i] = np.mean(dic[i]['accuracy'])
           
    sorted_f = [(k, results[k]) for k in sorted(results, key=results.get, reverse=True)]
    sorted_acc = [(k, results_acc[k]) for k in sorted(results_acc, key=results_acc.get, reverse=True)]

    first_f = next(iter(sorted_f))
    first_acc = next(iter(sorted_acc))
        
    print('RESULTS FOR ITERATION: ' + str(len(FEATURES)))
    print(first_f)
    print(first_acc)
    
    if(first_f[0] != first_acc[0]):
        print('Not equal')
    
    FEATURES.append(first_f[0])
    print(FEATURES)
    return

def filter_file(dic, file):
    ''' 
    This method gets a dictionary with model results for one iteration of
    feature selection and returns :
        - the best feature for this iteration (regarding f1 and accuracy)
        - f1 and accuracy values for the best feature
    '''
    results = {} 
    results_acc = {}
    
    for i in dic.keys():
        mean = np.mean(dic[i]['f1'])
        results[i] = mean
        results_acc[i] = np.mean(dic[i]['accuracy'])
               
    sorted_f = [(k, results[k]) for k in sorted(results, key=results.get, reverse=True)]
    sorted_acc = [(k, results_acc[k]) for k in sorted(results_acc, key=results_acc.get, reverse=True)]
    
    first_f = next(iter(sorted_f))
    first_acc = next(iter(sorted_acc))
        
    print('RESULTS FOR ITERATION: ' + str(file))
    print(first_f)
    print(first_acc)
    
    return first_f[0], first_acc[0], first_f[1], first_acc[1]

def rank_params(results):
    '''
    Rank parameters by f-measure
    Input: dictionary with models results for different choices of parameters
    Output: ordered list of parameters (best parameters appear first)
    '''
    return [(k, results[k]) for k in sorted(results, key=results.get, reverse=True)]
    