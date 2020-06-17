#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 00:21:25 2019

@author: carlosalcantara
"""

'''
Conducts supervised machine learning algorithms (KNN, decision tree and random
forest) classifiers on csv data sets specified in the data directory using the
feature subset according to the options below, where 1-7 correspond to the addition
of the specified features to the base feature set 0.
		0: duration, dPkts, dOctets
        1: + dstaddrcount
        2: + srcportcount
        3: + dstportunique
        4: + dstaddrcount, srcportcount
        5: + dstaddrcount, dstportunique
        6: + srcportcount, dstportunique
        7: + dstaddrcount, srcportcount, dstportunique
Once the classification is conducted on the test sets, the results are saved to
txt files to the specified directory along with the trained model for each machine 
learning algorithm.

Usage: ML.py feature_subset_# #_of_test_files path/to/data/dir/ path/to/results/dir/
'''

import pandas as pd
from sklearn import tree
from sklearn import ensemble
from sklearn import neighbors
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.stats import randint as sp_randint
import sys
import json
from joblib import dump

# Check for command line argument
if len(sys.argv) < 4:
    print('Usage: python3 ML.py feature_subset_# #_of_test_files path/to/data/dir/ path/to/results/dir/')
    exit()

no_test_files = sys.argv[2]
data_path = sys.argv[3]
results_path = sys.argv[4]

# load train data set
trainfile = data_path+'train_scale.csv'
traindf = pd.read_csv(trainfile)

# load test data sets
test_dict = {}
for i in range(1,no_test_files+1):
    test_dict[i] = pd.read_csv(f'{data_path}test{i}_scale.csv')

combination_dict = {0: [['duration', 'dPkts', 'dOctets']],
                    1: [['duration', 'dPkts', 'dOctets', 'dstaddrcount']],
                    2: [['duration', 'dPkts', 'dOctets', 'srcportcount']],
                    3: [['duration', 'dPkts', 'dOctets', 'dstportunique']],
                    4: [['duration', 'dPkts', 'dOctets', 'dstaddrcount', 'srcportcount']],
                    5: [['duration', 'dPkts', 'dOctets', 'dstaddrcount', 'dstportunique']],
                    6: [['duration', 'dPkts', 'dOctets', 'srcportcount', 'dstportunique']],
                    7: [['duration', 'dPkts', 'dOctets', 'dstaddrcount', 'srcportcount', 'dstportunique']]}

combos  = combination_dict[int(sys.argv[1])]

labels = list(traindf['class'].unique())
labels.sort()

for features in combos:

    print('############################################################################################################')
    print('############################################################################################################')
    print('\tEXPERIMENT '+str(features))
    print('############################################################################################################')
    print('############################################################################################################')
    
    # Setup the training feature and target data set
    traindfTarget = traindf['class']
    traindfTrain = pd.DataFrame(traindf, columns=features)
    
    
    ############################################################################################################
    ################################################## KNN #####################################################
    ############################################################################################################
    
    
    print("KNN classifier")
    
    # Do a Grid Search for KNN's nearest-neighbor hyper-parameter
    param_grid = {"n_neighbors": sp_randint(1,128),
                  "weights": ["uniform", "distance"]}
    knn = neighbors.KNeighborsClassifier()
    knn_cv = RandomizedSearchCV(knn, param_grid, cv=5, n_jobs=-1, n_iter=16)
    print('KNN hyperparameters tuned')

    # Fit model
    knn_cv.fit(traindfTrain, traindfTarget)
    print('A KNN classifier with n_neighbors = ' + str(knn_cv.best_params_) + ' performed best (' + str(knn_cv.best_score_) + ')')
    # Save trained model for potential future use
    dump(knn_cv, str(features).replace(' ','_').replace('[','').replace(']','').replace(',','').replace("'",'')+'_knn.joblib')    

    # Save best hyper-parameters for model for documentation
    with open(results_path+str(features).replace(' ','_').replace('[','').replace(']','').replace(',','').replace("'",'')+'knn_test_best_params.txt','w') as f:
        f.write(str(knn_cv.best_params_))

    # Evaluate fitted model using test sets
    for testno in range(1,no_test_files+1):
        testdf = test_dict[testno]
        testdfTarget = testdf['class']
        testdfTest = testdf[features]
        
        # Predict test set
        y_pred = knn_cv.predict(testdfTest)
        knn_score = accuracy_score(testdfTarget, y_pred)
        print('test '+str(testno)+' accuracy:\t\t'+str(knn_score))
        
        # Save prediction results for further analysis
        report = classification_report(testdfTarget, y_pred,output_dict=True)
        report_json = json.dumps(report)
        matrix = confusion_matrix(testdfTarget, y_pred,labels=labels)
        with open(results_path+str(features).replace(' ','_').replace('[','').replace(']','').replace(',','').replace("'",'')+'knn_test'+str(testno)+'_classification_report.txt','w') as f:
            f.write(report_json)
        with open(results_path+str(features).replace(' ','_').replace('[','').replace(']','').replace(',','').replace("'",'')+'knn_test'+str(testno)+'_confusion_matrix.txt','w') as f:
            f.write(str(matrix))
        print('KNN test '+str(testno), report)
        
    
    ############################################################################################################
    ############################################ DECISION TREE #################################################
    ############################################################################################################
    
    
    print("Decision Tree classifier")
    
    # Do a Grid Search over Decision Tree's hyper-parameters
    param_dist = {"max_depth": [x for x in range(1,20)] + [None],
                              "max_features": [1, 2, 3],
                              "criterion": ["gini", "entropy"],
                              "min_samples_leaf": [x for x in range(1,20)]}
    dtree = tree.DecisionTreeClassifier(random_state=42)
    dtree_cv = RandomizedSearchCV(dtree, param_dist, cv=5, n_jobs=-1, n_iter=16)
    print('DT hyperparameters tuned')
    
    # Fit model
    dtree_cv.fit(traindfTrain, traindfTarget)
    print('A Decision Tree classifier with params = ' + str(dtree_cv.best_params_) + ' performed best (' + str(dtree_cv.best_score_) + ')')
    # Save trained model for potential future use
    dump(dtree_cv, str(features).replace(' ','_').replace('[','').replace(']','').replace(',','').replace("'",'')+'_dt.joblib')
    
    # Save best hyper-parameters for model for documentation
    with open(results_path+str(features).replace(' ','_').replace('[','').replace(']','').replace(',','').replace("'",'')+'dt_test_best_params.txt','w') as f:
        f.write(str(dtree_cv.best_params_))

    # Evaluate fitted model using test sets
    for testno in range(1,no_test_files+1):
        testdf = test_dict[testno]
        testdfTarget = testdf['class']
        testdfTest = testdf[features]
        
        # Predict test set
        y_pred = dtree_cv.predict(testdfTest)
        dtree_score = accuracy_score(testdfTarget, y_pred) 
        print('test accuracy:\t\t'+str(dtree_score))
        
        # Save prediction results for further analysis
        report = classification_report(testdfTarget, y_pred,output_dict=True)
        report_json = json.dumps(report)
        matrix = confusion_matrix(testdfTarget, y_pred,labels=labels)
        with open(results_path+str(features).replace(' ','_').replace('[','').replace(']','').replace(',','').replace("'",'')+'dt_test'+str(testno)+'_classification_report.txt','w') as f:
            f.write(report_json)
        with open(results_path+str(features).replace(' ','_').replace('[','').replace(']','').replace(',','').replace("'",'')+'dt_test'+str(testno)+'_confusion_matrix.txt','w') as f:
            f.write(str(matrix))
        print('DT test '+str(testno), report)
    
    ############################################################################################################
    ############################################ RANDOM FOREST #################################################
    ############################################################################################################
    
    
    print("Random Forest classifier")
    
    # Do a Grid Search over Random Forest's hyper-parameters
    param_dist = {"n_estimators": sp_randint(1,64),
                              "max_features": [1, 2, 3],
                              "max_depth": [x for x in range(1,20)] + [None],
                              "criterion": ["gini", "entropy"],
                              "min_samples_leaf": [x for x in range(1,20)]}
    rforest = ensemble.RandomForestClassifier(random_state=42)
    rforest_cv = RandomizedSearchCV(rforest, param_dist, cv=5, n_jobs=-1, n_iter=16)
    print('RF hyperparameters tuned')
    
    # Fit model
    rforest_cv.fit(traindfTrain, traindfTarget)
    print('A Random Forest classifier with params = ' + str(rforest_cv.best_params_) + ' performed best (' + str(rforest_cv.best_score_) + ')')
    # Save trained model for potential future use
    dump(rforest_cv, str(features).replace(' ','_').replace('[','').replace(']','').replace(',','').replace("'",'')+'_rf.joblib')

    # Save best hyper-parameters for model for documentation
    with open(results_path+str(features).replace(' ','_').replace('[','').replace(']','').replace(',','').replace("'",'')+'rf_test_best_params.txt','w') as f:
        f.write(str(rforest_cv.best_params_))

    # Evaluate fitted model using test sets
    for testno in range(1,no_test_files+1):
        testdf = test_dict[testno]
        testdfTarget = testdf['class']
        testdfTest = testdf[features]
        
        # Predict test set
        y_pred = rforest_cv.predict(testdfTest)
        rforest_score = accuracy_score(testdfTarget, y_pred)
        print('test accuracy:\t\t'+str(rforest_score))
        
        # Save prediction results for further analysis
        report = classification_report(testdfTarget, y_pred,output_dict=True)
        report_json = json.dumps(report)
        matrix = confusion_matrix(testdfTarget, y_pred,labels=labels)
        with open(results_path+str(features).replace(' ','_').replace('[','').replace(']','').replace(',','').replace("'",'')+'rf_test'+str(testno)+'_classification_report.txt','w') as f:
            f.write(report_json)
        with open(results_path+str(features).replace(' ','_').replace('[','').replace(']','').replace(',','').replace("'",'')+'rf_test'+str(testno)+'_confusion_matrix.txt','w') as f:
            f.write(str(matrix))
        print('test '+str(testno), report)
