#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 10:42:52 2020

@author: carlosalcantara
"""

'''
Creates separate precision and recall 99% confidence interval plots for each machine
learning algorithm based on the feature subset utilized. 

Usage: ci_viz.py numTestSets path/to/report/dir/ path/to/save/figures/
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Check for command line argument
if len(sys.argv) < 3:
    print('Usage: ci_viz.py numTestSets path/to/report/dir/ path/to/save/figures/')
    sys.exit(-1)

numTestSets = int(sys.argv[1])
reportDirectory = sys.argv[2]
vizDirectory = sys.argv[3]

# if specified result path does not exist, it is created
if not os.path.exists(vizDirectory):
    os.makedirs(vizDirectory)

# calculate 99% confidence intervals for all ml algorithms using the test sets
ci = {}
for ml in ['knn','dt','rf']:
    temp = pd.read_csv(reportDirectory+'/precision_recall_'+ml+'_experiments.csv')
    for metric in ['precision','recall']:
        df = pd.DataFrame()
        
        for i in range(temp.shape[0]):
            for test in range(1,numTestSets+1):
                df = df.append( {'features':temp['features'][i],
                                  'type':metric,
                                  'test':test,
                                  'score':temp.iloc[i]['test'+str(test)+' '+metric]}, ignore_index=True)
        ci[ml] = pd.DataFrame()
        for x in df['features'].unique():
            tempdf = df[df['features']==x]
            mean = np.mean(tempdf['score'])
            error = 3.291 * ( np.std(tempdf['score']) / np.sqrt(numTestSets) )
            ci[ml] = ci[ml].append({'features':x, 'mean':mean, 'error':error}, ignore_index=True)

# find absolute max and min values for errorbars
max_lim = 0
min_lim = 100
max_er = 0
for i in ci.keys():
    if max_lim < ci[i]['mean'].max():
        max_lim = ci[i]['mean'].max()
    if min_lim > ci[i]['mean'].min():
        min_lim = ci[i]['mean'].min()
    if max_er < ci[i]['error'].max():
        max_er = ci[i]['error'].max()
max_lim = int((max_lim+max_er+.02)*100)/100
min_lim = int((min_lim-max_er-.01)*100)/100
        
# generate plots
for ml in ['knn','dt','rf']:
    for metric in ['precision','recall']:     
        plt.style.use('seaborn-whitegrid')
        # create errorbars
        ax = plt.errorbar(x=ci[ml]['features'], y=ci[ml]['mean'], yerr=ci[ml]['error'], capsize=5, fmt='.k')
        
        # plot labels
        plt.xlabel('FEATURES')
        plt.ylabel('SCORE')
        plt.ylim(min_lim, max_lim)
        if ml == 'knn':
            plt.title('KNN '+metric)
        elif ml == 'dt':
            plt.title('Decision Tree '+metric)
        else:
            plt.title('Random Forest '+metric)
        # rotate x-axis labels for better readability
        plt.xticks(rotation=90)

        # save plots noting if figure represents precision or recall
        if metric == 'recall':
            plt.savefig(vizDirectory+'/ci_'+ml.upper()+'_'+metric, bbox_inches = "tight")
        else:
            plt.savefig(vizDirectory+'/ci_'+ml.upper(), bbox_inches = "tight")
        plt.close()