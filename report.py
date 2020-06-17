#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 14:12:13 2020

@author: carlosalcantara
"""

'''
Generates classification report of ML.py results. To be run once all feature subsets
have completed execution. This script will generate 3 csv files, one for each
machine learning algorithm. These reports are then utilized to generate plots in the
class_breakdown_boxplots.py, ci_viz.py and latex_table_generator.py scripts.

Usage: report.py #ofTestFiles path/to/ML.py/results/ path/to/save/resulting/reports/
'''

import os
import pandas as pd
from itertools import combinations
import sys

# Check for command line argument
if len(sys.argv) < 3:
    print('Usage: report.py #ofTestFiles path/to/ML.py/results/ path/to/save/reports/')
    sys.exit(-1)

noOfTests = sys.argv[1]
test = []
src = sys.argv[2]
report_path = sys.argv[3]


for filename in os.listdir(src):
    test.append('_'.join(filename.split('_')[:-3]))
test.sort()
test = list(set(test))
test.sort()
# if hidden file is read it should be excluded
test = test[1:] # comment this out if this is not the case

# create feature subset list to match file names from ML.py output
combos = []
features = ['duration, dPkts, dOctets_']
for combinationVal in range(1,9):
    newfeatures = ['dstaddrcount', 'srcportcount', 'dstportunique']
    for subset in combinations(newfeatures, combinationVal):
        combos.append(str(features)+'_'.join(subset))
combos1 = list(features) + combos
combos2 = [(str(s)).replace(', ','_').replace("['",'').replace("']",'').replace("'_'",'') for s in combos1]
combos2[0] = combos2[0][:-1]
count = 0
count2 = 0
cols = ['features']
for i in range(1,noOfTests+1):
    cols.append(f'test{i} precision')
    cols.append(f'test{i} recall')
# generate pandas dataframe for each machine learning algorithm
data_knn = pd.DataFrame(columns=cols)
data_dt = pd.DataFrame(columns=cols)
data_rf = pd.DataFrame(columns=cols)
# add features to all pandas dataframes
data_knn['features'] = combos2
data_knn.set_index('features', inplace = True)
data_dt['features'] = combos2
data_dt.set_index('features', inplace = True)
data_rf['features'] = combos2
data_rf.set_index('features', inplace = True)

# generate the ending of ML.py filenames for all test sets
report = []
for i in range(1,noOfTests+1):
    report.append(f'_test{i}_classification_report.txt')


# function to remove decimal places from results
def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

# read each file and load result info into appropriate dataframe
text = {}
for filename in test:
    for j in range(0,noOfTests):
        print(src+filename+report[j])
        with open(src+filename+report[j],'r') as f:
            text_string = f.read()
            text[j+1] = eval(text_string)
    if 'knn' in filename:
        index = filename.split('knn')[0]
        for i in range(1,noOfTests+1):
            data_knn.loc[index,f'test{i} precision'] = truncate(text[i]['weighted avg']['precision'],4)
            data_knn.loc[index,f'test{i} recall'] = truncate(text[i]['weighted avg']['recall'],4)
    elif 'dt' in filename:
        index = filename.split('dt')[0]
        for i in range(1,noOfTests+1):
            data_dt.loc[index,f'test{i} precision'] = truncate(text[i]['weighted avg']['precision'],4)
            data_dt.loc[index,f'test{i} recall'] = truncate(text[i]['weighted avg']['recall'],4)
    elif 'rf' in filename:
        index = filename.split('rf')[0]
        for i in range(1,noOfTests+1):
            data_rf.loc[index,f'test{i} precision'] = truncate(text[i]['weighted avg']['precision'],4)
            data_rf.loc[index,f'test{i} recall'] = truncate(text[i]['weighted avg']['recall'],4)

# change format of feature sets to make more readable for later use in subsequent scripts
z = ')'
combos3 = [str(str(s) + z).replace('duration_dPkts_dOctets_','+ (') for s in combos2]
combos3[0] = combos3[0][:-1]

# join src and dst with as, no longer needed
# combos4 = [s.replace('_',', ').replace('src, as','src_as').replace('dst, as',
#            'dst_as') for s in combos3]

# update feature subset with newly generated values
data_knn['features'] = combos4
data_knn.set_index('features', inplace = True)
data_dt['features'] = combos4
data_dt.set_index('features', inplace = True)
data_rf['features'] = combos4
data_rf.set_index('features', inplace = True)

# save csv files of reports in specified path
data_knn.to_csv(report_path+'precision_recall_knn_experiments.csv')
data_dt.to_csv(report_path+'precision_recall_dt_experiments.csv')
data_rf.to_csv(report_path+'precision_recall_rf_experiments.csv')
