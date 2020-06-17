#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 18:12:08 2020

@author: carlosalcantara
"""

'''
Generates a boxplot for each machine learning algorithm results aggregated by class
label for each feature subset. Uses ML.py output files and saves all figures to the
specified path.

Usage: python3 class_breakdown_boxplots.py path/to/ML.py/output/ path/to/save/figures/
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import glob

# Check for command line argument
if len(sys.argv) < 4:
    print('Usage: python3 class_breakdown_boxplots.py path/to/ML.py/output/ path/to/save/figures/')
    exit()

path = sys.argv[1]
results_path = sys.argv[2]

# read all available result files from ML.py script
mylist = [f for f in glob.glob(path+"*classification_report.txt")]
knn_list = [f for f in mylist if 'knn' in f]

# create feature list from file names
feature_list = [f.split('/')[-1].split('knn')[0] for f in knn_list]
feature_list = list(set(feature_list))

# load result information from ML.py text files and generate plots
for l in feature_list:
    for ml in ['knn','dt','rf']:
    	# load data
        temp_list = [f for f in glob.glob(path+l+ml+"*classification_report.txt")]
        print(ml,len(temp_list))
        df = pd.DataFrame()
        for e in temp_list:
            print(e)
            with open(e) as f:
                data = f.read()
                data = eval(data)
                for key, val in data.items():
                   if key != 'accuracy':
                        df = df.append( {'test':str(e.split('test')[1][0]),
                                         'class':key,
                                         'score':val['precision']}, ignore_index=True)
        # remove unnecessary metrics
        df = df[(df['class'] != 'weighted avg')&(df['class'] != 'micro avg')&(df['class'] != 'macro avg')]
        # create plot
        fig, ax = plt.subplots()
        ax = sns.boxplot(x='class',y='score',data=df, ax=ax,color='white')
        # plot labels
        ax.set_xlabel('CLASS')
        ax.set_ylabel('PRECISION')
        if ml == 'knn':
            ax.set_title('KNN')
        if ml == 'dt':
            ax.set_title('Decision Tree')
        if ml == 'rf':
            ax.set_title('Random Forest')
        # rotate x-axis labels for readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.tick_params(axis='both', which='major')
        # save plot as png
        print(ml+'_'+e.split('/')[-1].split(ml)[0]+'.pdf')
        fig.savefig(results_path+ml+'_'+e.split('/')[-1].split(ml)[0]+'.png',bbox_inches='tight')
        plt.close()
