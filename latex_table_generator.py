#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:46:45 2020

@author: carlosalcantara
"""

'''
Creates LaTeX table code for each machine learning algorithm. Output text files saved
to specified location.

Usage: python3 latex_table_generator.py #ofTests path/to/reports/ path/to/save/latex/files/
'''

import pandas as pd
import numpy as np
from scipy.stats import t
import sys

if len(sys.argv) < 3:
    print('Usage: python3 latex_table_generator.py #ofTests path/to/reports/ path/to/save/latex/files/')
	exit()

noOfTests = int(sys.argv[1])
report_path = sys.argv[2]
tables_path = sys.argv[3]

# read report files for each ml algorithm
df_dict = {'knn' : pd.read_csv(report_path+'precision_recall_knn_experiments.csv'),
           'dt' : pd.read_csv(report_path+'precision_recall_dt_experiments.csv'),
           'rf' : pd.read_csv(report_path+'precision_recall_rf_experiments.csv')}

# generate LaTeX code and save in text file
for ml in ['knn','dt','rf']:
    precision = df_dict[ml][[f'test{x} precision' for x in range(1,noOfTests+1)]]
    recall = df_dict[ml][[f'test{x} recall' for x in range(1,noOfTests+1)]]
    
    # calculate mean and 99.9% error interval for precision and recall
    df_dict[ml]['precision mean'] = precision.mean(axis = 1)
    df_dict[ml]['precision error'] = t.ppf(.999, noOfTests-1) * ( precision.std(axis = 1) / np.sqrt(noOfTests))
    df_dict[ml]['recall mean'] = recall.mean(axis = 1)
    df_dict[ml]['recall error'] = t.ppf(.999, noOfTests-1) * ( recall.std(axis = 1) / np.sqrt(noOfTests))
    
    # LaTeX code generation
    head = '\\begin{table}[htpb]\n\centering\n\\resizebox{\\textwidth}{!}{%\n'
    table = '\\begin{tabular}{l'+'c'*(noOfTests*2)+'cc}\n\cline{2-'+str(noOfTests*2+3)+'}\n'
    title1 = '\multicolumn{1}{c}{\\textbf{}} & '
    title2 = '\\textbf{Features}'
    for i in range(1,noOfTests+1):
        title1 += '\multicolumn{2}{c}{\\textbf{Test '+str(i)+'}} & '
        title2 += ' & \\textbf{Precision} & \\textbf{Recall}'
    title1 += '\multicolumn{2}{c}{\\textbf{\\begin{tabular}[c]{@{}c@{}}Confidence\\\\ Interval 99\%\end{tabular}}} \\\\ \hline \n'
    title2 += ' & \\textbf{Precision} & \\textbf{Recall} \\\\ \hline \hline \n'
    
    body = ''
    for i in range(df_dict[ml].shape[0]):
        line = f'{df_dict["knn"]["features"][i]}'
        for testNo in range(1,noOfTests+1):
            line += ' & '+format(df_dict[ml][f"test{testNo} precision"][i],'.4f')+' & '+format(df_dict[ml][f"test{testNo} recall"][i],'.4f')
        line += ' & '+format(df_dict[ml]["precision mean"][i],'.4f')+' $\pm$ '+format(df_dict[ml]["precision error"][i],'.4f')+' & '+format(df_dict[ml]["recall mean"][i],'.4f')+' $\pm$ '+format(df_dict[ml]["recall error"][i],'.4f')+' \\\\ \hline \n'
        body += line
    tail = '\end{tabular}%\n}\n\caption{'+ml.upper()+' experimental results.}\n\label{tab:'+ml.upper()+'_results}\n\end{table}\n'
    # combine all code sections of LaTeX table
    latex = head+table+title1+title2+body+tail
    
    # save as text file
    with open(f'{tables_path}{ml}_latex.txt','w') as f:
        f.write(latex)