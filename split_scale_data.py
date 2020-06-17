#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:58:52 2020

@author: carlosalcantara
"""

'''
Separates data into test and train sets in a 2:1 ratio (33% train, 67% test) using
stratified sampling according to class label. Further subdivides test sets into 
separate subsets maintaining the stratified sampling. Resulting train/test data sets
are then standardized (0 mean, unit variance), with scaler object saved for future
use if necessary. Finally, generates LaTeX table code where the number of data rows
for each class are noted for each trian and test data sets.

Usage: python3 split_scale_data.py #ofTests path/to/csv/with/classes/ path/to/save/new/files/ path/to/save/latex/table/and/scaler/object/
'''
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pickle import dump
import sys

if len(sys.argv) < 4:
    print('Usage: python3 split_scale_data.py #ofTests path/to/csv/with/classes/ path/to/save/new/files/ path/to/save/latex/table/and/scaler/object/')
    exit()

NoOfTests = sys.argv[1]
tables_path = sys.argv[2]

# Provide file location for csv data file
file = sys.argv[3]

# Provide file location for csv data file
saveFilePath = sys.argv[4]

# Columns to be scaled
features = ['duration', 'dPkts', 'dOctets','dstaddrcount', 'srcportcount', 'dstportunique']

df = pd.read_csv(file)     # Load csv

###############################################################################
#                              TRAIN TEST SPLIT
###############################################################################

# Dictionary of dataframes by class
class_dict = dict(tuple(df.groupby(df['class'])))

# create dataframes
train = pd.DataFrame()

test_dict ={}
for t in range(1,NoOfTests + 1):
    test_dict[t] = pd.DataFrame()

for i in class_dict:        # iterate over all classes
    print(i)
    # split class df into train and test sets
    train_temp, test_temp = train_test_split(class_dict[i], test_size=0.33, random_state=15)
    # add to overall train df
    train = train.append(train_temp)
    # reset index
    test_temp = test_temp.reset_index(drop=True)
    # split test set into 6 subsets
    for t in range(1,NoOfTests + 1):
        test_dict[t] = test_dict[t].append(test_temp[test_temp.index % NoOfTests+1 == t])

###############################################################################
#                                STANDARDIZE
###############################################################################

labels = list(train['class']) # Save class labels as they are not to be standardized
train = train[features]       # reduce df to only the pertinent feature columns

scaler = StandardScaler()     # create scaler object
df_scaled = scaler.fit_transform(train) # fit scaler object and scale the data

# create df with scaled values
train = pd.DataFrame(data = df_scaled, columns = features)
train['class'] = labels       # add the class labels

# standardize test dfs
for i in range(1,NoOfTests + 1):
    # Save class labels as they are not to be standardized
    labels = list(test_dict[i]['class'])    
    # reduce df to only the pertinent feature columns
    test_dict[i] = test_dict[i][features]
    
    # scale the test subsets
    temp = scaler.transform(test_dict[i])
    # transform back into df
    test_dict[i] = pd.DataFrame(data = temp, columns = features)
    test_dict[i]['class'] = labels       # add the class labels

# save the scaler
dump(scaler, open(tables_path+'scaler.pkl', 'wb'))

###############################################################################
#                                  SAVE CSV
###############################################################################

for i in range(1,NoOfTests + 1):
    test_dict[i].to_csv(saveFilePath+'test'+str(i)+'_scale.csv',index=False)
train.to_csv(saveFilePath+'train_scale.csv',index=False)

###############################################################################
#                              LATEX TABLE GENERATION
###############################################################################

# save class labels
index = df['class'].unique()
index.sort()

# create table df
table = pd.DataFrame(index=index)
        
# load table df with data from all train and test sets
table['Train'] = train.groupby('class').count()['dPkts']
for i in range(1, NoOfTests+1):
    table[i] = test_dict[i].groupby('class').count()['dPkts']

# create latex table from table df
head = '\\begin{table}[htpb]\n\centering\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{l'+'r'*(NoOfTests)+'r}\n\cline{2-'+str(NoOfTests+2)+'}\n'

heading = '& \multicolumn{1}{c}{Train}'
for i in range(1,NoOfTests+1):
    heading += '& \multicolumn{1}{c}{Test '+str(i)+'}'
heading += ' \\\\ \hline \hline\n'

body = ''
for i, row in table.iterrows():
    line = f'{" ".join([w.title() if w.islower() else w for w in i.replace("_"," ").split()])} & {row["Train"]}'
    for t in range(1,NoOfTests+1):
        line += f' & {row[t]}'
    line += ' \\\\ \hline\n'
    body += line
    
tail = '''\end{tabular}
}
\caption{Train and Test sets breakdown by class label}
\label{tab:datasets_breakdown}
\end{table}
'''

latex = head+heading+body+tail

# save latex table to text file
with open(f'{tables_path}train_test_split_latex.txt','w') as f:
    f.write(latex)
