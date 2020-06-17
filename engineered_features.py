#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:14:37 2019

@author: carlosalcantara
"""

'''
Expand data with engineered features using the feature_engineering_function.py
Saves new csv file with specified name, overwriting input file if no save file
name is given.

Usage: engineered_features.py csvfile [savefile=csvfile]
'''
import pandas as pd
import sys
import feature_engineering_function

# Check for command line arguments
if len(sys.argv) < 1:
    print('Usage: engineered_features.py csvfile [savefile=csvfile]')
    sys.exit(-1)

# use original file name as new csv filename if none specified
file = sys.argv[1]
if len(sys.argv) > 2:
    savefile = sys.argv[2]
else:
    savefile = file

# read NetFlow data file
df = pd.read_csv(file)
# add engineered features
df = feature_engineering_function.BLINC_features(df)
# write NetFlow data file
df.to_csv(savefile, index=False)