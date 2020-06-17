#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:33:44 2019

@author: carlosalcantara
"""

import pandas as pd

def BLINC_features(originaldf, savefile = False, filename = 'BLINC_features.csv'):
    """
    This function computes flow features based on the BLINC research
    
    Args:
        originaldf (pandas.DataFrame): df to be augmented with engineered features.
        savefile (bool): Set True to save resulting df as csv in addition to being returned. Default False.
        filename (str): File name for csv if savefile is True. Default 'BLINC_features.csv'
        
    Returns:
        originaldf (pandas.DataFrame): df augmented with engineered features.
    """

    # Create featuredf with unique source address
    featuresdf = pd.DataFrame(columns=['srcaddr','dstaddrcount','srcportcount','dstportunique'])
    featuresdf['srcaddr'] = list(originaldf['srcaddr'].unique())

    # find unique src/dst port and dst address counts and add to featuresdf
    for i, row in featuresdf.iterrows():
        tempdf = originaldf[originaldf['srcaddr'] == row['srcaddr']]
        featuresdf['dstaddrcount'][i] = len(tempdf['dstaddr'].unique())
        featuresdf['srcportcount'][i] = len(tempdf['srcport'].unique())
        featuresdf['dstportunique'][i] = len(tempdf['dstport'].unique())
        print(f'{(i/len(featuresdf))*100:.0f}%')	# print percentage completed

    # append engineered data from featuresdf to originaldf
    originaldf = originaldf.reset_index().merge(featuresdf, how='left').set_index('index')
    if savefile == True:
    	# save originaldf with features to csv
    	originaldf.to_csv(filename, index = False)
    return originaldf