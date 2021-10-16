#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 18:38:51 2021

@author: craigpeck
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from pandas import datetime
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from itertools import chain



    
def createFeaturesNew(file):
    
        
        new_df = pd.DataFrame(file)
   # Moving averages - different periods
        new_df['Points MA5'] = new_df['Points Scored'].rolling(window=5).mean() 
        new_df['FG MA3'] = new_df['FG'].rolling(window = 3).mean()
        new_df['FG% MA3'] = new_df['FG%'].rolling(window = 3).mean()
        new_df['3P MA3'] = new_df['3P'].rolling(window = 3).mean()
        new_df['3P% MA3'] = new_df['3P%'].rolling(window = 3).mean()
        new_df['ORB MA3'] = new_df['ORB'].rolling(window = 3).mean()
        new_df['OffRank MA3'] = new_df['OffRank'].rolling(window = 3).mean()
        
        
     
     
    # #  # SMA Differences - different periods
    #     file['DIFF-MA200-MA50'] = file['MA200'] - file['MA50']
    #     file['DIFF-MA200-MA100'] =file['MA200'] - file['MA100']
    #     file['DIFF-MA200-CLOSE'] = file['MA200'] - file[0]
    #     file['DIFF-MA100-CLOSE'] = file['MA100'] - file[0]
    #     file['DIFF-MA50-CLOSE'] = file['MA50'] - file[0]
    
    # # # Moving Averages on high, lows, and std - different periods
    #     #file['MA200_low'] = file[1].rolling(window=200).min()
    #     new_df['MA14_low'] = new_df[1].rolling(window=14).min()
    #     #file['MA200_high'] = file[2].rolling(window=200).max()
    #     new_df['MA14_high'] = new_df[2].rolling(window=14).max()
        #new_df['MA20dSTD'] = new_df['Close'].rolling(window=20).std() 
    
    # # Exponential Moving Averages (EMAS) - different periods
        new_df['Points EMA3'] = new_df['Points Scored'].ewm(span=3, adjust=False).mean() 
        new_df['FG EMA3'] = new_df['FG'].ewm(span=3, adjust=False).mean()
        new_df['FG% EMA3'] = new_df['FG%'].ewm(span=3, adjust=False).mean()
        new_df['3P EMA3'] = new_df['3P'].ewm(span=3, adjust=False).mean()
        new_df['3P% EMA3'] = new_df['3P%'].ewm(span=3, adjust=False).mean()
        new_df['ORB EMA3'] = new_df['ORB'].ewm(span=3, adjust=False).mean()
        new_df['OffRank EMA3'] = new_df['OffRank'].ewm(span=3, adjust=False).mean()
        #new_df['EMA20'] = new_df['Close'].ewm(span=20, adjust=False).mean()
        #new_df['EMA26'] = new_df['Close'].ewm(span=26, adjust=False).mean()
        #file['EMA100'] = file[0].ewm(span=100, adjust=False).mean()
        #file['EMA200'] = file[0].ewm(span=200, adjust=False).mean()


    
    # # # Relative Strength Index (StochRSI)
    #     file['K-ratio'] = 100*((file[0] - file['MA14_low']) / (file['MA14_high'] - file['MA14_low']) )
    #     file['StochRSI'] = file['K-ratio'].rolling(window=3).mean() 
        
    #     # Moving Average Convergence/Divergence (MACD)
        #new_df['MACD'] = new_df['EMA12'] - new_df['EMA26']
        
        #nareplace = new_df.at[new_df.index.max(), '']    
        #new_df.fillna((nareplace), inplace=True)
        
        
    
        return new_df