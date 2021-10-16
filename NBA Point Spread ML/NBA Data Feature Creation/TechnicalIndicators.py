#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 08:33:26 2021

@author: craigpeck
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import quandl
from itertools import chain



    
def createFeatures(file):
        file = pd.DataFrame(file)
   # Moving averages - different periods
        # file['MA200'] = file[0].rolling(window=200).mean() 
        # file['MA100'] = file[0].rolling(window=100).mean() 
        # file['MA50'] = file[0].rolling(window=50).mean() 
        # file['MA26'] = file[0].rolling(window=26).mean() 
        # file['MA20'] = file[0].rolling(window=20).mean() 
        file['MA5'] = file[0].rolling(window=5).mean() 
     
     
    # #  # SMA Differences - different periods
    #     file['DIFF-MA200-MA50'] = file['MA200'] - file['MA50']
    #     file['DIFF-MA200-MA100'] =file['MA200'] - file['MA100']
    #     file['DIFF-MA200-CLOSE'] = file['MA200'] - file[0]
    #     file['DIFF-MA100-CLOSE'] = file['MA100'] - file[0]
    #     file['DIFF-MA50-CLOSE'] = file['MA50'] - file[0]
    
    # # Moving Averages on high, lows, and std - different periods
        #file['MA200_low'] = file[1].rolling(window=200).min()
        # file['MA14_low'] = file[1].rolling(window=14).min()
        # #file['MA200_high'] = file[2].rolling(window=200).max()
        # file['MA14_high'] = file[2].rolling(window=14).max()
        # file['MA20dSTD'] = file[0].rolling(window=20).std() 
    
    # # Exponential Moving Averages (EMAS) - different periods
        file['EMA3'] = file[0].ewm(span=3, adjust=False).mean()
        # file['EMA20'] = file[0].ewm(span=20, adjust=False).mean()
        # file['EMA26'] = file[0].ewm(span=26, adjust=False).mean()
        # file['EMA100'] = file[0].ewm(span=100, adjust=False).mean()
        # file['EMA200'] = file[0].ewm(span=200, adjust=False).mean()

    # # Shifts (one day before and two days before)
        # file['close_shift-1'] = file.shift(-1)[0]
        # file['close_shift-2'] = file.shift(-2)[0]

    # # Bollinger Bands
        # file['Bollinger_Upper'] = file['MA20'] + (file['MA20dSTD'] * 2)
        # file['Bollinger_Lower'] = file['MA20'] - (file['MA20dSTD'] * 2)
    
    # # # Relative Strength Index (StochRSI)
    #     file['K-ratio'] = 100*((file[0] - file['MA14_low']) / (file['MA14_high'] - file['MA14_low']) )
    #     file['StochRSI'] = file['K-ratio'].rolling(window=3).mean() 
        
    #     # Moving Average Convergence/Divergence (MACD)
        #file['MACD'] = file['EMA12'] - file['EMA26']
        
        nareplace = file.at[file.index.max(), 0]    
        file.fillna((nareplace), inplace=True)
    
        return file