#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:46:46 2021

@author: craigpeck
"""

import pandas as pd
import numpy as np
from TechnicalIndicators import *
from createFeaturesNew import *
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler 
import csv  
import pandas_datareader as webreader
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import train_test_split



def BlenderSportStatData(df3, df4):
    
    
    file = pd.DataFrame({'Points Scored': df3['Points Scored'], 
                               'FG': df3['FG'], 
                               'FG%': df3['FG%'], 
                               '3P': df3['3P'],
                               '3P%': df3['3P%'],
                               'ORB': df3['ORB'],
                               'Assists': df3['Assists'],
                               'Steals': df3['Steals'],
                               'OffRank': df3['OffRank'],
                               'OppDefRank': df3['OppDefRank'],
                               'Pace': df3['Pace'],
                               'Player Offensive Rating Avg': df4['AvOffRat']})
        
        
    
    
    file_fit = pd.DataFrame({'Points Scored': file['Points Scored']})

   
    ##Convert DataFrame into an array
    #file = np.array(file)
    #Call function createFeatures to create technical indicators from file 
    features = createFeaturesNew(file)
    
    file_feature = pd.DataFrame({'Points Scored EMA3': features['Points EMA3'],
                                 'FG EMA3': features['FG EMA3'], 
                                 'FG% EMA3' : features['FG% EMA3'],
                                 '3P EMA3' : features['3P EMA3'],
                                 '3P% EMA3' : features['3P% EMA3'],
                                 'ORB EMA3' : features['ORB EMA3'],
                                 'OffRank EMA3' : features['OffRank EMA3'],
                                 'OppDefRank' : file['OppDefRank']})


                                 
    ## Shift the data to predict games with past game data                            
    file_feature = file_feature.head(int(len(file_feature)))
    file_feature = file_feature.head(int(len(file_feature) - 1))                             
    
    
    file_fit = file_fit.head(int(len(file_fit)))
    file_fit = file_fit.tail(int(len(file_fit) - 1))
    
    
    #Create the scaler to normalize the data between 0 and 1 to make model training easier
    scaler = MinMaxScaler(feature_range=(0,1))
    #Create a second scaler for future inverse transform on correct data set size
    scaler_pred = MinMaxScaler(feature_range = (0,1))
    #Scale selected data to fit model 
    file_feature = scaler.fit_transform(np.array(file_feature))
    file_feature = pd.DataFrame(file_feature)
    file_fit = scaler_pred.fit_transform(np.array(file_fit))
    file_fit = pd.DataFrame(file_fit)

    
    return file_feature, file_fit
    