#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 19:30:05 2021

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



def SportStatData(df1, df2):
    
    
    file = pd.DataFrame({'Points Scored': df1['Points Scored'], 
                               'FG': df1['FG'], 
                               'FG%': df1['FG%'], 
                               '3P': df1['3P'],
                               '3P%': df1['3P%'],
                               'ORB': df1['ORB'],
                               'Assists': df1['Assists'],
                               'Steals': df1['Steals'],
                               'OffRank': df1['OffRank'],
                               'OppDefRank': df1['OppDefRank'],
                               'Pace': df1['Pace'],
                               'Player Offensive Rating Avg': df2['AvOffRat']})
        
        
    
    
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
    
    ## Create training and test sets
    X_train,X_test, Y_train, Y_test = train_test_split(file_feature, file_fit, train_size = 0.75, test_size = 0.25, random_state = 0)
    
    #Fix index from train_test_split 
    real_game_score = Y_test.sort_index()
    real_game_score = real_game_score.reset_index()
    real_game_score = real_game_score['Points Scored']
    
    #Create the scaler to normalize the data between 0 and 1 to make model training easier
    scaler = MinMaxScaler(feature_range=(0,1))
    #Create a second scaler for future inverse transform on correct data set size
    scaler_pred = MinMaxScaler(feature_range = (0,1))
    #Scale selected data to fit model 
    X_train = scaler.fit_transform(np.array(X_train))
    X_train = pd.DataFrame(X_train)
    Y_train = scaler.fit_transform(np.array(Y_train))
    Y_train = pd.DataFrame(Y_train)
    X_test = scaler.fit_transform(np.array(X_test))
    X_test = pd.DataFrame(X_test)
    Y_test = scaler.fit_transform(np.array(Y_test))
    Y_test = pd.DataFrame(Y_test)
    np_scaled = scaler_pred.fit(np.array(file_fit))
    
    return X_train, X_test, Y_train, Y_test, real_game_score, file_feature, file_fit