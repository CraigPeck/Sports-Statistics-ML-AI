#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 15:33:53 2021

@author: craigpeck
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from TechnicalIndicators import *
from createFeaturesNew import *
from AI_DataSplit_CSV import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler 
import csv  
import pandas_datareader as webreader
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from lightgbm import LGBMRegressor
from lightgbm import *
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import *
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from speedml import Speedml
from sklearn.model_selection import cross_val_score
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.model_selection import learning_curve
from Learning_Curve_Def import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from SportStat_Data_Def import *
from GradientBoostingRegress_Def import *
from SVR_model_Def import *
from LightGBM_model_Def import *
from XGBoost_model_Def import *
from RandomForrest_model_Def import *
from Blender_SportStat_data import *

with open('nbateamstatsdataFull.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
   
with open('nbaavoffratdataFull.csv') as csv_file1:
    csv_reader = csv.reader(csv_file1, delimiter=',')


df11 = pd.read_csv("nbateamstatsdataFull.csv" )
df22 = pd.read_csv("nbaavoffratdataFull.csv")
df22 = pd.DataFrame({'AvOffRat': df22['0']})


df1 = df11.head(int(len(df11)*(75/100)))
df2 = df22.head(int(len(df22)*(75/100)))

df3 = df11.tail(int(len(df11)*(25/100)))
df4 = df22.tail(int(len(df22)*(25/100)))


Data = SportStatData(df1, df2)
Data_New = BlenderSportStatData(df3, df4)

X_train = pd.DataFrame(Data[0])
X_test = pd.DataFrame(Data[1])
Y_train = pd.DataFrame(Data[2])
Y_test = pd.DataFrame(Data[3])
real_game_score = pd.DataFrame(Data[4])
file_feature = pd.DataFrame(Data[5])
file_fit = pd.DataFrame(Data[6])

Blender_X_test = pd.DataFrame(Data_New[0])
Blender_Y_test = pd.DataFrame(Data_New[1])
Blender_Y_test = Blender_Y_test.reset_index()
Blender_Y_test = Blender_Y_test.drop(columns = "index")


## Call ML Models
GradBoostRegress = GradientBoostingRegress(X_train, Y_train, X_test, Y_test, file_feature, real_game_score, file_fit,Blender_X_test )
SVR = SVRegress(X_train, Y_train, X_test, Y_test, file_feature, real_game_score, file_fit,Blender_X_test)
LightGBM = LightGBM(X_train, Y_train, X_test, Y_test, file_feature, real_game_score, file_fit,Blender_X_test)
XGBoost = XGBoost(X_train, Y_train, X_test, Y_test, file_feature, real_game_score, file_fit,Blender_X_test)
RandomForrest = RandomForrestRegress(X_train, Y_train, X_test, Y_test, file_feature, real_game_score, file_fit,Blender_X_test)


## Predicted Data 
GradBoostRegress_test_predict = pd.DataFrame(GradBoostRegress[0])
GradBoostRegress_MAPE = pd.DataFrame(GradBoostRegress[1])
GradBoostRegress_Blend_pred = pd.DataFrame(GradBoostRegress[2])
GradBoostRegress_Blend_pred = pd.DataFrame({'Points Scored':GradBoostRegress_Blend_pred[0]})
SVR_test_predict = pd.DataFrame(SVR[0])
SVR_MAPE = pd.DataFrame(SVR[1]) 
SVR_Blend_pred = pd.DataFrame(SVR[2]) 
SVR_Blend_pred = pd.DataFrame({'Points Scored': SVR_Blend_pred[0]})                                           
LightGBM_test_predict = pd.DataFrame(LightGBM[0])
LightGBM_MAPE = pd.DataFrame(LightGBM[1])
LightGBM_Blend_pred = pd.DataFrame(LightGBM[2])
LightGBM_Blend_pred = pd.DataFrame({'Points Scored': LightGBM_Blend_pred[0]})
XGBoost_test_predict = pd.DataFrame(XGBoost[0])
XGBoost_MAPE = pd.DataFrame(XGBoost[1])
XGBoost_Blend_pred = pd.DataFrame(XGBoost[2])
XGBoost_Blend_pred = pd.DataFrame({'Points Scored': XGBoost_Blend_pred[0]})
RandomForrest_test_predict = pd.DataFrame(RandomForrest[0])
RandomForrest_MAPE = pd.DataFrame(RandomForrest[1])
RandomForrest_Blend_pred = pd.DataFrame(RandomForrest[2])
RandomForrest_Blend_pred = pd.DataFrame({'Points Scored': RandomForrest_Blend_pred[0]})


## Aggregate Data Predictions
Predicted_Data = pd.DataFrame({'Gradient Boost': GradBoostRegress_test_predict['Predicted Points Scored'], 
                                'SVR': SVR_test_predict['Predicted Points Scored'],
                                'LightGBM':LightGBM_test_predict['Predicted Points Scored'],
                                'XGBoost': XGBoost_test_predict['Predicted Points Scored'],
                                'Random Forrest': RandomForrest_test_predict['Predicted Points Scored']})

Blender_test_data = pd.DataFrame({'Gradient Boost': GradBoostRegress_Blend_pred['Points Scored'], 
                                'SVR': SVR_Blend_pred['Points Scored'],
                                'LightGBM':LightGBM_Blend_pred['Points Scored'],
                                'XGBoost': XGBoost_Blend_pred['Points Scored'],
                                'Random Forrest': RandomForrest_Blend_pred['Points Scored']})

## Define and train Blender model on the predictions from the level zero models

Blender_model = XGBRegressor()

Blender_model.fit(Predicted_Data, real_game_score)

Blender_test_predict = Blender_model.predict(Blender_test_data)


 #Plot results of the predictions against the true prices 
plt.figure(figsize=((10,8)))
plt.plot(Blender_Y_test, color = 'black', label = 'Actual Game Score')
plt.plot(Blender_test_predict, color = 'green', label = 'Predicted Predicted')
plt.title('Game Score Prediction: GradientBoost')
plt.xlabel('Games Played')
plt.ylabel('Points Scored')
plt.legend()
plt.show()
