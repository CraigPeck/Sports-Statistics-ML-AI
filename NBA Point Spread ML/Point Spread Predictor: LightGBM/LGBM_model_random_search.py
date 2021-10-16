#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 16:17:58 2021

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

with open('nbateamstatsdata.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
   
with open('nbaavoffratdata.csv') as csv_file1:
    csv_reader = csv.reader(csv_file1, delimiter=',')

df1 = pd.read_csv("nbateamstatsdata.csv")
df2 = pd.read_csv("nbaavoffratdata.csv")
df2 = pd.DataFrame({'AvOffRat': df2['0']})


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



########Convert DataFrame into an array
file = np.array(file)
#Call function createFeatures to create technical indicators from file 
features = createFeaturesNew(file)

file = pd.DataFrame(file)

file_fit = pd.DataFrame({'Points Scored': file['Points Scored']})

#Combine any chosen features such as closing price and a technical indicator into a single DataFrame 

file_feature = pd.DataFrame({'Points Scored EMA3': features['Points EMA3'],
                             'FG EMA3': features['FG EMA3'], 
                             'FG% EMA3' : features['FG% EMA3'],
                             '3P EMA3' : features['3P EMA3'],
                             '3P% EMA3' : features['3P% EMA3'],
                             'ORB EMA3' : features['ORB EMA3'],
                             'OffRank EMA3' : features['OffRank EMA3'],
                             'OppDefRank' : file['OppDefRank']})
                             
 ## Shift the data to predict games with past game data                            
file_feature = file_feature.head(60)
file_feature = file_feature.head(59)                             


file_fit = file_fit.head(60)
file_fit = file_fit.tail(59)

## Create training and test sets
X_train,X_test, Y_train, Y_test = train_test_split(file_feature, file_fit, train_size = 0.75, test_size = 0.25, random_state = 0)



## Eplore data with speedml
sml = Speedml('HOUX_train_new.csv', 'HOUX_test_new.csv', target = 'Points Scored')
Speed_shape = sml.shape()
Speed_ml_info = sml.configure('overfit_threshold', sml.np.sqrt(sml.train.shape[0]) / sml.train.shape[0])
sml.plot.correlate()
sml.plot.distribute()
sml.feature.impute()



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

## Create a grid search for optimal hyperparameters 
param_grid = [{ 'max_depth': sp_randint(10, 100), 'num_leaves': sp_randint(10, 100)}]

#Generate the LSTM model we will use
model=LGBMRegressor()

cv = KFold(5, shuffle=True, random_state=1).get_n_splits(X_train)

rand_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter = 100, scoring = 'neg_mean_squared_error', cv=cv, return_train_score = True)

##Fit rand_search to firnd the best estimator
rand_search.fit(X_train, Y_train)
## Fit the model based on the Hyperparameter grid search
model = rand_search.best_estimator_
model.fit(X_train, Y_train)

n_scores = cross_val_score(model, X_train, Y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
n_scores_mean = n_scores.mean()
n_scores_std = n_scores.std()
print('MAE: %.3f (%.3f)' % ( -1 * mean(n_scores), std(n_scores)))


##Plot the Learning Curves
Visualizer = plot_learning_curve(model, X_train, Y_train, cv)
Visualizer.show()
LearningC = plot_total_learning_curve(model, X_train, X_test, Y_train, Y_test)
LearningC.show()


## Get the feature weights of the model for feature selection
perm = PermutationImportance(rand_search, random_state=1).fit(X_test, Y_test)
Weights_features = eli5.explain_weights_df(perm, feature_names = file_feature.columns.tolist())

## Pull hyperparameters for grid search
Rand_Search_HP =rand_search.best_params_
Best_Score = model.best_score_


train_predict=model.predict(X_train)


test_predict_scaled = model.predict(X_test)
test_predict = test_predict_scaled.reshape(-1,1)
test_predict = scaler_pred.inverse_transform(test_predict)
test_predict = pd.DataFrame(test_predict)
test_predict = pd.DataFrame({'Predicted Points Scored': test_predict[0]})


# Mean Absolute Percentage Error (MAPE) as a means to show model performance
MAPE = np.mean((np.abs(np.subtract(Y_test, test_predict_scaled[0])/ Y_test))) * 10
print('Mean Absolute Percentage Error (MAPE): ' + str(np.round(MAPE, 2)) + ' %')



#Plot results of the predictions against the true prices 
plt.figure(figsize=((10,8)))
plt.plot(real_game_score, color = 'black', label = 'Actual Game Score')
plt.plot(test_predict, color = 'green', label = 'Predicted Predicted')
plt.title('Game Score Prediction')
plt.xlabel('Games Played')
plt.ylabel('Points Scored')
plt.legend()
plt.show()