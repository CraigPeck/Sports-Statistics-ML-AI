#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 21:04:45 2021

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


def LightGBM(X_train, Y_train, X_test, Y_test, file_feature, real_game_score, file_fit, Blender_X_test):
     #Create the scaler to normalize the data between 0 and 1 to make model training easier
    scaler = MinMaxScaler(feature_range=(0,1))
    #Create a second scaler for future inverse transform on correct data set size
    scaler_pred = MinMaxScaler(feature_range = (0,1))
    #Scale selected data to fit model
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
    
    
    train_predict=model.predict(X_train)
    
    
    test_predict_scaled = model.predict(X_test)
    test_predict = test_predict_scaled.reshape(-1,1)
    test_predict = scaler_pred.inverse_transform(test_predict)
    test_predict = pd.DataFrame(test_predict)
    test_predict = pd.DataFrame({'Predicted Points Scored': test_predict[0]})
    
    
    Blender_predict = model.predict(Blender_X_test)
    Blender_predict = Blender_predict.reshape(-1,1)
    Blender_predict = scaler_pred.inverse_transform(Blender_predict)
    Blender_predict = pd.DataFrame(Blender_predict)
    #Blender_predict = pd.DataFrame({'Predicted Points Scored': Blender_predict[0]})
    
    # Mean Absolute Percentage Error (MAPE) as a means to show model performance
    MAPE = np.mean((np.abs(np.subtract(Y_test, test_predict_scaled[0])/ Y_test))) * 10
    print('Mean Absolute Percentage Error (MAPE): ' + str(np.round(MAPE, 2)) + ' %')
    
    
    
    #Plot results of the predictions against the true prices 
    plt.figure(figsize=((10,8)))
    plt.plot(real_game_score, color = 'black', label = 'Actual Game Score')
    plt.plot(test_predict, color = 'green', label = 'Predicted Predicted')
    plt.title('Game Score Prediction: LightGBM')
    plt.xlabel('Games Played')
    plt.ylabel('Points Scored')
    plt.legend()
    plt.show()
    
    return test_predict, MAPE, Blender_predict