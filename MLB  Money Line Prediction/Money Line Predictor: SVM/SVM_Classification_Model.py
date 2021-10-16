#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:27:03 2021

@author: craigpeck
"""


import pandas as pd
import numpy as np
import tensorflow as tf
from TechnicalIndicators import *
from createFeaturesMLBClass import *
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
import scipy.stats as sp
from sklearn.model_selection import learning_curve
from sklearn import svm 
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer


with open('mlbteamstatsdataFull1.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

with open('mlbteamstatstestset.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')


df1 = pd.read_csv("mlbteamstatsdataFull1.csv")

df_test = pd.read_csv("mlbteamstatstestset.csv")


df_winner_adj = df1['Winner']

for i in range(len(df_winner_adj)):
    if df_winner_adj[i] == 0:
        df_winner_adj[i] = -1
    else:
        df_winner_adj[i] = 1





file =pd.DataFrame({'Winner': df1['Winner'],
                           'Runs_Scored': df1['Runs'], 
                           'At_Bats': df1['At_Bats'], 
                           'Hits': df1['Hits'], 
                           'RBI': df1['RBI'],
                           'Earned_Runs': df1['Earned_Runs'],
                           'Bases_on_Balls': df1['Bases_on_Balls'],
                           'Strikeouts': df1['Strikeouts'],
                           'Batting_Average': df1['Batting_Average'],
                           'On_Base_Percentage':df1['On_Base_Percentage'],
                           'Slugging_Percentage': df1['Slugging_Percentage'],
                           'Pitches_Faced': df1['Pitches_Faced'],
                           'Pitches_Pitched': df1['Pitches_Pitched'],
                           'Strikes_Earned': df1['Strikes_Earned'],
                           'Strikes_Given': df1['Strikes_Given'],
                           'Off_Win_Prob_Contribution': df1['Off_Win_Prob_Contribution'],
                           'Pressure_Pitcher_Faced': df1['Pressure_Pitcher_Faced'],
                           'Off_Win_Prob_Sub': df1['Off_Win_Prob_Sub'],
                           'Base_Out_Runs': df1['Base_Out_Runs'],
                           'Assists': df1['Assists'],
                           'Home_Runs_Givenup': df1['Home_Runs_Givenup'],
                           'Grounded_Balls_Allowed': df1['Grounded_Balls_Allowed'],
                           'Fly_Balls_Allowed': df1['Fly_Balls_Allowed'],
                           'Line_Drives_Allowed': df1['Line_Drives_Allowed'],
                           'Pitcher_Win_Contribution': df1['Pitcher_Win_Contribution']})


file_test =pd.DataFrame({'Winner': df_test['Winner'],
                           'Runs_Scored': df_test['Runs'], 
                           'At_Bats': df_test['At_Bats'], 
                           'Hits': df_test['Hits'], 
                           'RBI': df_test['RBI'],
                           'Earned_Runs': df_test['Earned_Runs'],
                           'Bases_on_Balls': df_test['Bases_on_Balls'],
                           'Strikeouts': df_test['Strikeouts'],
                           'Batting_Average': df_test['Batting_Average'],
                           'On_Base_Percentage':df_test['On_Base_Percentage'],
                           'Slugging_Percentage': df_test['Slugging_Percentage'],
                           'Pitches_Faced': df_test['Pitches_Faced'],
                           'Pitches_Pitched': df_test['Pitches_Pitched'],
                           'Strikes_Earned': df_test['Strikes_Earned'],
                           'Strikes_Given': df_test['Strikes_Given'],
                           'Off_Win_Prob_Contribution': df_test['Off_Win_Prob_Contribution'],
                           'Pressure_Pitcher_Faced': df_test['Pressure_Pitcher_Faced'],
                           'Off_Win_Prob_Sub': df_test['Off_Win_Prob_Sub'],
                           'Base_Out_Runs': df_test['Base_Out_Runs'],
                           'Assists': df_test['Assists'],
                           'Home_Runs_Givenup': df_test['Home_Runs_Givenup'],
                           'Grounded_Balls_Allowed': df_test['Grounded_Balls_Allowed'],
                           'Fly_Balls_Allowed': df_test['Fly_Balls_Allowed'],
                           'Line_Drives_Allowed': df_test['Line_Drives_Allowed'],
                           'Pitcher_Win_Contribution': df_test['Pitcher_Win_Contribution']})


#Call function createFeatures to create technical indicators from file 
features = createFeaturesClassMLB(file)

features_test = createFeaturesClassMLB(file_test)

file = pd.DataFrame(file)

file_fit = pd.DataFrame({'Win': file['Winner']})

file_fit_test = pd.DataFrame({'Win': file_test['Winner']})

#Combine any chosen features such as closing price and a technical indicator into a single DataFrame 

file_feature_test = pd.DataFrame({'Runs_Scored EMA3': features_test['Runs_Scored EMA3'],
                             'At_Bats EMA3': features_test['At_Bats EMA3'], 
                             'Hits EMA3' : features_test['Hits EMA3'],
                             'RBI EMA3' : features_test['RBI EMA3'],
                             'Earned_Runs EMA3' : features_test['Earned_Runs EMA3'],
                             'Bases_on_Balls EMA3' : features_test['Bases_on_Balls EMA3'],
                             'Strikeouts EMA3' : features_test['Strikeouts EMA3'],
                             'On_Base_Percentage EMA' : features_test['On_Base_Percentage EMA3'],
                             'Slugging_Percentage EMA' : features_test['Slugging_Percentage EMA3'],
                             'Pitches_Faced EMA' : features_test['Pitches_Faced EMA3'],
                             'Pitches_Pitched EMA' : features_test['Pitches_Pitched EMA3'],
                             'Strikes_Earned EMA' : features_test['Strikes_Earned EMA3'],
                             'Strikes_Given EMA' : features_test['Strikes_Given EMA3'],
                             #'Off_Win_Prob_Sub EMA' : features['Off_Win_Prob_Sub EMA3'],
                             'Base_Out_Runs EMA' : features_test['Base_Out_Runs EMA3'],
                             'Assists EMA' : features_test['Assists EMA3'],
                             'Home_Runs_Givenup EMA' : features_test['Home_Runs_Givenup EMA3'],
                             'Grounded_Balls_Allowed EMA' : features_test['Grounded_Balls_Allowed EMA3'],
                             'Fly_Balls_Allowed EMA' : features_test['Fly_Balls_Allowed EMA3'],
                             'Line_Drives_Allowed EMA' : features_test['Line_Drives_Allowed EMA3'],
                             'Pitcher_Win_Contribution EMA' : features_test['Pitcher_Win_Contribution EMA3']})

file_feature = pd.DataFrame({'Runs_Scored EMA3': features['Runs_Scored EMA3'],
                             'At_Bats EMA3': features['At_Bats EMA3'], 
                             'Hits EMA3' : features['Hits EMA3'],
                             'RBI EMA3' : features['RBI EMA3'],
                             'Earned_Runs EMA3' : features['Earned_Runs EMA3'],
                             'Bases_on_Balls EMA3' : features['Bases_on_Balls EMA3'],
                             'Strikeouts EMA3' : features['Strikeouts EMA3'],
                             'On_Base_Percentage EMA' : features['On_Base_Percentage EMA3'],
                             'Slugging_Percentage EMA' : features['Slugging_Percentage EMA3'],
                             'Pitches_Faced EMA' : features['Pitches_Faced EMA3'],
                             'Pitches_Pitched EMA' : features['Pitches_Pitched EMA3'],
                             'Strikes_Earned EMA' : features['Strikes_Earned EMA3'],
                             'Strikes_Given EMA' : features['Strikes_Given EMA3'],
                             #'Off_Win_Prob_Sub EMA' : features['Off_Win_Prob_Sub EMA3'],
                             'Base_Out_Runs EMA' : features['Base_Out_Runs EMA3'],
                             'Assists EMA' : features['Assists EMA3'],
                             'Home_Runs_Givenup EMA' : features['Home_Runs_Givenup EMA3'],
                             'Grounded_Balls_Allowed EMA' : features['Grounded_Balls_Allowed EMA3'],
                             'Fly_Balls_Allowed EMA' : features['Fly_Balls_Allowed EMA3'],
                             'Line_Drives_Allowed EMA' : features['Line_Drives_Allowed EMA3'],
                             'Pitcher_Win_Contribution EMA' : features['Pitcher_Win_Contribution EMA3']})


                             
 ## Shift the data to predict games with past game data                            
file_feature = file_feature.head(len(file_feature))
file_feature = file_feature.head(int(len(file_feature) - 1)) 


file_feature_test = file_feature_test.head(len(file_feature_test))
file_feature_test = file_feature_test.head(int(len(file_feature_test) - 1))                            


file_fit = file_fit.head(int(len(file_fit)))
file_fit = file_fit.tail(int(len(file_fit) - 1))

file_fit_test = file_fit_test.head(int(len(file_fit_test)))
file_fit_test = file_fit_test.tail(int(len(file_fit_test) - 1))

file_corr_data = pd.concat([file_feature, file_fit], axis = 1)

# use the pands .corr() function to compute pairwise correlations for the dataframe
corr = file_feature.corr()
# visualise the data with seaborn
mask = np.triu(np.ones_like(corr, dtype=np.bool))
sns.set_style(style = 'white')
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(10, 250, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, square=True,
        linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

## Create training and test sets
X_train,X_test, Y_train, Y_test = train_test_split(file_feature, file_fit, train_size = 0.9, test_size = 0.1, random_state = 0)



#Fix index from train_test_split 
real_game_score = Y_test.sort_index()
real_game_score = real_game_score.reset_index()
real_game_score = real_game_score['Win']

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
file_feature_test = scaler.fit_transform(np.array(file_feature_test))
file_feature_test = pd.DataFrame(file_feature_test)




# ## Create a grid search for optimal hyperparameters 
param_grid = {'C': Integer(1, 500), 'gamma': Real(0.001, 0.3), 'kernel': Categorical({"linear", "rbf", "poly"}) }

#Generate the SVM model we will use
model = svm.SVC()


cv = KFold(5, shuffle=True, random_state=1).get_n_splits(X_train)

model = BayesSearchCV(model, param_grid, n_iter = 20, scoring = 'precision', cv=cv, return_train_score = True, refit = True).fit(X_train, Y_train)


scores = cross_val_score(model, X_train, Y_train, cv = cv, scoring = "precision")

model = model.best_estimator_


## Get the feature weights of the model for feature selection
perm = PermutationImportance(model, random_state=1).fit(X_test, Y_test)
Weights_features = eli5.explain_weights_df(perm, feature_names = file_feature.columns.tolist())


test_predict_scaled = model.predict(X_test)
test_predict = test_predict_scaled.reshape(-1,1)
test_predict = scaler_pred.inverse_transform(test_predict)
test_predict = pd.DataFrame(test_predict)
test_predict = pd.DataFrame({'Predicted Points Scored': test_predict[0]})



test_predict_test = model.predict(file_feature_test)
test_predict_test = test_predict_test.reshape(-1,1)
test_predict_test = scaler_pred.inverse_transform(test_predict_test)
test_predict_test = pd.DataFrame(test_predict_test)
Test_Confusion_Matrix = confusion_matrix(file_fit_test, test_predict_test)
Test_Prediction_Accuracy = ((Test_Confusion_Matrix[0,0]+Test_Confusion_Matrix[1,1])/(len(file_fit_test)))*100


#pickle.dump(model, open("test_RandForrClass_model_file.pkl", "wb"))

### call back saved model code: restored_model = pickle.load(open("test_svc_model_file.pkl", "rb"))
# prediction = restored_model.predict(new_data)



Y_test = pd.DataFrame(real_game_score)
Confusion_Matrix = confusion_matrix(Y_test, test_predict)
Prediction_Accuracy = ((Confusion_Matrix[0,0]+Confusion_Matrix[1,1])/(len(Y_test)))*100
print(Prediction_Accuracy)

#Plot results of the predictions against the true prices 
plt.figure(figsize=((10,8)))
plt.plot(real_game_score, color = 'black', label = 'Actual Game Score')
plt.plot(test_predict, color = 'green', label = 'Predicted Predicted')
plt.title('Game Score Prediction')
plt.xlabel('Games Played')
plt.ylabel('Points Scored')
plt.legend()
plt.show()