# -*- coding: utf-8 -*-
"""W&B_NN_Test_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1emv8e-hbgkRxt-Uu1wKmNzul7QIIbLun
"""

from google.colab import files
files.upload()

from google.colab import files
uploaded = files.upload()

from google.colab import files
uploaded = files.upload()

!pip install scikit-optimize
!pip install eli5

pip install wandb

import pandas as pd
import numpy as np
import tensorflow as tf
from createFeaturesMLBClass import *
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
from sklearn import svm 
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import scipy.stats as sp
from sklearn.model_selection import learning_curve
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier 
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
import wandb 
from wandb.keras import WandbCallback

df1 = pd.read_csv("MLBFullDataSet_16_20.csv")
df1 = pd.DataFrame(df1).head(1380)
df1 = df1.drop(columns = ['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.2'])

df2 = pd.read_csv("MLBstatdataFull2021_v5.csv")
df2.drop(columns = ['Unnamed: 0', 'Unnamed: 0.1'], inplace = True)

file =pd.DataFrame({'Winner': df1['Winner'],
                           'Runs_Scored': df1['Runs'],
                           'Runs_Allowed': df1['Runs_Allowed'], 
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
                           'Base_Out_Runs': df1['Base_Out_Runs'],
                           'Assists': df1['Assists'],
                           'Home_Runs_Givenup': df1['Home_Runs_Givenup'],
                           'Grounded_Balls_Allowed': df1['Grounded_Balls_Allowed'],
                           'Fly_Balls_Allowed': df1['Fly_Balls_Allowed'],
                           'Line_Drives_Allowed': df1['Line_Drives_Allowed'],
                           'Pitcher_Win_Contribution': df1['Pitcher_Win_Contribution'],
                           'Pitcher_Hits_Allowed_Per_Batter': df1['Pitcher_Hits_Allowed_Per_Batter'],
                           'Pitcher_Runs_Allowed_Per_Batter': df1['Pitcher_Runs_Allowed_Per_Batter'],
                           'Pitcher_ERA_Per_Batter': df1['Pitcher_ERA_Per_Batter'],
                           'Pitcher_Home_Runs_Allowed_Per_Batter': df1['Pitcher_Home_Runs_Allowed_Per_Batter'],
                           'Pitcher_Strikeouts_Per_Batter': df1['Pitcher_Strikeouts_Per_Batter'],
                           'Pythagorean W%': df1['Pythagorean W%']})

file_pitcher_data = pd.DataFrame({
                             
                             'Pitcher_Hits_Allowed_Per_Batter': df1['Pitcher_Hits_Allowed_Per_Batter'],
                             'Pitcher_Runs_Allowed_Per_Batter': df1['Pitcher_Runs_Allowed_Per_Batter'],
                             'Pitcher_ERA_Per_Batter': df1['Pitcher_ERA_Per_Batter'],
                             'Pitcher_Home_Runs_Allowed_Per_Batter': df1['Pitcher_Home_Runs_Allowed_Per_Batter'],
                             'Pitcher_Strikeouts_Per_Batter': df1['Pitcher_Strikeouts_Per_Batter']})

file_test =pd.DataFrame({'Winner': df2['Winner'],
                           'Runs_Scored': df2['Runs'],
                           'Runs_Allowed': df2['Runs_Allowed'], 
                           'At_Bats': df2['At_Bats'], 
                           'Hits': df2['Hits'], 
                           'RBI': df2['RBI'],
                           'Earned_Runs': df2['Earned_Runs'],
                           'Bases_on_Balls': df2['Bases_on_Balls'],
                           'Strikeouts': df2['Strikeouts'],
                           'Batting_Average': df2['Batting_Average'],
                           'On_Base_Percentage':df2['On_Base_Percentage'],
                           'Slugging_Percentage': df2['Slugging_Percentage'],
                           'Pitches_Faced': df2['Pitches_Faced'],
                           'Pitches_Pitched': df2['Pitches_Pitched'],
                           'Strikes_Earned': df2['Strikes_Earned'],
                           'Strikes_Given': df2['Strikes_Given'],
                           'Off_Win_Prob_Contribution': df2['Off_Win_Prob_Contribution'],
                           'Pressure_Pitcher_Faced': df2['Pressure_Pitcher_Faced'],
                           'Base_Out_Runs': df2['Base_Out_Runs'],
                           'Assists': df2['Assists'],
                           'Home_Runs_Givenup': df2['Home_Runs_Givenup'],
                           'Grounded_Balls_Allowed': df2['Grounded_Balls_Allowed'],
                           'Fly_Balls_Allowed': df2['Fly_Balls_Allowed'],
                           'Line_Drives_Allowed': df2['Line_Drives_Allowed'],
                           'Pitcher_Win_Contribution': df2['Pitcher_Win_Contribution'],
                           'Pitcher_Hits_Allowed_Per_Batter': df2['Pitcher_Hits_Allowed_Per_Batter'],
                           'Pitcher_Runs_Allowed_Per_Batter': df2['Pitcher_Runs_Allowed_Per_Batter'],
                           'Pitcher_ERA_Per_Batter': df2['Pitcher_ERA_Per_Batter'],
                           'Pitcher_Home_Runs_Allowed_Per_Batter': df2['Pitcher_Home_Runs_Allowed_Per_Batter'],
                           'Pitcher_Strikeouts_Per_Batter': df2['Pitcher_Strikeouts_Per_Batter'],
                           'Pythagorean W%': df2['Pythagorean W%']})

file_pitcher_data_test = pd.DataFrame({
                             
                             'Pitcher_Hits_Allowed_Per_Batter': df2['Pitcher_Hits_Allowed_Per_Batter'],
                             'Pitcher_Runs_Allowed_Per_Batter': df2['Pitcher_Runs_Allowed_Per_Batter'],
                             'Pitcher_ERA_Per_Batter': df2['Pitcher_ERA_Per_Batter'],
                             'Pitcher_Home_Runs_Allowed_Per_Batter': df2['Pitcher_Home_Runs_Allowed_Per_Batter'],
                             'Pitcher_Strikeouts_Per_Batter': df2['Pitcher_Strikeouts_Per_Batter']})

#Call function createFeatures to create technical indicators from file 
features = createFeaturesClassMLB(file)
features_test = createFeaturesClassMLB(file_test)


file = pd.DataFrame(file)
file_test = pd.DataFrame(file_test)

file_fit = pd.DataFrame({'Win': file['Winner']})
file_fit_test = pd.DataFrame({'Win': file_test['Winner']})

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
                             'Base_Out_Runs EMA' : features['Base_Out_Runs EMA3'],
                             'Home_Runs_Givenup EMA' : features['Home_Runs_Givenup EMA3'],
                             'Grounded_Balls_Allowed EMA' : features['Grounded_Balls_Allowed EMA3'],
                             'Fly_Balls_Allowed EMA' : features['Fly_Balls_Allowed EMA3'],
                             'Line_Drives_Allowed EMA' : features['Line_Drives_Allowed EMA3'],
                             'Pitcher_Win_Contribution EMA': features['Pitcher_Win_Contribution EMA3'],
                             'Pythagorean W%': file['Pythagorean W%']})

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
                             'Base_Out_Runs EMA' : features_test['Base_Out_Runs EMA3'],
                             'Home_Runs_Givenup EMA' : features_test['Home_Runs_Givenup EMA3'],
                             'Grounded_Balls_Allowed EMA' : features_test['Grounded_Balls_Allowed EMA3'],
                             'Fly_Balls_Allowed EMA' : features_test['Fly_Balls_Allowed EMA3'],
                             'Line_Drives_Allowed EMA' : features_test['Line_Drives_Allowed EMA3'],
                             'Pitcher_Win_Contribution EMA': features_test['Pitcher_Win_Contribution EMA3'],
                             'Pythagorean W%': file_test['Pythagorean W%']})

file_feature_adj = file_feature.head(int(len(file_feature) - 1)) 
file_pitcher_data = file_pitcher_data.tail(int(len(file_pitcher_data) - 1)).reset_index()
file_pitcher_data = file_pitcher_data.drop(columns = ['index'])

file_feature = pd.concat([file_feature_adj, file_pitcher_data], axis = 1)

print(file_feature.shape)
                           
file_fit = file_fit.head(int(len(file_fit)))
file_fit = file_fit.tail(int(len(file_fit) - 1))

file_feature_adj_test = file_feature_test.head(int(len(file_feature_test) - 1)).reset_index()
file_pitcher_data_test = file_pitcher_data_test.tail(int(len(file_pitcher_data_test) - 1)).reset_index()
file_pitcher_data_test = file_pitcher_data_test.drop(columns = ['index'])


file_feature_test = pd.concat([file_feature_adj_test, file_pitcher_data_test], axis = 1).drop(columns = ['index'])

print(file_feature_test.shape)


file_fit_test = file_fit_test.head(int(len(file_fit_test)))
file_fit_test = file_fit_test.tail(int(len(file_fit_test) - 1))

print(file_fit_test.shape)

#Create the scaler to normalize the data between 0 and 1 to make model training easier
scaler = MinMaxScaler(feature_range=(-1,1))
#Create a second scaler for future inverse transform on correct data set size
scaler_pred = MinMaxScaler(feature_range = (-1,1))
#Scale selected data to fit model

file_feature = pd.DataFrame(scaler.fit_transform(file_feature))
file_feature_test = pd.DataFrame(scaler.transform(file_feature_test))

feature_corr_data = pd.concat([file_feature, file_fit], axis = 1)

# use the pands .corr() function to compute pairwise correlations for the dataframe
corr = feature_corr_data.corr()
print(corr)
# visualise the data with seaborn
mask = np.triu(np.ones_like(corr, dtype=np.bool))
sns.set_style(style = 'white')
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(10, 250, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, square=True,
        linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

## Create training and test sets
X_train,X_test, Y_train, Y_test = train_test_split(file_feature, file_fit, train_size = 0.9, test_size = 0.1, random_state = 0, stratify = file_fit)

#Reshape Datasets

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

# Launch 20 experiments, trying different dropout rates
for run in range(20):
  # Start a run, tracking hyperparameters
  wandb.init(
      project="W&B Test",
      # Set entity to specify your username or team name
      # ex: entity="wandb",
      config={
          "layer_1": 120,
          "activation_1": "relu",
          "dropout": random.uniform(0.01, 0.80),
          "layer_2": 60,
          "activation_2": "relu",
          "layer_3": 30,
          "activation_3": "relu",
          "dropout": random.uniform(0.01, 0.80),
          "layer_4": 15,
          "activation_4": "relu",
          "dropout": random.uniform(0.01, 0.80),
          "layer_5": 1,
          "activation_5": "relu",
          "dropout": random.uniform(0.01, 0.80),
          "optimizer": "adam",
          "loss": "binary_crossentropy",
          "metric": "BinaryCrossentropy",
          "epoch": 50
          
      })
  config = wandb.config

# Create NN Using Keras w/ Tensoflow

NN_model = tf.keras.Sequential([

tf.keras.layers.Dense(config.layer_1, activation=config.activation_1,input_shape = (25,)),
tf.keras.layers.Dropout(config.dropout),
tf.keras.layers.Dense(config.layer_2, activation=config.activation_2),
tf.keras.layers.Dropout(config.dropout),
tf.keras.layers.Dense(config.layer_3, activation=config.activation_3),
tf.keras.layers.Dropout(config.dropout),
tf.keras.layers.Dense(config.layer_4, activation=config.activation_4),
tf.keras.layers.Dropout(config.dropout),
tf.keras.layers.Dense(config.layer_5, activation=config.activation_5),
tf.keras.layers.Dropout(config.dropout)
      ])

print(NN_model.summary())

NN_model.compile(optimizer=config.optimizer,
                loss=config.loss,
                metrics=[config.metric]
                )

logging_callback = WandbCallback(log_evaluation=True)

history = NN_model.fit(x=X_train, y=Y_train,
                      epochs=config.epoch,
                      validation_data=(X_test, Y_test),
                      callbacks=[logging_callback]
                      )
  
  wandb.finish()

results = NN_model.evaluate(X_test, Y_test, verbose = 0)
print('Test Loss : {:.4f} Test Acc. : {:.4f}'.format(*results))
print(results)

prediction = NN_model.predict(X_test)
prediction = pd.DataFrame(prediction)
print(prediction.head(25))