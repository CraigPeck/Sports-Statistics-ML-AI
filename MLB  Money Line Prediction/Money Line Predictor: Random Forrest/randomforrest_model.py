# -*- coding: utf-8 -*-
"""RandomForrest_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12EJwuwu-o93NQLEVxyvAt9cxhKbmnC-x
"""

from google.colab import files
files.upload()

from google.colab import files
uploaded = files.upload()

from google.colab import files
uploaded = files.upload()

!pip install scikit-optimize
!pip install eli5

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
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

df1 = pd.read_csv("MLBFullDataSet_16_20.csv")
df1 = pd.DataFrame(df1).head(1380)
df1 = df1.drop(columns = ['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.2'])

df2 = pd.read_csv("MLBstatdataFull2021_v5.csv")
df2.drop(columns = ['Unnamed: 0', 'Unnamed: 0.1'], inplace = True)

#df_full = pd.concat([df1, df2], ignore_index = True)

#df1 =df_full.head(int(0.8*len(df_full)))
#df2 =df_full.tail(int(0.2*len(df_full)))

#print(df1.shape)

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

print(file_fit.shape)

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

kpca = KernelPCA(n_components = 10, kernel = 'rbf', gamma = 25)

file_feature = pd.DataFrame(kpca.fit_transform(file_feature))
file_feature_test = pd.DataFrame(kpca.transform(file_feature_test))

file_corr_data = pd.concat([file_feature, file_fit], axis = 1)

print(file_corr_data.head(25))
# use the pands .corr() function to compute pairwise correlations for the dataframe
corr = file_corr_data.corr()
# visualise the data with seaborn
mask = np.triu(np.ones_like(corr, dtype=np.bool))
sns.set_style(style = 'white')
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(10, 250, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, square=True,
        linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

## Create training and test sets
X_train,X_test, Y_train, Y_test = train_test_split(file_feature, file_fit, train_size = 0.9, test_size = 0.1, random_state = 0, stratify = file_fit)


print(Y_test)

#param_grid = {'criterion': Categorical(['gini', 'entropy']),'n_estimators': Integer(10, 100), 'max_features': Categorical(['auto', 'sqrt','log2']), 'max_depth': Integer(1,10) }

param_grid = {'criterion': Categorical(['gini', 'entropy']) }



#Generate the model we will use

model = RandomForestClassifier()


cv = KFold(5, shuffle=True, random_state=1).get_n_splits(X_train)

model = BayesSearchCV(model, param_grid, n_iter = 50, scoring = 'roc_auc', cv=cv, return_train_score = True, refit = True).fit(X_train, Y_train)

scores = cross_val_score(model, X_train, Y_train, cv = cv, scoring = "roc_auc")
print(scores)

model = model.best_estimator_

## Get the feature weights of the model for feature selection
perm = PermutationImportance(model, random_state=1).fit(X_test, Y_test)
Weights_features = eli5.explain_weights_df(perm, feature_names = file_feature.columns.tolist())
print(Weights_features)

test_predict_scaled = model.predict(X_test)
test_predict = test_predict_scaled.reshape(-1,1)
test_predict = pd.DataFrame(test_predict)


print(test_predict.head(25))
print(Y_test)

#Y_test = pd.DataFrame(real_game_score)
Confusion_Matrix = confusion_matrix(Y_test, test_predict)
print(Confusion_Matrix)
Prediction_Accuracy = ((Confusion_Matrix[0,0]+Confusion_Matrix[1,1])/(len(Y_test)))*100
print(Prediction_Accuracy)

test_set_predictions = model.predict(file_feature_test)
test_set_predictions = test_set_predictions.reshape(-1,1)
test_set_predictions = pd.DataFrame(test_set_predictions)
test_set_predictions = pd.DataFrame({'Predicted_Outcome': test_set_predictions[0]})
print(test_set_predictions.head(25))

prediction_probabilities = model.predict_proba(file_feature_test)
prediction_probabilities = pd.DataFrame(prediction_probabilities)
print(prediction_probabilities.head(50))

Test_Set_Confusion_Matrix = confusion_matrix(file_fit_test, test_set_predictions)
print(Test_Set_Confusion_Matrix)
Test_Set_Prediction_Accuracy = ((Test_Set_Confusion_Matrix[0,0]+Test_Set_Confusion_Matrix[1,1])/(len(file_fit_test)))*100
print(Test_Set_Prediction_Accuracy)

predictions_and_probs = pd.concat([test_set_predictions, file_fit_test, prediction_probabilities], axis = 1 )
predictions_and_probs = pd.DataFrame({'Predicted_Outcome': predictions_and_probs['Predicted_Outcome'], 'Actual_Outcome': predictions_and_probs['Win'], 'Prob_Loss': predictions_and_probs[0], 'Prob_Win': predictions_and_probs[1]})
print(predictions_and_probs.head(20))

from google.colab import drive
drive.mount('drive')

test_set_predictions.to_csv('test_predictions_v6.csv')
!cp test_predictions_v6.csv "drive/My Drive/"

predictions_and_probs.to_csv('predictions_and_probs_v6.csv')
!cp predictions_and_probs_v6.csv "drive/My Drive/"