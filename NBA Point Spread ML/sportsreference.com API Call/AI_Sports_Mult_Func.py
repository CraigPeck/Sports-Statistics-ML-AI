#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 20:15:08 2021

@author: craigpeck
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from TechnicalIndicators import *
from createFeaturesNew import *
from AI_DataSplit_CSV import *
import quandl
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler 
import csv  
from datetime import date, timedelta, datetime
import pandas_datareader as webreader




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


file = file.head(60)

########Convert DataFrame into an array
file = np.array(file)
#Call function createFeatures to create technical indicators from file 
features = createFeatures(file)

#Combine any chosen features such as closing price and a technical indicator into a single DataFrame 
#file_feature = pd.DataFrame({'Adj Close': features[0], 'Bollinger Band Upper': features['Bollinger_Upper'], 'Bollinger Band Lower' : features['Bollinger_Lower'], 'MACD' : features['MACD'] })
file_feature = pd.DataFrame({'Points Scored': features[0], 
                             'FG': features[1], 
                             'FG%' : features[2],
                             '3P' : features[3],
                             '3P%' : features[4],
                             'ORB' : features[5],
                             #'Assists' : features[6],
                             #'Steals' : features[7],
                             'OffRank' : features[8],
                             'OppDefRank' : features[9]})
                             #'EMA3' : features['EMA3']}) 
                             #'MA5': features['MA5']})



#Create the scaler to normalize the data between 0 and 1 to make model training easier
scaler = MinMaxScaler(feature_range=(0,1))
#Create a second scaler for future inverse transform on correct data set size
scaler_pred = MinMaxScaler(feature_range = (0,1))
#Scale selected data to fit model 
file_scaled = scaler.fit_transform(np.array(file_feature).reshape(-8,8))
df_adj_close = file[:,0]
np_close_scaled = scaler_pred.fit(np.array(df_adj_close).reshape(-1,1))

#Create new DataFrame from scaled data
file = pd.DataFrame(file_scaled)

#Create training and test sets using DataProcessing class 
process = DataProcessing(0.6, file)
process.gen_test(3)
process.gen_train(3)


X_train = process.X_train
Y_train = process.Y_train 
Y_train = Y_train[:,:, 0]


X_test = process.X_test
Y_test = process.Y_test 
Y_test = Y_test[:,:, 0]

#Set aside actual stock prices for later comparrison to the predicted prices
real_stock_price = scaler.inverse_transform(X_test[:, 0])
real_stock_price = real_stock_price[:,0]



#Generate the LSTM model we will use
model=Sequential()

#Define the number of neurons in the model based on the size of the input training set
n_nuerons = X_train.shape[1] * X_train.shape[2]

#Create first layer of the LSTM model with the specified input shape and number of neurons
model.add(LSTM(n_nuerons,return_sequences=True,input_shape=(X_train.shape[1], X_train.shape[2])))
#Add drop out of 20% to help avoid overfitting the model to the data 
model.add(Dropout(0.1))
model.add(LSTM(n_nuerons,return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(n_nuerons, return_sequences=False))
model.add(Dropout(0.1))
#model.add(Dense(128))
#Define desired output and sigmoid activation function relu 
model.add(Dense(1, activation = 'relu'))
#Define loss metric and optimizer for reweighting in the neural network 
model.compile(loss='mean_squared_error',optimizer='adam')

model.summary()

#Fit the created model to the training sets and define the epochs: number of times the model iterates and reweights 
model.fit(X_train,Y_train,epochs= 100,batch_size=32,verbose=1)


train_predict=model.predict(X_train)
#Predict next day's closing price based on the generated model and test set
test_predict=model.predict(X_test)

# Mean Absolute Percentage Error (MAPE) as a means to show model performance
MAPE = np.mean((np.abs(np.subtract(Y_test, test_predict)/ Y_test))) * 100
print('Mean Absolute Percentage Error (MAPE): ' + str(np.round(MAPE, 2)) + ' %')

#Convert the scaled values back into real prices 
train_predict=scaler_pred.inverse_transform(train_predict)
test_predict=scaler_pred.inverse_transform(test_predict)

#Plot results of the predictions against the true prices 
plt.plot(real_stock_price, color = 'black', label = 'Actual Game Score')
plt.plot(test_predict, color = 'green', label = 'Predicted Predicted')
plt.title('Game Score Prediction')
plt.xlabel('Time')
plt.ylabel('Points Scored')
plt.legend()
plt.show()