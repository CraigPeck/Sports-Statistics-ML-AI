#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 19:58:22 2021

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



class DataProcessing:
    
    
    
    def __init__(self, train, file):
        self.file = file
        self.train = train
        self.i = int(self.train * len(self.file))
        #self.i = int(self.train * file.size)
        self.stock_train = self.file[0: self.i]
        self.stock_test = self.file[self.i:]
        self.input_train = []
        self.output_train = []
        self.input_test = []
        self.output_test = []

    def gen_train(self, seq_len):

        
        for i in range((len(self.stock_train)//seq_len)*seq_len -seq_len - 1):
            x = np.array(self.stock_train.iloc[i: i + seq_len])
            y = np.array([self.stock_train.iloc[i + seq_len + 1]],np.float64) 
            self.input_train.append(x)
            self.output_train.append(y) 
            self.X_train = np.array(self.input_train) 
            self.Y_train = np.array(self.output_train)
        
        
    def gen_test(self, seq_len):



        for i in range((len(self.stock_test)//seq_len)*seq_len -seq_len - 1):
            x_test = np.array(self.stock_test.iloc[i: i + seq_len])
            y_test = np.array([self.stock_test.iloc[i +seq_len + 1]],np.float64) 
            self.input_test.append(x_test)
            self.output_test.append(y_test) 
            self.X_test = np.array(self.input_test) 
            self.Y_test = np.array(self.output_test)