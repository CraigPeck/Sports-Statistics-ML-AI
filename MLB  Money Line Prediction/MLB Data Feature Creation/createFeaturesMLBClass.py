#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:48:49 2021

@author: craigpeck
"""


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from pandas import datetime
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from itertools import chain



    
def createFeaturesClassMLB(file):
    
        
        new_df = pd.DataFrame(file)


    
    # # Exponential Moving Averages (EMAS) - different periods
        new_df['Runs_Scored EMA3'] = new_df['Runs_Scored'].ewm(span=3, adjust=False).mean() 
        new_df['At_Bats EMA3'] = new_df['At_Bats'].ewm(span=3, adjust=False).mean()
        new_df['Hits EMA3'] = new_df['Hits'].ewm(span=3, adjust=False).mean()
        new_df['RBI EMA3'] = new_df['RBI'].ewm(span=3, adjust=False).mean()
        new_df['Earned_Runs EMA3'] = new_df['Earned_Runs'].ewm(span=3, adjust=False).mean()
        new_df['Bases_on_Balls EMA3'] = new_df['Bases_on_Balls'].ewm(span=3, adjust=False).mean()
        new_df['Strikeouts EMA3'] = new_df['Strikeouts'].ewm(span=3, adjust=False).mean()
        new_df['Batting_Average EMA3'] = new_df['Batting_Average'].ewm(span=3, adjust=False).mean()
        new_df['On_Base_Percentage EMA3'] = new_df['On_Base_Percentage'].ewm(span=3, adjust=False).mean()
        new_df['Slugging_Percentage EMA3'] = new_df['Slugging_Percentage'].ewm(span=3, adjust=False).mean()
        new_df['Pitches_Faced EMA3'] = new_df['Pitches_Faced'].ewm(span=3, adjust=False).mean()
        new_df['Strikes_Earned EMA3'] = new_df['Strikes_Earned'].ewm(span=3, adjust=False).mean()
        new_df['Strikes_Given EMA3'] = new_df['Strikes_Given'].ewm(span=3, adjust=False).mean()
        new_df['Off_Win_Prob_Contribution EMA3'] = new_df['Off_Win_Prob_Contribution'].ewm(span=3, adjust=False).mean()
        new_df['Pressure_Pitcher_Faced EMA3'] = new_df['Pressure_Pitcher_Faced'].ewm(span=3, adjust=False).mean()
        new_df['Base_Out_Runs EMA3'] = new_df['Base_Out_Runs'].ewm(span=3, adjust=False).mean()
        new_df['Assists EMA3'] = new_df['Assists'].ewm(span=3, adjust=False).mean()
        new_df['Home_Runs_Givenup EMA3'] = new_df['Home_Runs_Givenup'].ewm(span=3, adjust=False).mean()
        new_df['Grounded_Balls_Allowed EMA3'] = new_df['Grounded_Balls_Allowed'].ewm(span=3, adjust=False).mean()
        new_df['Fly_Balls_Allowed EMA3'] = new_df['Fly_Balls_Allowed'].ewm(span=3, adjust=False).mean()
        new_df['Line_Drives_Allowed EMA3'] = new_df['Line_Drives_Allowed'].ewm(span=3, adjust=False).mean()
        new_df['Pitcher_Win_Contribution EMA3'] = new_df['Pitcher_Win_Contribution'].ewm(span=3, adjust=False).mean()
        new_df['Pitches_Pitched EMA3'] = new_df['Pitches_Pitched'].ewm(span=3, adjust=False).mean()
        new_df['Runs_Allowed EMA3'] = new_df['Runs_Allowed'].ewm(span=3, adjust=False).mean()
        
        
    
        return new_df