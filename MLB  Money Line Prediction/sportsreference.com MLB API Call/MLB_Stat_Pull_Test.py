#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 22:04:06 2021

@author: craigpeck
"""

import pandas as pd
import numpy as np
from sportsreference.mlb.teams import *
from sportsreference.mlb.schedule import *
from sportsreference.mlb.boxscore import *
from sportsreference.mlb.roster import *
from sportsreference.mlb.roster import Player
import mlbgame
import mlbgame.data
import mlbgame.object
from datetime import datetime


Runs_Scored = []
At_Bats = []
Hits = []
RBI = []
Earned_Runs = []
Bases_on_Balls = []
Strikeouts = []
Batting_Average = []
On_Base_Percentage = []
Slugging_Percentage = []
Pitches = []
Pitches_Faced = []
Pitches_Pitched = []
Strikes_Earned = []
Strikes_Given = []
Off_Win_Prob_Contribution = []
Pressure_Pitcher_Faced = []
Off_Win_Prob_Sub = []
Base_Out_Runs = []
Assists = []
Home_Runs_GivenUp = []
Grounded_Balls_Allowed = []
Fly_Balls_Allowed = []
Line_Drives_Allowed = []
Pitcher_Win_Contribution = []
Winner = []
Scoreboard = []
Team_Rank = []
Last_Ten_Games = []
Pitcher_Hits_Allowed_Per_Batter = []
Pitcher_Runs_Allowed_Per_Batter = []
Pitcher_ERA_Per_Batter = []
Pitcher_Home_Runs_Allowed_Per_Batter = []
Pitcher_Strikeouts_Per_Batter = []
Pitcher_Name = []




team_abbrev = 'NYM'
team_name = 'Mets'
Team_schedule = Schedule(team_abbrev, year = '2020')



for game in Team_schedule:
    
    date = game.datetime 
    Y, M, D = date.year, date.month, date.day
    test_location = game.location

    
    if game.boxscore.winning_abbr == team_abbrev:
        winner = 1
    else:
        winner = 0
    
    if game.location == 'Home':
        
        try:
        
            scoreboard = mlbgame.game.scoreboard(Y, M, D, home = team_name)
         
            key_list = list(scoreboard.keys())[0]
            
         
          
            
            if game.boxscore.winning_abbr == team_abbrev:
                pitcher = scoreboard[key_list]['w_pitcher']
                
                
            
            else:
                pitcher = scoreboard[key_list]['l_pitcher']
              
            
                    
            pitcher_id1 = pitcher.split() 
            pitcher_id1 = pitcher_id1[1]
            pitcher_id1 = pitcher_id1[:5]
            pitcher_id2 = pitcher[:2]
            pitcher_id3 = "01"
            pitcher_id31 = "02"
            
            try:
                pitcher_id4 = [pitcher_id1, pitcher_id2, pitcher_id3]
                p = ""
                pitcher_id = p.join(pitcher_id4).lower() 
                pitcher_info = Player(pitcher_id)
            except TypeError:
                print("wrong id")
                
            try:
                pitcher_id4 = [pitcher_id1, pitcher_id2, pitcher_id31]
                p = ""
                pitcher_id = p.join(pitcher_id4).lower()
                pitcher_info = Player(pitcher_id)
            except TypeError:
                print("wrong id")
                
            pitcher_batters_faced = pitcher_info.batters_faced
            
            if pitcher_info.hits_allowed is not None:
                pitcher_hits_allowed_per_batter = pitcher_info.hits_allowed/pitcher_batters_faced
            else:   
                pitcher_hits_allowed_per_batter = 0
                
            if pitcher_info.runs_allowed is not None:
                pitcher_runs_allowed_per_batter = pitcher_info.runs_allowed/pitcher_batters_faced
            else: 
                pitcher_runs_allowed_per_batter = 0   
                
            if pitcher_info.earned_runs_allowed is not None:
                pitcher_era_per_batter = pitcher_info.earned_runs_allowed/pitcher_batters_faced
            else:
                pitcher_era_per_batter = 0
                
            if pitcher_info.home_runs_allowed is not None:
                
                pitcher_home_runs_allowed_per_batter = pitcher_info.home_runs_allowed/pitcher_batters_faced
            else:
                pitcher_home_runs_allowed_per_batter = 0
            
            if pitcher_info.strikeouts is not None:
            
                pitcher_strikeouts_per_batter = pitcher_info.strikeouts/pitcher_batters_faced
            
            else:
                pitcher_strikeouts_per_batter = 0
                
        except TypeError:
            print("No Scoreboard")
        
        team_rank = game.rank
        runs_Scored = game.boxscore.home_runs
        at_Bats = game.boxscore.home_at_bats 
        hits = game.boxscore.home_hits
        rBI = game.boxscore.home_rbi
        earned_Runs = game.boxscore.home_earned_runs
        bases_on_Balls = game.boxscore.home_bases_on_balls
        strikeouts = game.boxscore.home_strikeouts
        batting_Average = game.boxscore.home_batting_average
        on_Base_Percentage = game.boxscore.home_on_base_percentage
        slugging_Percentage = game.boxscore.home_slugging_percentage
        pitches_Faced = game.boxscore.home_pitches
        pitches_Pitched = game.boxscore.away_pitches
        strikes_Earned = game.boxscore.home_strikes
        strikes_Given = game.boxscore.away_strikes
        off_Win_Prob_Contribution = game.boxscore.home_win_probability_for_offensive_player
        pressure_Pitcher_Faced = game.boxscore.home_average_leverage_index
        off_Win_Prob_Sub = game.boxscore.home_win_probability_subtracted
        base_Out_Runs = game.boxscore.home_base_out_runs_added
        assists = game.boxscore.home_assists
        home_Runs_GivenUp = game.boxscore.home_home_runs
        grounded_Balls_Allowed = game.boxscore.home_grounded_balls
        fly_Balls_Allowed = game.boxscore.home_fly_balls
        line_Drives_Allowed = game.boxscore.home_line_drives
        pitcher_Win_Contribution = game.boxscore.home_win_probability_by_pitcher
        
        
        
       
    elif game.location == 'Away':
        
        try:
        
            scoreboard = mlbgame.game.scoreboard(Y, M, D, away = team_name)
            
           
            key_list = list(scoreboard.keys())[0]
            
            if game.boxscore.winning_abbr == team_abbrev:
                pitcher = scoreboard[key_list]['w_pitcher']
            else:
                pitcher = scoreboard[key_list]['l_pitcher']
                
                    
            pitcher_id1 = pitcher.split() 
            pitcher_id1 = pitcher_id1[1]
            pitcher_id1 = pitcher_id1[:5]
            pitcher_id2 = pitcher[:2]
            pitcher_id3 = "01"
            pitcher_id31 = "02"
            
            try:
                pitcher_id4 = [pitcher_id1, pitcher_id2, pitcher_id3]
                p = ""
                pitcher_id = p.join(pitcher_id4).lower() 
                pitcher_info = Player(pitcher_id)
            except TypeError:
                print("wrong id")
                
            try:
                pitcher_id4 = [pitcher_id1, pitcher_id2, pitcher_id31]
                p = ""
                pitcher_id = p.join(pitcher_id4).lower()
                pitcher_info = Player(pitcher_id)
            except TypeError:
                print("wrong id")    
    
            pitcher_batters_faced = pitcher_info.batters_faced
            
            if pitcher_info.hits_allowed is not None:
                pitcher_hits_allowed_per_batter = pitcher_info.hits_allowed/pitcher_batters_faced
            else:   
                pitcher_hits_allowed_per_batter = 0
                
            if pitcher_info.runs_allowed is not None:
                pitcher_runs_allowed_per_batter = pitcher_info.runs_allowed/pitcher_batters_faced
            else: 
                pitcher_runs_allowed_per_batter = 0   
                
            if pitcher_info.earned_runs_allowed is not None:
                pitcher_era_per_batter = pitcher_info.earned_runs_allowed/pitcher_batters_faced
            else:
                pitcher_era_per_batter = 0
                
            if pitcher_info.home_runs_allowed is not None:
                
                pitcher_home_runs_allowed_per_batter = pitcher_info.home_runs_allowed/pitcher_batters_faced
            else:
                pitcher_home_runs_allowed_per_batter = 0
            
            if pitcher_info.strikeouts is not None:
            
                pitcher_strikeouts_per_batter = pitcher_info.strikeouts/pitcher_batters_faced
            
            else:
                pitcher_strikeouts_per_batter = 0   
                
        except TypeError:
            print("No Scoreboard")
            
        team_rank = game.rank
        runs_Scored = game.boxscore.away_runs
        at_Bats = game.boxscore.away_at_bats 
        hits = game.boxscore.away_hits
        rBI = game.boxscore.away_rbi
        earned_Runs = game.boxscore.away_earned_runs
        bases_on_Balls = game.boxscore.away_bases_on_balls
        strikeouts = game.boxscore.away_strikeouts
        batting_Average = game.boxscore.away_batting_average
        on_Base_Percentage = game.boxscore.away_on_base_percentage
        slugging_Percentage = game.boxscore.away_slugging_percentage
        pitches_Faced = game.boxscore.away_pitches
        pitches_Pitched = game.boxscore.home_pitches
        strikes_Earned = game.boxscore.away_strikes
        strikes_Given = game.boxscore.home_strikes
        off_Win_Prob_Contribution = game.boxscore.away_win_probability_for_offensive_player
        pressure_Pitcher_Faced = game.boxscore.away_average_leverage_index
        off_Win_Prob_Sub = game.boxscore.away_win_probability_subtracted
        base_Out_Runs = game.boxscore.away_base_out_runs_added
        assists = game.boxscore.away_assists
        home_Runs_GivenUp = game.boxscore.away_home_runs
        grounded_Balls_Allowed = game.boxscore.away_grounded_balls
        fly_Balls_Allowed = game.boxscore.away_fly_balls
        line_Drives_Allowed = game.boxscore.away_line_drives
        pitcher_Win_Contribution = game.boxscore.away_win_probability_by_pitcher


 
        
    Winner.append(winner)
    Runs_Scored.append(runs_Scored)
    At_Bats.append(at_Bats)
    Hits.append(hits)
    RBI.append(rBI)
    Earned_Runs.append(earned_Runs)
    Bases_on_Balls.append(bases_on_Balls)
    Strikeouts.append(strikeouts)
    Batting_Average.append(batting_Average)
    On_Base_Percentage.append(on_Base_Percentage)
    Slugging_Percentage.append(slugging_Percentage)
    Pitches_Faced.append(pitches_Faced)
    Pitches_Pitched.append(pitches_Pitched)
    Strikes_Earned.append(strikes_Earned)
    Strikes_Given.append(strikes_Given)
    Off_Win_Prob_Contribution.append(off_Win_Prob_Contribution)
    Pressure_Pitcher_Faced.append(pressure_Pitcher_Faced)
    Off_Win_Prob_Sub.append(off_Win_Prob_Sub)
    Base_Out_Runs.append(base_Out_Runs)
    Assists.append(assists) 
    Home_Runs_GivenUp.append(home_Runs_GivenUp)
    Grounded_Balls_Allowed.append(grounded_Balls_Allowed)
    Fly_Balls_Allowed.append(fly_Balls_Allowed)
    Line_Drives_Allowed.append(line_Drives_Allowed)
    Pitcher_Win_Contribution.append(pitcher_Win_Contribution)
    Scoreboard.append(scoreboard)
    Team_Rank.append(team_rank)
    Pitcher_Hits_Allowed_Per_Batter.append(pitcher_hits_allowed_per_batter)
    Pitcher_Runs_Allowed_Per_Batter.append(pitcher_runs_allowed_per_batter)
    Pitcher_ERA_Per_Batter.append(pitcher_era_per_batter)
    Pitcher_Home_Runs_Allowed_Per_Batter.append(pitcher_home_runs_allowed_per_batter)
    Pitcher_Strikeouts_Per_Batter.append(pitcher_strikeouts_per_batter)
    Pitcher_Name.append(pitcher)




   
    
Team_Stats = pd.DataFrame({'Winner': Winner,
                           'Runs':Runs_Scored, 
                           'At_Bats': At_Bats, 
                           'Hits': Hits, 
                           'RBI': RBI,
                           'Earned_Runs': Earned_Runs,
                           'Bases_on_Balls': Bases_on_Balls,
                           'Strikeouts':Strikeouts,
                           'Batting_Average': Batting_Average,
                           'On_Base_Percentage':On_Base_Percentage,
                           'Slugging_Percentage': Slugging_Percentage,
                           'Pitches_Faced': Pitches_Faced,
                           'Pitches_Pitched': Pitches_Pitched,
                           'Strikes_Earned': Strikes_Earned,
                           'Strikes_Given': Strikes_Given,
                           'Off_Win_Prob_Contribution': Off_Win_Prob_Contribution,
                           'Pressure_Pitcher_Faced': Pressure_Pitcher_Faced,
                           'Off_Win_Prob_Sub': Off_Win_Prob_Sub,
                           'Base_Out_Runs': Base_Out_Runs,
                           'Assists': Assists,
                           'Home_Runs_Givenup': Home_Runs_GivenUp,
                           'Grounded_Balls_Allowed': Grounded_Balls_Allowed,
                           'Fly_Balls_Allowed': Fly_Balls_Allowed,
                           'Line_Drives_Allowed': Line_Drives_Allowed,
                           'Pitcher_Win_Contribution': Pitcher_Win_Contribution,
                           'Pitcher_Hits_Allowed_Per_Batter': Pitcher_Hits_Allowed_Per_Batter,
                           'Pitcher_Runs_Allowed_Per_Batter': Pitcher_Runs_Allowed_Per_Batter,
                           'Pitcher_ERA_Per_Batter': Pitcher_ERA_Per_Batter,
                           'Pitcher_Home_Runs_Allowed_Per_Batter': Pitcher_Home_Runs_Allowed_Per_Batter,
                           'Pitcher_Strikeouts_Per_Batter': Pitcher_Strikeouts_Per_Batter})


Team_Stats.to_csv('mlbteamstatstestset1.csv')