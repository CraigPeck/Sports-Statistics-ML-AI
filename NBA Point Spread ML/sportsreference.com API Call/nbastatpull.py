#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 20:53:31 2021

@author: craigpeck
"""

import pandas as pd
import numpy as np
from sportsreference.nba.teams import Teams
from sportsreference.nba.schedule import *
from sportsreference.nba.boxscore import *
from sportsreference.nba.roster import *


Points_Scored = []
Field_Goals = []
FG_Perc = []
Three_P = []
Three_P_Perc = []
OReb = []
Assists = []
Steals = []
ORank = []
DRank = []
Pace = []
Team_Players = []
Average_Offensive_Rating = []
Team_Stats_Total = []
Average_Offensive_Rating_Total = []



for team in Teams():

    Team_schedule = Schedule(team.abbreviation)
    Houston = Roster(team.abbreviation)

    
    for game in Team_schedule:
        
        if game.boxscore._home_name == team.name:
            FG = game.points_scored
            fieldgoals = game.boxscore.home_field_goals 
            fg_perc = game.boxscore.home_field_goal_percentage
            threeP = game.boxscore.home_three_point_field_goals
            threeP_perc = game.boxscore.home_three_point_field_goal_percentage
            orb = game.boxscore.home_offensive_rebounds
            assits = game.boxscore.home_assists
            steals = game.boxscore.home_steals
            orank = game.boxscore.home_offensive_rating
            drank = game.boscore.away_defensive_rating
            #players = game.boxscore.home_players
           
        else:
            FG = game.points_scored
            fieldgoals = game.boxscore.away_field_goals 
            fg_perc = game.boxscore.away_field_goal_percentage
            threeP = game.boxscore.away_three_point_field_goals
            threeP_perc = game.boxscore.away_three_point_field_goal_percentage
            orb = game.boxscore.away_offensive_rebounds
            assists = game.boxscore.away_assists
            steals = game.boxscore.away_steals
            orank = game.boxscore.away_offensive_rating
            drank = game.boxscore.home_defensive_rating
            #players = game.boxscore.away_players 
    
            
        Field_Goals.append(fieldgoals)
        Points_Scored.append(FG)
        FG_Perc.append(fg_perc)
        Three_P.append(threeP)
        Three_P_Perc.append(threeP_perc)
        OReb.append(orb)
        Assists.append(assists)
        Steals.append(steals)
        ORank.append(orank)
        DRank.append(drank)
        Pace = game.boxscore.pace
        #Team_Players.append(players)
        
    Team_Stats = pd.DataFrame({'Points Scored':Points_Scored, 
                               'FG': Field_Goals, 
                               'FG%': FG_Perc, 
                               '3P': Three_P,
                               '3P%': Three_P_Perc,
                               'ORB': OReb,
                               'Assists':Assists,
                               'Steals': Steals,
                               'OffRank':ORank,
                               'OppDefRank': DRank,
                               'Pace': Pace})
    
    
    
    
    
    for game in Team_schedule:
      
        
      
        game_data = game.boxscore
       
        
        
        if game.points_scored is not None:
            
            if game.location == 'Home':
            
                home_df = game_data.home_players[0].dataframe
          
                for player in game_data.home_players[1:]:
                    home_df = pd.concat([home_df, player.dataframe], axis = 0)
            
                home_df['name'] = [x.name for x in game_data.home_players]
                home_df.set_index('name', inplace = True)
                
            
                highest_minutes_played = home_df.sort_values(by = 'minutes_played', ascending = False)
                starters = highest_minutes_played.head(5)
                collective_offensive_rating = starters['minutes_played'].mean()
            
            else:
                away_df = game_data.away_players[0].dataframe
          
                for player in game_data.away_players[1:]:
                    away_df = pd.concat([away_df, player.dataframe], axis = 0)
            
                away_df['name'] = [x.name for x in game_data.away_players]
                away_df.set_index('name', inplace = True)
                
            
                highest_minutes_played = away_df.sort_values(by = 'minutes_played', ascending = False)
                starters = highest_minutes_played.head(5)
                collective_offensive_rating = starters['minutes_played'].mean() 
                
        else:
            break
        
        Average_Offensive_Rating.append(collective_offensive_rating)
    
    Average_Offensive_Rating_TST = pd.DataFrame( Average_Offensive_Rating)   
 
Team_Stats_Total.append(Team_Stats)
Team_Stats_Total = pd.DataFrame(Team_Stats_Total[0])
Average_Offensive_Rating_Total.append(Average_Offensive_Rating_TST)
Average_Offensive_Rating_Total = pd.DataFrame(Average_Offensive_Rating_Total[0])


Team_Stats_Total.to_csv('nbateamstatsdataFull.csv')
Average_Offensive_Rating_Total.to_csv('nbaavoffratdataFull.csv')

  