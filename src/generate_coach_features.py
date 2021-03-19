'''
This script generates feature vectors for each coach in each season.
'''
import pandas as pd
import numpy as np
from POSITION_ASSIGNMENT import *

if __name__ == "__main__":
    #################################################################
    # Load datasets
    coach_profile_filename = "../datasets/all_coach_profile.csv"
    nonNFL_coaching_record_filename = "../datasets/nonNFL_coaching_records.csv"
    NFL_coach_record_filename = "../datasets/2002-2019_NFL_Coach_Data_final_position.csv"
    playoff_filename = "../datasets/Playoff.csv"
    total_win_filename = "../datasets/Total_Win.csv"

    coach_profile_df = pd.read_csv(coach_profile_filename)
    nonNFL_record_df = pd.read_csv(nonNFL_coaching_record_filename)
    NFL_record_df = pd.read_csv(NFL_coach_record_filename)
    playoff_df = pd.read_csv(playoff_filename)
    total_win_df = pd.read_csv(total_win_filename)
    #################################################################

    row = NFL_record_df.iloc[100]
    name = row.Name
    team = row.Team
    year = row.Year
    position_id = row.final_position
    hier_num = row.final_hier_num

    # Total years in NFL
    previous_NFL_seasons = NFL_record_df[(NFL_record_df.Name == name) & (NFL_record_df.Year < year)]
    total_years_NFL = previous_NFL_seasons.Year.unique().shape[0]

    # Winning percentage during the past 5 years as college or NFL coach

    # HC indicator
    # distinguish head coaches by position ID because there is interim HC.
    if position_id == "HC": 
        HC = 1
    else:
        HC = 0

    # Coordinator indicator
    if position_hier == 2:
        Coord = 1
    else:
        Coord = 0
