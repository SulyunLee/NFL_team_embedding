'''
This script generates feature vectors for each coach in each season.
'''
import pandas as pd
import numpy as np
from POSITION_ASSIGNMENT import *
from tqdm import tqdm

def total_years_NFL_feature(row, df):
    name = row.Name
    year = row.Year
    
    # extract the coach'es previous year seasons
    previous_NFL_seasons = df[(df.Name == name) & (df.Year < year)]
    # count the number of years
    total_years_NFL = previous_NFL_seasons.Year.unique().shape[0]

    return total_years_NFL

def past_seasons_winning_features(row, record_df, win_result_df):
    name = row.Name
    year = row.Year

    # collect last 5 years of game records
    previous_5yr_NFL_seasons = record_df[(record_df.Name == name) & (record_df.Year < year) & (record_df.Year >= year-5)]
    # retrieve the game results (winning percentages)
    previous_5yr_NFL_seasons_results = previous_5yr_NFL_seasons.merge(win_result_df, how="left") 

    # compute best and average winning percentages
    best_prev5yr_winperc = previous_5yr_NFL_seasons_results.Win_Percentage.max()
    avg_prev5yr_winperc = round(previous_5yr_NFL_seasons_results.Win_Percentage.mean(),3)

    return best_prev5yr_winperc, avg_prev5yr_winperc

def position_name_features(row):
    position_id = row.final_position
    hier_num = row.final_hier_num

    # head coach indicator
    if position_id == "HC": 
        HC = 1
    else:
        HC = 0

    # Coordinator indicator
    if hier_num == 2:
        Coord = 1
    else:
        Coord = 0

    return HC, Coord

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

    tqdm.pandas()

    # total win dataset modification
    print("Calculating winning percentages...")
    win_perc = total_win_df.Total_Win / 16
    team_name_modified = total_win_df.Team + " (NFL)"
    total_win_df = total_win_df.assign(Win_Percentage = win_perc.round(3))
    total_win_df = total_win_df.assign(Team = team_name_modified)

    ## Feature 1: Total years in NFL
    print("Feature 1: Total years in NFL")
    total_years_NFL = NFL_record_df.progress_apply(total_years_NFL_feature, \
            args=[NFL_record_df], axis=1)
    NFL_record_df = NFL_record_df.assign(TotalYearsInNFL = total_years_NFL)

    ## Feature 2: Winning percentage during the past 5 years as college or NFL coach
    print("Feature 2: Best winning percentage during the past 5 years in NFL")
    print("Feature 3: Average winning percentage during the past 5 years in NFL")
    best_prev5yr_winperc, avg_prev5yr_winperc = zip(*NFL_record_df.progress_apply(past_seasons_winning_features, args=[NFL_record_df, total_win_df], axis=1))
    NFL_record_df = NFL_record_df.assign(Past5yrsWinningPerc_best = best_prev5yr_winperc)
    NFL_record_df = NFL_record_df.assign(Past5yrsWinningPerc_avg = avg_prev5yr_winperc)

    ## Feature 3: Position names (head coach or coordinator)
    print("Feature 4: Head coach")
    print("Feature 5: Coordinator")
    # distinguish head coaches by position ID because there is interim HC.
    HC, Coord = zip(*NFL_record_df.progress_apply(position_name_features, axis=1))
    NFL_record_df = NFL_record_df.assign(HC=HC)
    NFL_record_df = NFL_record_df.assign(Coord=Coord)

    ## Feature 4: Node embedding (collaboration)



    NFL_record_df.to_csv("../datasets/2002-2019_NFL_Coach_Data_with_features.csv",\
            index=False, encoding="utf-8-sig")
    
