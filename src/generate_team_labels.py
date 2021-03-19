'''
This script generates labels for each team in each season.
Label: failure of team.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
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

    # calculate winning percentage
    print("Calculating winning percentages...")
    win_perc = total_win_df.Total_Win / 16
    total_win_df = total_win_df.assign(Win_Percentage = win_perc.round(3))

    # Calculate the winning percentage
    # If winning percentage less  than 50 or head coach fired in the middle of season, the team is a failure team
    years = NFL_record_df.Year.unique()
    teams = NFL_record_df.Team.unique()
    failure_label_df = total_win_df[total_win_df.Year.isin(years)]
    failure_label_df.reset_index(drop=True, inplace=True)
    failure_label_arr = np.zeros((failure_label_df.shape[0]))

    
    # This code block checks the teams with more than one coach (fired head coach)
    # or winning percentage below 50%.
    print("Generating failure labels...")
    two_hc_dict = dict() # store teams with two head coaches
    two_hc_count = 0
    for idx, row in failure_label_df.iterrows():
        year = row.Year
        team = row.Team
        win_perc = row.Win_Percentage
        season_coaches = NFL_record_df[(NFL_record_df.Year == year) & (NFL_record_df.Team == "{} (NFL)".format(team))]
        if (win_perc < 0.5) or (season_coaches[season_coaches.final_hier_num == 1].shape[0] > 1):
            failure_label_arr[idx] = 1


    # append team success label to dataframe
    failure_label_df = failure_label_df.assign(failure=failure_label_arr)
    failure_label_df.to_csv("../datasets/team_labels.csv", index=False, encoding="utf-8-sig")
    print(failure_label_df.failure.value_counts())

    # plot the distribution of winning percentages
    print("Plotting distribution of winning percentages")
    fig, ax = plt.subplots(figsize=(10,8))
    ax.hist(failure_label_df.Win_Percentage * 100, bins=10)
    ax.set_xlabel("Winning percentage")
    ax.set_ylabel("Count")
    plt.savefig("../plots/win_percentage_distr.png")
    plt.close()

    # Check if failed teams ever got to the playoff
    print("Teams that failed but went to the playoff: ")
    failed_seasons = failure_label_df[failure_label_df.failure == 1]
    failed_seasons.reset_index(drop=True, inplace=True)
    for idx, row in failed_seasons.iterrows():
        year = row.Year
        team = row.Team
        win_perc = row.Win_Percentage
        playoff_result = playoff_df[(playoff_df.Year == year) & (playoff_df.Team == team)]
        if playoff_result.shape[0] != 0:
            print("Year: {}, Team: {}, Winning %: {}, Playoffstage: {}".format(year, team, win_perc, playoff_result.iloc[0].StageAchieved))



