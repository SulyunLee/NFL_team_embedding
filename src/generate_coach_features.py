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
    previous_NFL_seasons = df[(df.Name == name) & (df.NFL == 1)]
    if previous_NFL_seasons.shape[0] == 0:
        total_years_NFL = 0
    else:
        year_list = []
        for i, row in previous_NFL_seasons.iterrows():
            year_list.extend(range(int(row.StartYear), int(row.EndYear)+1))

        # only select the years before the current year
        year_list = [num for num in year_list if num < year]

        # count the number of years
        total_years_NFL = len(set(year_list))

    return total_years_NFL

def past_seasons_winning_features(row, record_df, win_result_df):
    name = row.Name
    year = row.Year

    # collect last 5 years of game records
    previous_5yr_NFL_seasons = record_df[(record_df.Name == name) & (record_df.Year < year) & (record_df.Year >= year-5)]
    if previous_5yr_NFL_seasons.shape[0] == 0:
        best_prev5yr_winperc, avg_prev5yr_winperc = 0, 0
    else:
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
    all_coaching_record_filename = "../datasets/all_coach_records_cleaned.csv"
    NFL_coach_record_filename = "../datasets/NFL_Coach_Data_final_position.csv"
    total_win_filename = "../datasets/Total_Win.csv"
    cumulative_node_embedding_filename = "../datasets/cumulative_colleague_G_node_embedding_all_df.csv"

    all_record_df = pd.read_csv(all_coaching_record_filename)
    NFL_record_df = pd.read_csv(NFL_coach_record_filename)
    total_win_df = pd.read_csv(total_win_filename)
    cumulative_node_embedding_df = pd.read_csv(cumulative_node_embedding_filename)
    #################################################################

    # exclude interim head coaches
    NFL_coach_instances = NFL_record_df[NFL_record_df.final_position != "iHC"]
    NFL_coach_instances.reset_index(drop=True, inplace=True)

    # exclude coaches with no proper positions
    NFL_coach_instances = NFL_coach_instances[(NFL_coach_instances.final_position != -1) & (NFL_coach_instances.final_hier_num != -1)]
    NFL_coach_instances.reset_index(drop=True, inplace=True)

    # Include only 2002-2019 seasons
    NFL_coach_instances = NFL_coach_instances[(NFL_coach_instances.Year >= 2002) & (NFL_coach_instances.Year <= 2019)]
    NFL_coach_instances.reset_index(drop=True, inplace=True)

    print("Total number of NFL coach instances: {}".format(NFL_coach_instances.shape[0]))

    tqdm.pandas()

    # total win dataset modification
    print("Calculating winning percentages...")
    win_perc = total_win_df.Total_Win / 16
    team_name_modified = total_win_df.Team + " (NFL)"
    total_win_df = total_win_df.assign(Win_Percentage = win_perc.round(3))
    total_win_df = total_win_df.assign(Team = team_name_modified)

    ## Feature 1: Total years in NFL - 
    print("Feature 1: Total years in NFL")
    total_years_NFL = NFL_coach_instances.progress_apply(total_years_NFL_feature, \
            args=[all_record_df], axis=1)
    NFL_coach_instances = NFL_coach_instances.assign(TotalYearsInNFL = total_years_NFL)

    ## Feature 2: Winning percentage during the past 5 years as college or NFL coach
    print("Feature 2: Best winning percentage during the past 5 years in NFL")
    print("Feature 3: Average winning percentage during the past 5 years in NFL")
    best_prev5yr_winperc, avg_prev5yr_winperc = zip(*NFL_coach_instances.progress_apply(past_seasons_winning_features, args=[NFL_record_df, total_win_df], axis=1))
    NFL_coach_instances = NFL_coach_instances.assign(Past5yrsWinningPerc_best = best_prev5yr_winperc)
    NFL_coach_instances = NFL_coach_instances.assign(Past5yrsWinningPerc_avg = avg_prev5yr_winperc)

    ## Feature 3: Position names (head coach or coordinator)
    print("Feature 4: Head coach")
    print("Feature 5: Coordinator")
    # distinguish head coaches by position ID because there is interim HC.
    HC, Coord = zip(*NFL_coach_instances.progress_apply(position_name_features, axis=1))
    NFL_coach_instances = NFL_coach_instances.assign(HC=HC)
    NFL_coach_instances = NFL_coach_instances.assign(Coord=Coord)

    ## Feature 4: Node embedding (collaboration)
    # Node embedding that contains the collaboration information during the past seasons.
    # node embedding for predicting year t = average of embedding upto year t-2 
    # and yearly embedding at year t-1
    print("Feature 6: Collaboration features (node embedding)")
    cumul_emb_columns = cumulative_node_embedding_df.columns[cumulative_node_embedding_df.columns.str.contains("cumul_emb")].tolist()
    coach_emb_features = np.zeros((NFL_coach_instances.shape[0], len(cumul_emb_columns)))
    no_node_emb_arr = np.zeros((NFL_coach_instances.shape[0])).astype(int) # if there is no node embedding for the corresponding coach
    for idx, row in tqdm(NFL_coach_instances.iterrows(), total=NFL_coach_instances.shape[0]):
        year = row.Year
        name = row.Name
            
        cumulative_emb = cumulative_node_embedding_df[(cumulative_node_embedding_df.Name==name) & (cumulative_node_embedding_df.Year==year)] 
        cumulative_emb = np.array(cumulative_emb[cumul_emb_columns])

        if cumulative_emb.shape[0] != 0:
            coach_emb_features[idx,:] = cumulative_emb
        else:
            no_node_emb_arr[idx] = 1

    NFL_coach_instances = pd.concat([NFL_coach_instances, pd.DataFrame(coach_emb_features,\
            index=NFL_coach_instances.index, columns=cumul_emb_columns)], axis=1)

    NFL_coach_instances = NFL_coach_instances.assign(no_node_emb=no_node_emb_arr)

    # check nodes with no embedding learned from Deepwalk on cumulative network until previous years.
    for hier_num in range(1, 4):
        # print("Hier_num: {}".format(hier_num))
        for year in range(2002, 2020):
            num_no_emb = NFL_coach_instances[(NFL_coach_instances.final_hier_num==hier_num) & (NFL_coach_instances.Year==year)].no_node_emb.sum()
            # print(num_no_emb)

    no_node_emb_instances = NFL_coach_instances[NFL_coach_instances.no_node_emb==1]
    no_node_emb_instances.reset_index(drop=True, inplace=True)
    for idx, coach in no_node_emb_instances.iterrows():
        year = coach.Year
        college_coaching_record = all_record_df[(all_record_df.StartYear<year) & (all_record_df.Name==coach.Name)]

    NFL_coach_instances.to_csv("../datasets/NFL_Coach_Data_with_features.csv",\
            index=False, encoding="utf-8-sig")
    
