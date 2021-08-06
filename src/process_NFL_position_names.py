'''
Author: Sulyun Lee
This script processes the cleaned data of "2002-2019_NFL_Coach_Data.csv" in the
format for constructing network of NFL teamts.

'''
import pandas as pd
from tqdm import tqdm
from POSITION_ASSIGNMENT import *

def assign_unique_position_apply(row, position_id_mapping):
    position_list = row.Position_list

    # there is only one position for the coach
    if len(position_list) == 1:
        try:
            # search the position ID and hierarchy number from the dictionary
            position_id, hier_num = position_id_mapping[position_list[0]]
        except:
            # if not found, this coach will be excluded in the graph
            position_id = hier_num = -1
    # multiple positions for one coach
    else:
        # if "head coach" in position_list:
            # position_list.remove("head coach")
        ids = []
        hier_nums = []
        # iterate over each position and find the position ID and hierarchy number
        for position in position_list:
            try:
                position_id, hier_num = position_id_mapping[position]
                ids.append(position_id)
                hier_nums.append(hier_num)
            except:
                continue

        if len(ids) == 0:
            position_id = hier_num = -1
        elif len(ids) == 1:
            position_id = ids[0]
            hier_num = hier_nums[0]
        else:
            # assign the position in the higher hierarchy as the final position
            high_position_idx = hier_nums.index(min(hier_nums))
            position_id = ids[high_position_idx]
            hier_num = hier_nums[high_position_idx]

    return position_id, hier_num

if __name__ == "__main__":
    ################################################################
    # Load datasets
    NFL_coach_record_filename = "../datasets/NFL_Coach_Data.csv"
    #################################################################

    NFL_record_df = pd.read_csv(NFL_coach_record_filename)

    # Generate the list of positions for each coach
    position_lists = NFL_record_df.Position.str.split("[/;-]")
    position_lists = position_lists.apply(lambda x: [e.lower().strip() for e in x])
    NFL_record_df = NFL_record_df.assign(Position_list=position_lists)

    # iterate over each row in NFL record
    # Match the position name to the position IDs and hierarchy umber.
    # If more than one position exists, select the position in the higher hierarchy.
    assigned_unique_positions = NFL_record_df.apply(assign_unique_position_apply, \
            args=[position_id_mapping], axis=1)
    NFL_record_df['final_position'], NFL_record_df['final_hier_num'] = zip(*assigned_unique_positions)

    # check if there are more than one HC in a team
    unique_teams = NFL_record_df.Team.unique()
    unique_years = NFL_record_df.Year.unique()
    for t in unique_teams:
        for y in unique_years:
            season = NFL_record_df[(NFL_record_df.Team == t) & (NFL_record_df.Year == y)]
            season = season.drop_duplicates(subset=["Name", "final_position"])
            season.reset_index(drop=True, inplace=True)
            season_hc = season[season.final_position == "HC"]
            if season_hc.shape[0] > 1:
                print(t, y)

    # drop duplicates
    NFL_record_df = NFL_record_df.drop_duplicates(subset=["Name", "Year", "Team", "final_position", "final_hier_num"])

    # write to a separate csv file
    NFL_record_df.to_csv("../datasets/NFL_Coach_Data_final_position.csv", \
            index=False, encoding="utf-8-sig")



    

