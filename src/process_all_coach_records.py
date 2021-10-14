'''
This script process NFL & college football history to be used for learning node embeddings. First, it only extracts NFL & college football history of NFL coaches between 2002-2019. Then, it maps with the hierarchical position names and position numbers. Here, the unqualified positions have the value of -1 for both final_position and final_hier_num.
'''
import pandas as pd
import numpy as np
import networkx as nx
import itertools
from tqdm import tqdm
from POSITION_ASSIGNMENT import *

if __name__ == "__main__":
    #################################################################
    # Load datasets
    NFL_coach_record_filename = "../datasets/NFL_Coach_Data_final_position.csv"
    all_coach_record_filename = "../datasets/all_coach_records_cleaned.csv"

    NFL_record_df = pd.read_csv(NFL_coach_record_filename)
    all_coach_record_df = pd.read_csv(all_coach_record_filename)
    #################################################################

    # exclude NFL records before 2002
    NFL_record_df = NFL_record_df[(NFL_record_df.Year >= 2002) & (NFL_record_df.Year <= 2019)]
    NFL_record_df.reset_index(drop=True, inplace=True)
    # exclude interim head coaches
    NFL_record_df = NFL_record_df[NFL_record_df.final_position != "iHC"]
    NFL_record_df.reset_index(drop=True, inplace=True)
    # exclude coaches with no proper positions
    NFL_record_df = NFL_record_df[(NFL_record_df.final_position != -1) & (NFL_record_df.final_hier_num != -1)]
    NFL_record_df.reset_index(drop=True, inplace=True)

    print("The number of NFL records: {}".format(NFL_record_df.shape[0]))
    print("The number of NFL coaches: {}".format(NFL_record_df.Name.unique().shape[0]))

    ### Clean all coach record dataframe
    # drop coach records with data not known
    all_coach_record_df.dropna(inplace=True)

    # clean position name texts
    position_lists = all_coach_record_df.Position.str.replace('coach', '')
    position_lists = position_lists.str.replace('Coach', '')
    position_lists = position_lists.str.replace('coordinatorinator', 'coordinator')
    position_lists = position_lists.str.split("[/;-]| &")
    position_lists = position_lists.apply(lambda x: [e.lower().strip() for e in x])
    all_coach_record_df = all_coach_record_df.assign(Position_list=position_lists)


    # iterate over each row in record
    # Match the position name to the position IDs and hierarchy umber.
    # If more than one position exists, select the position in the higher hierarchy.
    assigned_unique_positions = all_coach_record_df.apply(assign_unique_position_apply,\
            args=[position_id_mapping], axis=1)
    all_coach_record_df['final_position'], all_coach_record_df['final_hier_num'] = zip(*assigned_unique_positions)

    NFL_coach_before2002_record_df = all_coach_record_df[all_coach_record_df.Name.isin(NFL_record_df.Name.unique())]
    NFL_coach_before2002_record_df.reset_index(drop=True, inplace=True)
    print("The number of NFL coaches' history records before 2002: {}".format(NFL_coach_before2002_record_df.shape[0]))

    NFL_coach_before2002_record_df.to_csv("../datasets/NFL_coach_before2002_records_cleaned.csv", index=False, encoding="utf-8-sig")

    # remove unqualified coaches
    df = NFL_coach_before2002_record_df[NFL_coach_before2002_record_df.final_position != -1]
    df.reset_index(drop=True, inplace=True)
    
    ### Exclude coaches who are not in NFL data

    all_record_year_min = int(df.StartYear.min())
    min_year = all_record_year_min
    max_year = 2001

    # construct cumulative mentorship network
    try:
        teams = df.ServingTeam.unique()
    except:
        teams = df.Team.unique()

    g = nx.Graph()

    # iterate through every year and every team to search for collaborations
    # teams = ["florida state"]

    for year in tqdm(range(min_year, max_year+1)):
        for team in teams:
            try:
                # coaches who worked together
                records = df[(df.StartYear <= year) & (df.EndYear >= year) & (df.ServingTeam == team)]
            except:
                records = df[(df.Year == year) & (df.Team == team)]

            if (records.shape[0] > 1) and (records.Name.unique().shape[0] > 1):
                # Add coach names to the graph
                coach_list = list(records.Name)
                new_coaches = [coach for coach in coach_list if coach not in g.nodes()] # extract coaches not already in the graph
                g.add_nodes_from(new_coaches)

                hc = records[records.final_position == "HC"].Name.tolist()
                if len(hc) != 0:
                    hc = hc[:1]
                oc_list = records[records.final_position == "OC"].Name.tolist()
                dc_list = records[records.final_position == "DC"].Name.tolist()
                sc_list = records[records.final_position == "SC"].Name.tolist()
                o_list = records[records.final_position == "O"].Name.tolist()
                d_list = records[records.final_position == "D"].Name.tolist()
                s_list = records[records.final_position == "S"].Name.tolist()

                edgelist = []
                # connect position coaches and coordinators.
                # if coordinator not exist, connect position coaches with head coach.
                if len(o_list) > 0:
                    if len(oc_list) > 0:
                        edgelist.extend(list(itertools.product(oc_list, o_list)))
                    elif len(hc) > 0:
                        edgelist.extend(list(itertools.product(hc, o_list)))

                if len(d_list) > 0:
                    if len(dc_list) > 0:
                        edgelist.extend(list(itertools.product(dc_list, d_list)))
                    elif len(hc) > 0:
                        edgelist.extend(list(itertools.product(hc, d_list)))

                if len(s_list) > 0:
                    if len(sc_list) > 0:
                        edgelist.extend(list(itertools.product(sc_list, s_list)))
                    elif len(hc) > 0:
                        edgelist.extend(list(itertools.product(hc, s_list)))

                # connect coordinators and head coach
                if len(hc) > 0:
                    if len(oc_list) > 0:
                        edgelist.extend(list(itertools.product(hc, oc_list)))

                if len(hc) > 0:
                    if len(dc_list) > 0:
                        edgelist.extend(list(itertools.product(hc, dc_list)))

                if len(hc) > 0:
                    if len(sc_list) > 0:
                        edgelist.extend(list(itertools.product(hc, sc_list)))

                # add edges to the network
                g.add_edges_from(edgelist)





    
