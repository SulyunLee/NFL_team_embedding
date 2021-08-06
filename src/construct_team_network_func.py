
import pandas as pd
import numpy as np
import networkx as nx
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm

def construct_seasonal_mentorship_network(df, year, team):
    '''
    This function constructs mentorship team network based on coaches in specific team
    in specific year. It constructs the hierarchical graph (tree), where mentor-relationships
    are connected with edges between node pairs.
    '''
    # select list of coaches with specific year and team
    coaches = df[(df.Year == year) & (df.Team == team)]
    coaches.reset_index(drop=True, inplace=True)

    # Construct hierarchical coach graph
    team_G = nx.Graph(year=year, team=team)

    # iterate through each coach and add nodes.
    for idx, row in coaches.iterrows():
        position = row.final_position
        num = row.final_hier_num
        team_G.add_node(row.Name, id=position, hier_num=num)

    # extract coaches with specific position ID
    hc = [x for x,y in team_G.nodes().data() if y['id'] == 'HC'][0]
    oc_list = [x for x,y in team_G.nodes().data() if y['id'] == 'OC']
    dc_list = [x for x,y in team_G.nodes().data() if y['id'] == 'DC']
    sc_list = [x for x,y in team_G.nodes().data() if y['id'] == 'SC']
    o_list = [x for x,y in team_G.nodes().data() if y['id'] == 'O']
    d_list = [x for x,y in team_G.nodes().data() if y['id'] == 'D']
    s_list = [x for x,y in team_G.nodes().data() if y['id'] == 'S']

    edgelist = [] # list to store all edges
    ## Connect position coaches and coordinators
    for s in s_list:
        if len(sc_list) == 0:
            edgelist.append(tuple([s, hc]))
            continue
        for sc in sc_list:
            edgelist.append(tuple([s, sc]))

    for o in o_list:
        if len(oc_list) == 0:
            edgelist.append(tuple([o, hc]))
            continue
        for oc in oc_list:
            edgelist.append(tuple([o, oc]))

    for d in d_list:
        if len(dc_list) == 0:
            edgelist.append(tuple([d, hc]))
            continue
        for dc in dc_list:
            edgelist.append(tuple([d, dc]))

    ## Connect coordinators and head coach
    for sc in sc_list:
        edgelist.append(tuple([sc, hc]))
    
    for oc in oc_list:
        edgelist.append(tuple([oc, hc]))
    
    for dc in dc_list:
        edgelist.append(tuple([dc, hc]))
    
    # add edges from the edgelist
    team_G.add_edges_from(edgelist)

    return team_G

def construct_fullseason_mentorship_network(df, pos_connect):
    '''
    This function constructs a nework using all coaches in the full season.
    One network consists of subgraphs that includes each of the seasons.
    The nodes in this network are in the tuple form which is (Name, Year, Team, final_position). And the edges are the directed edges from the higher to the lower hierarchy. 
    '''
    ### Generate the IDs for each coach record
    # - Key: Unique ID given to each record
    # - Value: tuple of (Name, Year, Team, Position)
    ids = df["ID"]
    records = df[["Name", "Year", "Team", "final_position"]].to_dict('records')
    id_record_dict = dict(zip(ids, records))

    ### Initialize a graph
    entire_team_G = nx.DiGraph()

    ### Generate set of directed edges
    season_records = df[["Year", "Team"]].drop_duplicates().to_dict('records')
    print("There are {} seasons in the input data".format(len(season_records)))
    
    for season in season_records:
        year = season["Year"]
        team = season["Team"]
        coaches = df[(df.Year==year) & (df.Team==team)]
        coaches.reset_index(drop=True, inplace=True)

        # Split the coaches into different positions
        hc = []
        oc_list = []
        dc_list = []
        sc_list = []
        o_list = []
        d_list = []
        s_list = []
        for idx, row in coaches.iterrows():
            if row.final_position == "HC":
                hc.append(row.ID)
            elif row.final_position == "OC":
                oc_list.append(row.ID)
            elif row.final_position == "DC":
                dc_list.append(row.ID)
            elif row.final_position == "SC":
                sc_list.append(row.ID)
            elif row.final_position == "O":
                o_list.append(row.ID)
            elif row.final_position == "D":
                d_list.append(row.ID)
            elif row.final_position == "S":
                s_list.append(row.ID)

        hc = hc[0]

        edgelist = []
        # connect the pairwise position coaches
        if pos_connect == True:
            o_pairs = list(itertools.product(o_list, o_list))
            edgelist.extend(o_pairs)

            d_pairs = list(itertools.product(d_list, d_list))
            edgelist.extend(d_pairs)

            s_pairs = list(itertools.product(s_list, s_list))
            edgelist.extend(s_pairs)

        # connect the node pairs.
        for s in s_list:
            if len(sc_list) == 0:
                edgelist.append(tuple([hc, s]))
                continue
            for sc in sc_list:
                edgelist.append(tuple([sc, s]))

        for o in o_list:
            if len(oc_list) == 0:
                edgelist.append(tuple([hc, o]))
                continue
            for oc in oc_list:
                edgelist.append(tuple([oc, o]))

        for d in d_list:
            if len(dc_list) == 0:
                edgelist.append(tuple([hc, d]))
                continue
            for dc in dc_list:
                edgelist.append(tuple([dc, d]))

        for sc in sc_list:
            edgelist.append(tuple([hc, sc]))
            edgelist.append(tuple([sc, sc])) # add self-loop
        for oc in oc_list:
            edgelist.append(tuple([hc, oc]))
            edgelist.append(tuple([oc, oc])) # add self-loop
        for dc in dc_list:
            edgelist.append(tuple([hc, dc]))
            edgelist.append(tuple([dc, dc])) # add self-loop

        # add self loops of HC
        edgelist.append(tuple([hc, hc]))

        entire_team_G.add_edges_from(edgelist)

    # add the node attributes with the record.
    # The attributes include name, year, team, and final_position.
    nx.set_node_attributes(entire_team_G, id_record_dict)

    return id_record_dict, entire_team_G
    
def construct_cumulative_mentorship_network(initial_g, df, min_year, max_year):
    try:
        teams = df.ServingTeam.unique()
    except:
        teams = df.Team.unique()
    
    cumulative_g = initial_g.copy()

    # iterate through every year and every team to search for collaborations
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
                new_coaches = [coach for coach in coach_list if coach not in cumulative_g.nodes()] # extract coaches not already in the graph
                cumulative_g.add_nodes_from(new_coaches)

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
                cumulative_g.add_edges_from(edgelist)

    return cumulative_g



def construct_cumulative_colleague_network(initial_g, df, min_year, max_year):
    try:
        teams = df.ServingTeam.unique()
    except:
        teams = df.Team.unique()

    cumulative_g = initial_g.copy()

    # iterate through every year and every team to search for collaborations
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
                new_coaches = [coach for coach in coach_list if coach not in cumulative_g.nodes()] # extract coaches not already in the graph
                cumulative_g.add_nodes_from(new_coaches)

                # Add colleague edges to the graph
                edges = itertools.combinations(coach_list,2) # extract all combinations of coach pairs as edges

                new_edges = [edge for edge in edges if edge not in cumulative_g.edges()] # extract edges not already in the graph

                cumulative_g.add_edges_from(new_edges)

    return cumulative_g

def construct_cumulative_directed_colleague_network(initial_g, df, min_year, max_year, selection_prob_dict):
    try:
        teams = df.ServingTeam.unique()
    except:
        teams = df.Team.unique()

    cumulative_g = initial_g.copy()

    # iterate through every year and every team to search for collaborations
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
                new_coaches = [coach for coach in coach_list if coach not in cumulative_g.nodes()] # extract coaches not already in the graph
                cumulative_g.add_nodes_from(new_coaches)

                # generate a dictionary that maps coach name to position hierarchy number
                mapping_dict = dict(zip(records.Name, records.final_hier_num))

                # Add colleague edges to the graph
                edges = itertools.permutations(coach_list,2) # extract all combinations of coach pairs as edges
                # assign edge attributes: downward, peer, upward
                new_edges = []
                prob_list = []
                for edge in edges:
                    coach1_num = mapping_dict[edge[0]]
                    coach2_num = mapping_dict[edge[1]]

                    if coach1_num < coach2_num:
                        direction = 'downward'
                    elif coach1_num == coach2_num:
                        direction = 'peer'
                    elif coach1_num > coach2_num:
                        direction = 'upward'
                    
                    prob = selection_prob_dict[direction]
                    prob_list.append(prob)
                    new_edges.append(edge)

                cumulative_g.add_edges_from(new_edges)
                nx.set_edge_attributes(cumulative_g, name="prob", values=dict(zip(new_edges, prob_list)))

    return cumulative_g

def construct_seasonal_colleague_network(df, min_year, max_year):
    try:
        teams = df.ServingTeam.unique()
    except:
        teams = df.Team.unique()

    seasonal_G = nx.DiGraph()
    ids = df["ID"]
    records = df[["Name", "Year", "Team", "final_position"]].to_dict('records')
    id_record_dict = dict(zip(ids, records))
    for year in tqdm(range(min_year, max_year+1)):
        for team in teams:
            try: 
                records = df[(df.StartYear <= year) & (df.EndYear >= year) & (df.ServingTeam == team)]
            except:
                records = df[(df.Year == year) & (df.Team == team)]

            if (records.shape[0] > 1) and (records.Name.unique().shape[0] > 1):
                coach_list = list(records.ID)
                # all pairs of node directions
                edges = list(itertools.product(coach_list, coach_list))
                seasonal_G.add_edges_from(edges)

        nx.set_node_attributes(seasonal_G, id_record_dict)

    return seasonal_G
    
