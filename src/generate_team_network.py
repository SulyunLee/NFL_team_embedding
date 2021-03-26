
import pandas as pd
import numpy as np
import networkx as nx

def construct_network(df, year, team):
    '''
    This function constructs team network based on coaches in specific team
    in specific year. It constructs the hierarchical graph (tree).
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

