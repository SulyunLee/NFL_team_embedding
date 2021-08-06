'''
This script generates node embedding that considers the collaboration information.
1) Deepwalk (random walk + Skipgram)
'''
import random
import gensim
import operator
import argparse
import collections
import pickle
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import gensim
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.spatial import distance
from itertools import combinations
from construct_team_network_func import *
from datatype_change import *
from gensim.models.callbacks import CallbackAny2Vec
from POSITION_ASSIGNMENT import *

def assign_unique_position_apply(row, position_id_mapping):
    '''
    This function get the list of position names and map to the position ID and hierarchy numbers
    '''
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

def get_random_walk(G, node, walk_length, num_walks, biased, seed):
    '''
    Given a graph and a node, return a random walk starting from the node.
    '''
    walk_paths = []
    # repete for the number of walks
    for walk in range(num_walks):
        path = [node]
        current_node = node
        # sample the next visiting node for the walk length
        for step in range(walk_length):
            if biased == True:
                out_edges = G.out_edges(current_node)
                probability_list = [G.get_edge_data(edge[0], edge[1])['prob'] for edge in out_edges]
                next_walk = random.choices(list(out_edges), weights=probability_list, k=1)[0]
                next_visit = next_walk[1]
            else:
                neighbors = list(nx.all_neighbors(G, current_node)) # extract neighbors
                next_visit = random.choice(neighbors) # randomly select the next visiting node
            path.append(next_visit)
            current_node = next_visit
        walk_paths.append(path)

    # return the list of walks
    return walk_paths
        
def deepwalk(model, G, walk_length, num_walks, window_size, emb_size, epochs, update, biased, seed):
    '''
    Use DeepWalk approach to learn the node embeddings for nodes in the 
    given graph.
    '''
    total_walk_paths = [] # list that stores all walks for all nodes

    for node in G.nodes():
        if G.degree(node) != 0:
            walk_paths = get_random_walk(G, node, walk_length, num_walks, biased, seed)
            total_walk_paths.extend(walk_paths)

    if update != True:
        # initiate word2vec model
        model = gensim.models.Word2Vec(size=emb_size, window=window_size, sg=1, hs=0, workers=3, seed=seed)

    # Build vocabulary
    previous_vocab = list(model.wv.vocab)
    previous_vocab_num = len(previous_vocab)
    model.build_vocab(total_walk_paths, update=update)

    # Train
    model.train(total_walk_paths, total_examples=model.corpus_count, epochs=epochs)
    new_vocab = list(model.wv.vocab)
    new_vocab_num = len(new_vocab)
    nodes = list(model.wv.vocab) # list of node names
    embeddings = model.wv.__getitem__(model.wv.vocab) # embeddings for every node

    return model, nodes, embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-emb_size', '--emb_size', default=30, type=int)
    parser.add_argument('-window_size', '--window_size', default=3, type=int)
    parser.add_argument('-biased', '--biased', default=False, type=bool)
    parser.add_argument('-prob', '--prob', type=int)
    parser.add_argument('-collab_type', '--collab_type', default='all', type=str, help="Collaboration type to consider when constructing collaboration networks ('NFL' or 'all' for both NFL and college coaching)")
    parser.add_argument('-hier', '--hier', default=False, type=bool, help="If node embeddings are learned based on the hierchical networks (mentorship network)")

    args = parser.parse_args()
    emb_size = args.emb_size
    window_size = args.window_size
    biased = args.biased
    prob = args.prob
    collab_type = args.collab_type
    hier = args.hier

    #################################################################
    # Load datasets
    NFL_coach_record_filename = "../datasets/NFL_Coach_Data_final_position.csv"
    all_coach_record_filename = "../datasets/all_coach_records_cleaned.csv"

    NFL_record_df = pd.read_csv(NFL_coach_record_filename)
    all_coach_record_df = pd.read_csv(all_coach_record_filename)
    NFL_instances = pd.read_csv("../datasets/NFL_Coach_Data_with_features.csv")
    #################################################################
    walk_length = 10
    num_walks = 10
    epochs = 100
    
    if prob == 123:
        selection_prob = {'downward': 1, 'peer': 2, 'upward': 3}
    elif prob == 135:
        selection_prob = {'downward': 1, 'peer': 3, 'upward': 5}

    #################################################################
    ## Clean NFL and coach records datasets
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

    ### Exclude coaches who are not in NFL data
    # This dataset includes NFL&college football coaching history of NFL coaches (2002-2019) including both qualified and unqualified positions.
    nfl_coaches = NFL_record_df.Name.unique()
    all_coach_record_df = all_coach_record_df[all_coach_record_df.Name.isin(nfl_coaches)]
    all_coach_record_df.reset_index(drop=True, inplace=True)

    print("The number of all coach records: {}".format(all_coach_record_df.shape[0]))
    print("The number of coaches in all records: {}".format(all_coach_record_df.Name.unique().shape[0]))

    ### Process coaching history to get only qualified positions.
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

    NFL_coach_history_df = all_coach_record_df[all_coach_record_df.Name.isin(NFL_record_df.Name.unique())]
    NFL_coach_history_df.reset_index(drop=True, inplace=True)

    NFL_coach_history_qualified = NFL_coach_history_df[NFL_coach_history_df.final_position != -1]
    NFL_coach_history_qualified.reset_index(drop=True, inplace=True)
    print("The number of qualified NFL coaches' history records: {}".format(NFL_coach_history_qualified.shape[0]))


    #################################################################
    ## Generate cumulative colleague network embedding
    #################################################################
    print("Generating before 2002 NFL & college coaching network")

    ### Construct the cumulative network for all coaching records before 2002
    
    cumulative_NFL_collab_G_dict = dict() # stores graphs built by cumulative collaboration ties in each year.

    # If the random walk is biased or not
    if biased:
        # use colleague network
        if hier == False:
            before2002_collab_G = nx.DiGraph()
            before2002_collab_G = construct_cumulative_directed_colleague_network(before2002_collab_G, NFL_coach_history_qualified, int(NFL_coach_history_qualified.StartYear.min()), 2001, selection_prob)
    else:
        # use mentorship network
        if hier:
            # construct mentorship network up to 2001.
            before2002_collab_G = nx.Graph()
            before2002_collab_G = construct_cumulative_mentorship_network(before2002_collab_G, NFL_coach_history_qualified, int(NFL_coach_history_qualified.StartYear.min()), 2001)
        # use colleague network
        else:
            # construct colleague network up to 2001.
            before2002_collab_G = nx.Graph()
            before2002_collab_G = construct_cumulative_colleague_network(before2002_collab_G, NFL_coach_history_qualified, int(NFL_coach_history_qualified.StartYear.min()), 2001)

    print(nx.info(before2002_collab_G))
    try:
        num_cc = nx.number_connected_components(before2002_collab_G)
    except:
        num_cc = nx.number_strongly_connected_components(before2002_collab_G)
    print("Number of connected components: {}".format(num_cc))
    # if num_cc > 1:
        # c_spl = []
        # for c in nx.connected_components(before2002_collab_G):
            # spl = nx.average_shortest_path_length(before2002_collab_G.subgraph(c))
            # c_spl.append(spl)
        # print("Average shortest path length: {}".format(",".join(map(str, c_spl))))
    # else:
        # print("Average shortest path length: {}".format(nx.average_shortest_path_length(before2002_collab_G)))

    cumulative_NFL_collab_G_dict[2001] = before2002_collab_G

    ### Deepwalk based on the cumulative network before 2002
    print("Generating embedding for before 2002 cumulative network")
    model, before2002_nodes, before2002_emb = deepwalk(None, before2002_collab_G, walk_length, num_walks, window_size, emb_size, epochs, False, biased, 100)

    ### Create a dictionary that contains the coaches' embedding in each year.
    ### - Key: the coach name
    ### - Value: dictionary of embeddings for each year.
    ###     -Key: the next year (prediction year). 
    ###         e.g., if 2002 collaborations are added to the network,
    ###                 the prediction year is 2003.
    ###     - Value: the embedding to be used for the prediction.
    cumulative_node_emb_dict = dict()
    for idx, node in enumerate(before2002_nodes):
        cumulative_node_emb_dict[node] = dict()
        cumulative_node_emb_dict[node][2002] = before2002_emb[idx]
        
    # TODO: edit code for learning node embeddings after 2002
    ### Construct the cumulative network by adding one year of NFL record
    ### to the existing network before 2002
    years = range(2002, 2020)
    # years = range(2002, 2019)
    cumulative_NFL_collab_G = before2002_collab_G.copy()
    newly_appeared_coaches = []
    new_vocab_dict = dict() # stores the list of coaches newly appeared in each year
    for year in years:
        print("Constructing cumulative network for year {}".format(year))
        # add one year of colleague relationships
        # biased random walk
        if biased:
            if hier == False:
                cumulative_NFL_collab_G = construct_cumulative_directed_colleague_network(cumulative_NFL_collab_G, NFL_record_df, year, year, selection_prob)
                if collab_type == "all":
                    cumulative_NFL_collab_G = construct_cumulative_directed_colleague_network(cumulative_NFL_collab_G, NFL_coach_history_qualified, year, year, selection_prob)
            num_cc = nx.number_strongly_connected_components(cumulative_NFL_collab_G)

        else:
            if hier:
                cumulative_NFL_collab_G = construct_cumulative_mentorship_network(cumulative_NFL_collab_G, NFL_record_df, year, year)
                if collab_type == "all":
                    cumulative_NFL_collab_G = construct_cumulative_mentorship_network(cumulative_NFL_collab_G, NFL_coach_history_qualified, year, year)

            else:
                cumulative_NFL_collab_G = construct_cumulative_colleague_network(cumulative_NFL_collab_G, NFL_record_df, year, year)
                if collab_type == "all":
                    cumulative_NFL_collab_G = construct_cumulative_colleague_network(cumulative_NFL_collab_G, NFL_coach_history_qualified, year, year)
            num_cc = nx.number_connected_components(cumulative_NFL_collab_G)

        print(nx.info(cumulative_NFL_collab_G))
        print("Number of connected components: {}".format(num_cc))
        # if num_cc > 1:
            # c_spl = []
            # for c in nx.connected_components(cumulative_NFL_collab_G):
                # spl = nx.average_shortest_path_length(cumulative_NFL_collab_G.subgraph(c))
                # c_spl.append(str(spl))
            # print("Average shortest path length:{}".format(",".join(map(str,c_spl))))
        # else:
            # print("Average shortest_path length: {}".format(nx.average_shortest_path_length(cumulative_NFL_collab_G)))

        cumulative_NFL_collab_G_dict[year] = cumulative_NFL_collab_G
        previous_model_vocab = list(model.wv.vocab)
        # Learn node embeddings
        model, cumulative_NFL_nodes, cumulative_NFL_emb = deepwalk(model, cumulative_NFL_collab_G, walk_length, num_walks, window_size, emb_size, epochs, True, biased, 100)

        new_model_vocab = list(model.wv.vocab)
        random_initialize_vocab = [x for x in new_model_vocab if x not in previous_model_vocab]
        new_vocab_dict[year] = random_initialize_vocab

        # Add new embedding to the dictionary
        for idx, node in enumerate(cumulative_NFL_nodes):
            if node in cumulative_node_emb_dict:
                cumulative_node_emb_dict[node][year+1] = cumulative_NFL_emb[idx]
            else:
                cumulative_node_emb_dict[node] = dict()
                cumulative_node_emb_dict[node][year+1] = cumulative_NFL_emb[idx]

    cumulative_emb_df = dict_of_dict_to_dataframe(cumulative_node_emb_dict, emb_size)
    cumulative_emb_df.to_csv("../datasets/final_embedding/cumulative_collab_G_node_embedding_{}_hier{}_biased{}_selectionprob{}_w{}_df.csv".format(collab_type, hier, biased, prob, window_size), index=False, encoding="utf-8-sig")

    # distributions of downward, peer, and upward edges for all nodes (~2018)
    # count_dict = {'downward':[], 'peer':[], 'upward':[]}
    # g = cumulative_NFL_collab_G_dict[2018]
    # for node in tqdm(g.nodes()):
        # connected_edges = g.out_edges(node)
        # edge_directions = [g.get_edge_data(e[0],e[1])['prob'] for e in connected_edges]
        # count_dict['downward'].append(edge_directions.count(1))
        # count_dict['peer'].append(edge_directions.count(2))
        # count_dict['upward'].append(edge_directions.count(3))

    # print("** First graph **")
    # print("Diameter: {}, avg. path length: {}".format(nx.algorithms.distance_measures.diameter(before2002_collab_G), nx.average_shortest_path_length(before2002_collab_G)))

    # print("** Final graph **")
    # print("Diameter: {}, avg. path length: {}".format(nx.algorithms.distance_measures.diameter(g), nx.average_shortest_path_length(g)))




