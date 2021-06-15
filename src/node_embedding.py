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

class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    # def on_epoch_begin(self, model):
        # print("Epoch{} start".format(self.epoch))
        # print(model.wv.__getitem__(model.wv.vocab))

    def on_epoch_end(self, model):
        self.epoch += 1

def get_random_walk(graph, node, walk_length, num_walks):
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
            neighbors = list(nx.all_neighbors(graph, current_node)) # extract neighbors
            next_visit = random.choice(neighbors) # randomly select the next visiting node
            path.append(next_visit)
            current_node = next_visit
        walk_paths.append(path)

    # return the list of walks
    return walk_paths
        
def deepwalk(model, G, walk_length, num_walks, window_size, emb_size, epochs, update):
    '''
    Use DeepWalk approach to learn the node embeddings for nodes in the 
    given graph.
    '''
    total_walk_paths = [] # list that stores all walks for all nodes

    for node in G.nodes():
        walk_paths = get_random_walk(G, node, walk_length, num_walks)
        total_walk_paths.extend(walk_paths)

    if update != True:
        # initiate word2vec model
        model = gensim.models.Word2Vec(size=emb_size, window=window_size, sg=1, hs=0, workers=3, callbacks=[EpochLogger()])

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
    parser.add_argument('-collab_type', '--collab_type', type=str, help="Collaboration type to consider when constructing collaboration networks ('NFL' or 'all' for both NFL and college coaching)")

    args = parser.parse_args()
    emb_size = args.emb_size
    window_size = args.window_size
    collab_type = args.collab_type

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

    random.seed(100)

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
    nfl_coaches = NFL_record_df.Name.unique()
    all_coach_record_df = all_coach_record_df[all_coach_record_df.Name.isin(nfl_coaches)]
    all_coach_record_df.reset_index(drop=True, inplace=True)

    print("The number of all coach records: {}".format(all_coach_record_df.shape[0]))
    print("The number of coaches in all records: {}".format(all_coach_record_df.Name.unique().shape[0]))


    #################################################################
    ## Generate cumulative colleague network embedding
    #################################################################
    print("Generating before 2002 cumulative network")
    ### Construct the cumulative network for all coaching records before 2002
    before2002_colleague_G = nx.Graph()
    all_record_year_min = int(all_coach_record_df.StartYear.min())
    
    cumulative_NFL_colleague_G_dict = dict() # stores graphs built by cumulative collaboration ties in each year.
    before2002_colleague_G = construct_cumulative_colleague_network(before2002_colleague_G, all_coach_record_df, all_record_year_min, 2001)
    print(nx.info(before2002_colleague_G))
    num_cc = nx.number_connected_components(before2002_colleague_G)
    print("Number of connected components: {}".format(num_cc))
    if num_cc > 1:
        c_spl = []
        for c in nx.connected_components(before2002_colleague_G):
            spl = nx.average_shortest_path_length(before2002_colleague_G.subgraph(c))
            c_spl.append(spl)
        print("Average shortest path length: {}".format(",".join(c_spl)))
    else:
        print("Average shortest path length: {}".format(nx.average_shortest_path_length(before2002_colleague_G)))

    cumulative_NFL_colleague_G_dict[2001] = before2002_colleague_G

    ### Deepwalk based on the cumulative network before 2002
    print("Generating embedding for before 2002 cumulative network")
    model, before2002_nodes, before2002_emb = deepwalk(None, before2002_colleague_G, walk_length, num_walks, window_size, emb_size, epochs, False)

    ### Create a dictionary that contains the coaches' embedding in each year.
    ### - Key: the coach name
    ### - Value: dictionary of embeddings for each year.
    ###     -Key: the next year (prediction year). 
    ###         e.g., if 2002 colleague relationships are added to the network,
    ###                 the prediction year is 2003.
    ###     - Value: the embedding to be used for the prediction.
    cumulative_node_emb_dict = dict()
    for idx, node in enumerate(before2002_nodes):
        cumulative_node_emb_dict[node] = dict()
        cumulative_node_emb_dict[node][2002] = before2002_emb[idx]
        

    ### Construct the cumulative network by adding one year of NFL record
    ### to the existing network before 2002
    years = range(2002, 2020)
    # years = range(2002, 2019)
    cumulative_NFL_colleague_G = before2002_colleague_G.copy()
    newly_appeared_coaches = []
    new_vocab_dict = dict() # stores the list of coaches newly appeared in each year
    for year in years:
        print("Constructing cumulative network for year {}".format(year))
        # add one year of colleague relationships
        if collab_type == "NFL":
            cumulative_NFL_colleague_G = construct_cumulative_colleague_network(cumulative_NFL_colleague_G, NFL_record_df, year, year)
        elif collab_type == "all":
            cumulative_NFL_colleague_G = construct_cumulative_colleague_network(cumulative_NFL_colleague_G, NFL_record_df, year, year)
            cumulative_NFL_colleague_G = construct_cumulative_colleague_network(cumulative_NFL_colleague_G, all_coach_record_df, year, year)
        print(nx.info(cumulative_NFL_colleague_G))
        num_cc = nx.number_connected_components(cumulative_NFL_colleague_G)
        print("Number of connected components: {}".format(num_cc))
        if num_cc > 1:
            c_spl = []
            for c in nx.connected_components(cumulative_NFL_colleague_G):
                spl = nx.average_shortest_path_length(cumulative_NFL_colleague_G.subgraph(c))
                c_spl.append(str(spl))
            print("Average shortest path length:{}".format(",".join(c_spl)))
        else:
            print("Average shortest_path length: {}".format(nx.average_shortest_path_length(cumulative_NFL_colleague_G)))

        cumulative_NFL_colleague_G_dict[year] = cumulative_NFL_colleague_G
        previous_model_vocab = list(model.wv.vocab)
        # Learn node embeddings
        model, cumulative_NFL_nodes, cumulative_NFL_emb = deepwalk(model, cumulative_NFL_colleague_G, walk_length, num_walks, window_size, emb_size, epochs, True)

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
    cumulative_emb_df.to_csv("../datasets/cumulative_colleague_G_node_embedding_{}_df.csv".format(collab_type), index=False, encoding="utf-8-sig")

