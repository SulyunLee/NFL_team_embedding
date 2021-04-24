'''
This script generates node embedding that considers the collaboration information.
1) Deepwalk (random walk + Skipgram)
'''
import random
import gensim
import operator
import argparse
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.spatial import distance
from itertools import combinations
from construct_team_network_func import *
from datatype_change import *

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
        
def deepwalk(G, walk_length, num_walks, window_size, emb_size, epochs):
    '''
    Use DeepWalk approach to learn the node embeddings for nodes in the 
    given graph.
    '''
    total_walk_paths = [] # list that stores all walks for all nodes

    for node in G.nodes():
        walk_paths = get_random_walk(G, node, walk_length, num_walks)
        total_walk_paths.extend(walk_paths)

    # initiate word2vec model
    model = gensim.models.Word2Vec(size=emb_size, window=window_size, sg=1, hs=0, workers=3)

    # Build vocabulary
    model.build_vocab(total_walk_paths)

    # Train
    model.train(total_walk_paths, total_examples=model.corpus_count, epochs=epochs)
    nodes = list(model.wv.vocab) # list of node names
    embeddings = model.wv.__getitem__(model.wv.vocab) # embeddings for every node

    return nodes, embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-emb_size', '--emb_size', default=False, type=int)
    parser.add_argument('-window_size', '--window_size', default=False, type=int)

    args = parser.parse_args()
    emb_size = args.emb_size
    window_size = args.window_size

    #################################################################
    # Load datasets
    NFL_coach_record_filename = "../datasets/NFL_Coach_Data_final_position.csv"
    all_coach_record_filename = "../datasets/all_coach_records_cleaned.csv"

    NFL_record_df = pd.read_csv(NFL_coach_record_filename)
    all_coach_record_df = pd.read_csv(all_coach_record_filename)
    #################################################################
    walk_length = 10
    num_walks = 10
    epochs = 30

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
    
    before2002_colleague_G = construct_cumulative_colleague_network(before2002_colleague_G, all_coach_record_df, all_record_year_min, 2001)
    print(nx.info(before2002_colleague_G))

    ### Deepwalk based on the cumulative network before 2002
    print("Generating embedding for before 2002 cumulative network")
    before2002_nodes, before2002_emb = deepwalk(before2002_colleague_G, walk_length, num_walks, window_size, emb_size, epochs)

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
    cumulative_NFL_colleague_G = before2002_colleague_G.copy()
    newly_appeared_coaches = []
    for year in years:
        print("Constructing cumulative network for year {}".format(year))
        # add one year of colleague relationships
        cumulative_NFL_colleague_G = construct_cumulative_colleague_network(cumulative_NFL_colleague_G, NFL_record_df, year, year)
        print(nx.info(cumulative_NFL_colleague_G))
        # Learn node embeddings
        cumulative_NFL_nodes, cumulative_NFL_emb = deepwalk(cumulative_NFL_colleague_G, walk_length, num_walks, window_size, emb_size, epochs)

        # Add new embedding to the dictionary
        for idx, node in enumerate(cumulative_NFL_nodes):
            if node in cumulative_node_emb_dict:
                cumulative_node_emb_dict[node][year+1] = cumulative_NFL_emb[idx]
            else:
                cumulative_node_emb_dict[node] = dict()
                cumulative_node_emb_dict[node][year+1] = cumulative_NFL_emb[idx]

    cumulative_emb_df = dict_of_dict_to_dataframe(cumulative_node_emb_dict, emb_size)
    cumulative_emb_df.to_csv("../datasets/cumulative_colleague_G_node_embedding_df.csv", index=False, encoding="utf-8-sig")

    ### Check the newly appeared coaches in NFL dataset
    newly_appeared_coaches = []
    for node in cumulative_NFL_colleague_G.nodes():
        if node not in before2002_colleague_G.nodes():
            newly_appeared_coaches.append(node)

    NFL_all_record = all_coach_record_df[(all_coach_record_df.NFL == 1) & (all_coach_record_df.StartYear >= 2002) & (all_coach_record_df.EndYear <= 2019)]
    NFL_all_record.reset_index(drop=True, inplace=True)

    #################################################################
    ## Validate the learned embedding
    #################################################################
    print("Validating node embedding...")
    ## - x-axis: N-hops
    ## - y-axis: the cosine similarity
    # dictionary that stores the shortest path length between node pairs
    spl_dict = dict(nx.shortest_path_length(before2002_colleague_G))
    node_combination = list(combinations(before2002_nodes, 2)) # all combinations of nodes
    hop_arr = np.zeros((len(node_combination))).astype(int)
    emb_similarity_arr = np.zeros((len(node_combination)))
    for idx, comb in enumerate(node_combination):
        spl = spl_dict[comb[0]][comb[1]]
        node1_idx = before2002_nodes.index(comb[0])
        node2_idx = before2002_nodes.index(comb[1])

        node1_emb = before2002_emb[node1_idx,:]
        node2_emb = before2002_emb[node2_idx,:]
        cosine_similarity = 1 - distance.cosine(node1_emb, node2_emb)

        # append number of shortest paths and embedding similarity to array
        hop_arr[idx] = spl
        emb_similarity_arr[idx] = cosine_similarity

    ### Visualize boxplot of similarity for each shortest path length
    fig, ax = plt.subplots(figsize=(10,8))
    hop1_emb_similarity = emb_similarity_arr[np.where(hop_arr == 1)[0]]
    hop2_emb_similarity = emb_similarity_arr[np.where(hop_arr == 2)[0]]
    hop3_emb_similarity = emb_similarity_arr[np.where(hop_arr == 3)[0]]
    hop4_emb_similarity = emb_similarity_arr[np.where(hop_arr == 4)[0]]
    hop5_emb_similarity = emb_similarity_arr[np.where(hop_arr == 5)[0]]
    hop6_emb_similarity = emb_similarity_arr[np.where(hop_arr == 6)[0]]
    hop7_emb_similarity = emb_similarity_arr[np.where(hop_arr == 7)[0]]

    ax.boxplot([hop1_emb_similarity, hop2_emb_similarity, hop3_emb_similarity,\
            hop4_emb_similarity, hop5_emb_similarity, hop6_emb_similarity,\
            hop7_emb_similarity])
    ax.set_xlabel("Shortest path length", fontsize=12)
    ax.set_ylabel("Embedding cosine similarity", fontsize=12)
    ax.set_title("Embedding size = {}, window size = {}".format(emb_size, window_size),\
            fontsize=14)

    plt.tight_layout()
    plt.savefig("../plots/before2002_node_embedding_validation_embsize{}_windowsize{}.png".format(emb_size, window_size))
    plt.close()

    fig, ax = plt.subplots(figsize=(10,8))
    ax.hist(emb_similarity_arr)
    ax.set_xlabel("Embedding cosine similarity", fontsize=12)
    ax.set_title("Embedding size = {}, window size = {}".format(emb_size, window_size),\
            fontsize=14)

    plt.tight_layout()
    plt.savefig("../plots/before2002_node_embedding_overall_validation_embsize{}_windowsize{}.png".format(emb_size, window_size))
    plt.close()


    spl_dict = dict(nx.shortest_path_length(cumulative_NFL_colleague_G))
    node_combination = list(combinations(cumulative_NFL_nodes, 2)) # all combinations of nodes
    hop_arr = np.zeros((len(node_combination))).astype(int)
    emb_similarity_arr = np.zeros((len(node_combination)))
    for idx, comb in enumerate(node_combination):
        spl = spl_dict[comb[0]][comb[1]]
        node1_idx = cumulative_NFL_nodes.index(comb[0])
        node2_idx = cumulative_NFL_nodes.index(comb[1])

        node1_emb = cumulative_NFL_emb[node1_idx,:]
        node2_emb = cumulative_NFL_emb[node2_idx,:]
        cosine_similarity = 1 - distance.cosine(node1_emb, node2_emb)

        # append number of shortest paths and embedding similarity to array
        hop_arr[idx] = spl
        emb_similarity_arr[idx] = cosine_similarity

    # Visualie the embedding similarity distribution by shortest path length
    fig, ax = plt.subplots(figsize=(10,8))
    hop1_emb_similarity = emb_similarity_arr[np.where(hop_arr == 1)[0]]
    hop2_emb_similarity = emb_similarity_arr[np.where(hop_arr == 2)[0]]
    hop3_emb_similarity = emb_similarity_arr[np.where(hop_arr == 3)[0]]
    hop4_emb_similarity = emb_similarity_arr[np.where(hop_arr == 4)[0]]
    hop5_emb_similarity = emb_similarity_arr[np.where(hop_arr == 5)[0]]
    hop6_emb_similarity = emb_similarity_arr[np.where(hop_arr == 6)[0]]
    hop7_emb_similarity = emb_similarity_arr[np.where(hop_arr == 7)[0]]

    ax.boxplot([hop1_emb_similarity, hop2_emb_similarity, hop3_emb_similarity,\
            hop4_emb_similarity, hop5_emb_similarity, hop6_emb_similarity,\
            hop7_emb_similarity])
    ax.set_xlabel("Shortest path length", fontsize=12)
    ax.set_ylabel("Embedding cosine similarity", fontsize=12)
    ax.set_title("Embedding size = {}, window size = {}".format(emb_size, window_size),\
            fontsize=14)

    plt.tight_layout()
    plt.savefig("../plots/before2019_node_embedding_validation_embsize{}_windowsize{}.png".format(emb_size, window_size))
    plt.close()

    # Visualize the distribution of overall pairwise embedding similarity
    fig, ax = plt.subplots(figsize=(10,8))
    ax.hist(emb_similarity_arr)
    ax.set_xlabel("Embedding cosine similarity", fontsize=12)
    ax.set_title("Embedding size = {}, window size = {}".format(emb_size, window_size),\
            fontsize=14)

    plt.tight_layout()
    plt.savefig("../plots/before2019_node_embedding_overall_validation_embsize{}_windowsize{}.png".format(emb_size, window_size))
    plt.close()

