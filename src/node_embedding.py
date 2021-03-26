'''
This script generates node embedding that considers the collaboration information.
1) Deepwalk (random walk + Skipgram)
'''
import pandas as pd
import numpy as np
import networkx as nx
import random
import gensim
from tqdm import tqdm
from sklearn.decomposition import PCA
from generate_team_network import *
import matplotlib.pyplot as plt

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
        

if __name__ == "__main__":
    #################################################################
    # Load datasets
    NFL_coach_record_filename = "../datasets/2002-2019_NFL_Coach_Data_final_position.csv"

    NFL_record_df = pd.read_csv(NFL_coach_record_filename)
    #################################################################
    # exclude interim head coaches
    NFL_record_df = NFL_record_df[NFL_record_df.final_position != "iHC"]
    NFL_record_df.reset_index(drop=True, inplace=True)

    # exclude coaches with no proper positions
    NFL_record_df = NFL_record_df[(NFL_record_df.final_position != -1) & (NFL_record_df.final_hier_num != -1)]
    NFL_record_df.reset_index(drop=True, inplace=True)

    print("The number of NFL records: {}".format(NFL_record_df.shape[0]))

    yearly_node_embedding_dict = dict()

    years = NFL_record_df.Year.unique()
    teams = NFL_record_df.Team.unique()



    print("Generating embedding...")
    for year in tqdm(years):
        yearly_node_embedding_dict[year] = dict()
        for team in teams:
            team_G = construct_network(NFL_record_df, year, team)

            total_walk_paths = [] # list that stores all walks for all nodes
            for node in team_G.nodes():
                walk_paths = get_random_walk(team_G, node, 10, 10)
                total_walk_paths.extend(walk_paths)

            # initiate word2vec model
            model = gensim.models.Word2Vec(size=10, window=3, sg=1, hs=0, workers=3)

            # Build vocabulary
            model.build_vocab(total_walk_paths)

            # Train
            model.train(total_walk_paths, total_examples=model.corpus_count, epochs=20)
            nodes = list(model.wv.vocab)
            embeddings = model.wv.__getitem__(model.wv.vocab)

            for i in range(len(nodes)):
                node = nodes[i]
                if node in yearly_node_embedding_dict[year]:
                    print(year, team, node)
                else:
                    yearly_node_embedding_dict[year][node] = embeddings[i,:]

    ### validate node embedding
    print("Validaing the node embedding via visualization...")
    year = 2019
    year_embeddings_dict = yearly_node_embedding_dict[year]
    year_record = NFL_record_df[NFL_record_df.Year==year]
    year_record.reset_index(drop=True, inplace=True)
    # append the year embedding to the coach records
    year_embeddings = year_record.Name.map(year_embeddings_dict)

    # pca decomposition
    pca = PCA(n_components=2)
    transformed_arr = pca.fit_transform(year_embeddings.tolist())
    transformed_emb_cols = pd.DataFrame(transformed_arr, index=year_record.index, columns=["emb1", "emb2"])
    year_record = pd.concat([year_record, transformed_emb_cols], axis=1)
        
    # plot the 2-d embedding
    fig, ax = plt.subplots(figsize=(10,8))
    c = range(0, 20)
    for t in year_record.Team.unique()[:20]:
        year_team_record = year_record[year_record.Team==t]
        ax.scatter(np.array(year_team_record["emb1"]), np.array(year_team_record["emb2"]), c=t, label=c, cmap="tab20")

    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left", fontsize='small')
    plt.tight_layout()
    plt.savefig("../plots/node_embedding_pca.png")
    plt.close()
