'''
This script runs GCN model on hierarchical graphs as a benchmark.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse
import statistics
from tqdm import tqdm
from generate_team_features_func import *
from aggregate_team_embedding_func import *
from classifier_func import *
from utils import *
from construct_team_network_func import *

def prepare_features_labels(G, coach_features_df, team_labels_df, team_features_df, coach_feature_names, team_feature_names, label_name):

    coach_features = torch.zeros((coach_features_df.shape[0], len(coach_feature_names)))
    team_labels = torch.zeros((team_labels_df.shape[0], 1))
    team_features = torch.zeros((team_features_df.shape[0], len(team_feature_names)))
    team_idx = 0

    feature_extracted = np.array(coach_features_df[coach_feature_names])
    for idx, node in enumerate(G.nodes()):
        G.nodes[node]["f"] = torch.Tensor(feature_extracted[node,:])
        coach_features[idx,:] = torch.Tensor(feature_extracted[node,:])
        if G.nodes[node]["final_position"] == "HC":
            year = G.nodes[node]["Year"]
            team = G.nodes[node]["Team"]
            label = team_labels_df[(team_labels_df.Year==year) & (team_labels_df.Team==team.replace(" (NFL)", ""))].failure
            team_labels[team_idx] = int(label)

            if len(team_feature_names) != 0:
                team_feature = team_features_df[(team_features_df.Year == year) & (team_features_df.Team==team)][team_feature_names]
                team_features[team_idx,:] = team_feature.values

            team_idx += 1

    return G, coach_features, team_labels, team_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-feature_set', '--feature_set', type=int, help="Feature set number\n(0: Basic features, 1: Basic features + salary, 2: Basic features & node embedding + salary, 3: Basic features & node embedding + salary + diversity)")
    parser.add_argument('-train_split_year', '--train_split_year', type=int, help="Maximum year for training set")
    parser.add_argument('-valid_split_year', '--valid_split_year', type=int, help="Maximum year for validation set")
    parser.add_argument('-emb_size', '--emb_size', type=int, help="Node embedding size")
    parser.add_argument('-collab_type', '--collab_type', default='all', type=str, help="Collaboration type: NFL or all")
    parser.add_argument('-pos_connect', '--pos_connect', default=False, type=bool, help="Whether there's additional attention layer for pairwise position coaches")
    parser.add_argument('-hier', '--hier', default=False, type=bool, help="Mentorship network or not")
    parser.add_argument('-biased', '--biased', default=False, type=bool, help="Random walks are biased or not")
    parser.add_argument('-prob', '--prob', type=int)
    parser.add_argument('-w', '--w', default=3, type=int, help="window size")
    parser.add_argument('-agg', '--agg', type=str, help="aggregation (mean or concat)")
    parser.add_argument('-mode', '--mode', default='expanded', type=str, help="simple for only considering teams with 8 position coach titles or expanded for considering all teams with qualified position coach titles")


    args = parser.parse_args()
    emb_size = args.emb_size
    feature_set = args.feature_set
    train_split_year = args.train_split_year
    valid_split_year = args.valid_split_year
    collab_type = args.collab_type
    hier = args.hier
    biased = args.biased
    prob = args.prob
    w = args.w
    agg = args.agg
    mode = args.mode
    pos_connect = args.pos_connect

    #################################################################
    # Load datasets
    NFL_coach_record_filename = "../datasets/NFL_Coach_Data_with_features_{}_size{}_collab{}_hier{}_biased{}_selectionprob{}_w{}.csv".format(mode, emb_size, collab_type, hier, biased, prob, w)
    team_labels_filename = "../datasets/team_labels.csv"
    team_salary_filename = "../datasets/Total_Salary.csv"

    NFL_record_df = pd.read_csv(NFL_coach_record_filename)
    team_labels_df = pd.read_csv(team_labels_filename)
    team_salary_df = pd.read_csv(team_salary_filename)
    #################################################################
    # drop seasons with no salary data available
    team_salary_df = team_salary_df.dropna(subset=["Total_Salary"])
    # change team names with "(NFL)" appended
    team_names = team_salary_df.Team + " (NFL)"
    team_salary_df = team_salary_df.assign(Team = team_names)

    # Define column names to be used as coach features
    basic_features = ["TotalYearsInNFL", "Past5yrsWinningPerc_best", "Past5yrsWinningPerc_avg"]
    cumul_emb_features = NFL_record_df.columns[NFL_record_df.columns.str.contains("cumul_emb")].tolist()

    # generate team feature set
    team_diversity_df = generate_team_diversity_feature(NFL_record_df, cumul_emb_features)
    team_features = team_salary_df.merge(team_diversity_df, how="left", on=["Team", "Year"])

    # define feature set
    if feature_set == 0:
        coach_feature_names = basic_features
        team_feature_names = []
    elif feature_set == 1:
        coach_feature_names = basic_features
        team_feature_names = ["Salary_Rank"]
    elif feature_set == 21:
        coach_feature_names = basic_features + cumul_emb_features
        team_feature_names = ["Salary_Rank"]
    elif feature_set == 22:
        coach_feature_names = basic_features + cumul_emb_features
        team_feature_names = []
    elif feature_set == 31:
        coach_feature_names = basic_features + cumul_emb_features
        team_feature_names = ["Salary_Rank", "Max_Emb_Similarity", "Mean_Emb_Similarity"]
    elif feature_set == 32:
        coach_feature_names = basic_features + cumul_emb_features
        team_feature_names = ["Max_Emb_Similarity", "Mean_Emb_Similarity"]

    print("Feature set {}".format(feature_set))

    if mode == "simple":
        print("Extracting complete teams...")
        off_position_titles = ["QB", "RB", "OL", "WR", "TE"] 
        def_position_titles = ["LB", "DL", "Sec"]
        complete_year_team_pairs = get_complete_teams(NFL_record_df, off_position_titles + def_position_titles)
        year_team_tuple = NFL_record_df[["Year","Team"]].apply(tuple, axis=1)
        NFL_record_df = NFL_record_df.assign(year_team = year_team_tuple)
        NFL_record_df = NFL_record_df[NFL_record_df.year_team.isin(complete_year_team_pairs)]
        NFL_record_df.reset_index(drop=True, inplace=True)
        print("Number of teams with complete position coaches: {}".format(NFL_record_df[["Year","Team"]].drop_duplicates().shape[0]))
    elif mode == "expanded":
        off_position_titles = []
        def_position_titles = []

    #########################################################
    ### Split coach record, salary, and label into train, validation, and test set
    #########################################################

    train_record = NFL_record_df[(NFL_record_df.Year >= 2002) & (NFL_record_df.Year <= train_split_year)]
    train_record.reset_index(drop=True, inplace=True)

    train_team_features = team_features[(team_features.Year >= 2002) & (team_features.Year <= train_split_year)]
    train_team_features.reset_index(drop=True, inplace=True)
    
    train_labels = team_labels_df[(team_labels_df.Year >= 2002) & (team_labels_df.Year <= train_split_year)]
    train_labels.reset_index(drop=True, inplace=True)

    # valid
    valid_record = NFL_record_df[(NFL_record_df.Year > train_split_year) & (NFL_record_df.Year <= valid_split_year)]
    valid_record.reset_index(drop=True, inplace=True)

    valid_team_features = team_features[(team_features.Year > train_split_year) & (team_features.Year <= valid_split_year)]
    valid_team_features.reset_index(drop=True, inplace=True)

    valid_labels = team_labels_df[(team_labels_df.Year > train_split_year) & (team_labels_df.Year <= valid_split_year)]
    valid_labels.reset_index(drop=True, inplace=True)

    # test
    test_record = NFL_record_df[(NFL_record_df.Year > valid_split_year) & (NFL_record_df.Year <= 2019)]
    test_record.reset_index(drop=True, inplace=True)

    test_team_features = team_features[(team_features.Year > valid_split_year) & (team_features.Year <= 2019)]
    test_team_features.reset_index(drop=True, inplace=True)

    test_labels = team_labels_df[(team_labels_df.Year > valid_split_year) & (team_labels_df.Year <= 2019)]
    test_labels.reset_index(drop=True, inplace=True)

    print("Number of training records: {}, validation records: {}, testing records: {}".format(train_record.shape[0], valid_record.shape[0], test_record.shape[0]))

    # Generate ID columns for train and test data
    train_record.reset_index(inplace=True)
    train_record = train_record.rename(columns={'index':'ID'})

    valid_record.reset_index(inplace=True)
    valid_record = valid_record.rename(columns={'index':'ID'})

    test_record.reset_index(inplace=True)
    test_record = test_record.rename(columns={'index':'ID'})

    # Nodes are each coach record
    # Edges are directed edges from high to lower hierarchy.
    print("** Constructing train team network... **")
    train_id_record_dict, train_team_G = construct_fullseason_mentorship_network(train_record, pos_connect)
    print(nx.info(train_team_G))
    train_components = nx.weakly_connected_components(train_team_G)
    num_train_components = len(list(train_components))
    print("The number of train teams (components): {}".format(num_train_components))

    print("** Constructing validation team network... **")
    valid_id_record_dict, valid_team_G = construct_fullseason_mentorship_network(valid_record, pos_connect)
    print(nx.info(valid_team_G))
    valid_components = nx.weakly_connected_components(valid_team_G)
    num_valid_components = len(list(valid_components))
    print("The number of valid teams (components): {}".format(num_valid_components))

    print("** Constructing test team network... **")
    test_id_record_dict, test_team_G = construct_fullseason_mentorship_network(test_record, pos_connect)
    print(nx.info(test_team_G))
    test_components = nx.weakly_connected_components(test_team_G)
    num_test_components = len(list(test_components))
    print("The number of test teams (components): {}".format(num_test_components))


    #################################################################
    ### Construct features, labels, and salary tensors according to
    ### the order of nodes in the graph
    ### The order of features is the same as the order of nodes in the graph.
    ### The orders of labels and salary are the asme as the order of head coaches
    ### appearing in the graph.
    #################################################################
    print("Preparing features and labels...")

    train_feature_extracted = np.array(train_record[coach_feature_names])
    valid_feature_extracted = np.array(valid_record[coach_feature_names])
    test_feature_extracted = np.array(test_record[coach_feature_names])

    train_team_feature_extracted = np.array(train_team_features[team_feature_names])
    valid_team_feature_extracted = np.array(valid_team_features[team_feature_names])
    test_team_feature_extracted = np.array(test_team_features[team_feature_names])

    # generate features, labels, and salary vectors for training set
    # in the same order of the train team graph
    train_team_G, train_f, train_team_labels, train_team_features= prepare_features_labels(train_team_G, train_record, train_labels, train_team_features, coach_feature_names, team_feature_names, "failure")

    # generate features, labels, and salary vectors for valid set
    # in the same order of the valid team graph
    valid_team_G, valid_f, valid_team_labels, valid_team_features= prepare_features_labels(valid_team_G, valid_record, valid_labels, valid_team_features, coach_feature_names, team_feature_names, "failure")

    # generate features, labels, and salary vectors for test set
    # in the same order of the test team graph
    test_team_G, test_f, test_team_labels, test_team_features= prepare_features_labels(test_team_G, test_record, test_labels, test_team_features, coach_feature_names, team_feature_names, "failure")

    print("Model GCN")
    train_g = nx.Graph(train_team_G)
    valid_g = nx.Graph(valid_team_G)
    test_g = nx.Graph(test_team_G)

    x = train_f
    edge_index = nx.convert_matrix.to_numpy_array(train_g)
    conv = GCNConv(train_f.shape[1], train_f.shape[1])

