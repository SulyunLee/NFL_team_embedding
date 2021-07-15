import copy
import pandas as pd
import numpy as np
import statistics
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from generate_team_features_func import *
from construct_team_network_func import *
from hier_gat import *
from utils import *

def prepare_features_labels(team_G, coach_features_df, team_labels_df, team_features_df, coach_feature_names, team_feature_names, label_name, feature_set):

    coach_features = torch.zeros((coach_features_df.shape[0], len(coach_feature_names)))
    team_labels = torch.zeros((team_labels_df.shape[0], 1))
    team_features = torch.zeros((team_features_df.shape[0], len(team_feature_names)))
    team_idx = 0
    team_features_dict = dict()
    team_labels_dict = dict()

    feature_extracted = np.array(coach_features_df[coach_feature_names])
    for idx, node in enumerate(team_G.nodes()):
        # save the raw featuresr of each coach as the node attribute "f"
        team_G.nodes[node]["f"] = torch.Tensor(feature_extracted[node,:])
        coach_features[idx,:] = torch.Tensor(feature_extracted[node,:])
        # The order of features is the same as the order of graph nodes.
        if team_G.nodes[node]["final_position"] == "HC":
            # insert labels
            year = team_G.nodes[node]["Year"]
            team = team_G.nodes[node]["Team"]
            label = team_labels_df[(team_labels_df.Year==year) & (team_labels_df.Team==team.replace(" (NFL)", ""))].failure
            team_labels_dict[(year, team)] = int(label)


            if feature_set != 0:
                # insert team features
                team_feature = team_features_df[(team_features_df.Year == year) & (team_features_df.Team==team)][team_feature_names]
                team_features_dict[(year, team)] = torch.Tensor(team_feature.values)

            team_idx += 1

    return team_G, coach_features, team_labels_dict, team_features_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-feature_set', '--feature_set', type=int, help="Feature set number\n(0: Basic features, 1: Basic features + salary, 2: Basic features & node embedding + salary, 3: Basic features & node embedding + salary + diversity)")
    parser.add_argument('-train_split_year', '--train_split_year', type=int, help="Maximum year for training set")
    parser.add_argument('-valid_split_year', '--valid_split_year', type=int, help="Maximum year for validation set")
    parser.add_argument('-seed', '--seed', default=0, type=int, help="Random seed")
    parser.add_argument('-pos_connect', '--pos_connect', default=False, type=bool, help="Whether there's additional attention layer for pairwise position coaches")
    parser.add_argument('-num_heads', '--num_heads', default=1, type=int, help="number of multi-heads")

    args = parser.parse_args()
    feature_set = args.feature_set
    train_split_year = args.train_split_year
    valid_split_year = args.valid_split_year
    seed = args.seed
    pos_connect = args.pos_connect
    num_heads = args.num_heads

    dropout = True

    #################################################################
    # Load datasets
    NFL_coach_record_filename = "../datasets/NFL_Coach_Data_with_features.csv"
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
    # basic_features = ["TotalYearsInNFL", "Past5yrsWinningPerc_best", "Past5yrsWinningPerc_avg", "HC", "Coord"]
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
    elif feature_set == 2:
        coach_feature_names = basic_features + cumul_emb_features
        team_feature_names = ["Salary_Rank"]
    elif feature_set == 3:
        coach_feature_names = basic_features + cumul_emb_features
        team_feature_names = ["Salary_Rank", "Max_Emb_Similarity", "Mean_Emb_Similarity"]

    print("Feature set {}".format(feature_set))

    # define team embedding dimension
    if feature_set == 0 or feature_set == 1:
        emb_dim = 4
    elif feature_set == 2 or feature_set == 3:
        emb_dim = 16

    #########################################################
    ### Split coach record, salary, and label into train, validation, and test set
    #########################################################
    # Separate train and test datasets
    # Train
    train_record = copy.deepcopy(NFL_record_df[(NFL_record_df.Year >= 2002) & (NFL_record_df.Year <= train_split_year)])
    train_record.reset_index(drop=True, inplace=True)

    train_team_features = copy.deepcopy(team_features[(team_features.Year >= 2002) & (team_features.Year <= train_split_year)])
    train_team_features.reset_index(drop=True, inplace=True)

    train_labels = copy.deepcopy(team_labels_df[(team_labels_df.Year >= 2002) & (team_labels_df.Year <= train_split_year)])
    train_labels.reset_index(drop=True, inplace=True)

    # Valid
    valid_record = copy.deepcopy(NFL_record_df[(NFL_record_df.Year > train_split_year) & (NFL_record_df.Year <= valid_split_year)])
    valid_record.reset_index(drop=True, inplace=True)

    valid_team_features = copy.deepcopy(team_features[(team_features.Year > train_split_year) & (team_features.Year <= valid_split_year)])
    valid_team_features.reset_index(drop=True, inplace=True)

    valid_labels = copy.deepcopy(team_labels_df[(team_labels_df.Year > train_split_year) & (team_labels_df.Year <= valid_split_year)])
    valid_labels.reset_index(drop=True, inplace=True)

    # Test
    test_record = copy.deepcopy(NFL_record_df[(NFL_record_df.Year > valid_split_year) & (NFL_record_df.Year <= 2019)])
    test_record.reset_index(drop=True, inplace=True)

    test_team_features = copy.deepcopy(team_features[(team_features.Year > valid_split_year) & (team_features.Year <= 2019)])
    test_team_features.reset_index(drop=True, inplace=True)

    test_labels = copy.deepcopy(team_labels_df[(team_labels_df.Year > valid_split_year) & (team_labels_df.Year <= 2019)])
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


    print("Normalizing coach features...")
    normalize_features = coach_feature_names
    train_normalized_feature_extracted = np.array(train_record[normalize_features])
    valid_normalized_feature_extracted = np.array(valid_record[normalize_features])
    test_normalized_feature_extracted = np.array(test_record[normalize_features])

    train_normalized_feature, valid_normalized_feature, test_normalized_feature = normalize(train_normalized_feature_extracted, valid_normalized_feature_extracted, test_normalized_feature_extracted)

    train_record.loc[:,normalize_features] = train_normalized_feature
    valid_record.loc[:,normalize_features] = valid_normalized_feature
    test_record.loc[:,normalize_features] = test_normalized_feature

    print("Normalizing team features...")
    if feature_set != 0:
        train_team_feature_extracted = np.array(train_team_features[team_feature_names])
        valid_team_feature_extracted = np.array(valid_team_features[team_feature_names])
        test_team_feature_extracted = np.array(test_team_features[team_feature_names])

        train_normalized_team_features, valid_normalized_team_features, test_normalized_team_features = normalize(train_team_feature_extracted, valid_team_feature_extracted, test_team_feature_extracted)

        train_team_features.loc[:,team_feature_names] = train_normalized_team_features
        valid_team_features.loc[:,team_feature_names] = valid_normalized_team_features
        test_team_features.loc[:,team_feature_names] = test_normalized_team_features

    # generate features, labels, and salary vectors for training set
    # in the same order of the train team graph
    train_team_G, normalized_train_f, train_team_labels_dict, train_team_features_dict= prepare_features_labels(train_team_G, train_record, train_labels, train_team_features, coach_feature_names, team_feature_names, "failure", feature_set)

    # generate features, labels, and salary vectors for valid set
    # in the same order of the valid team graph
    valid_team_G, normalized_valid_f, valid_team_labels_dict, valid_team_features_dict= prepare_features_labels(valid_team_G, valid_record, valid_labels, valid_team_features, coach_feature_names, team_feature_names, "failure", feature_set)

    # generate features, labels, and salary vectors for test set
    # in the same order of the test team graph
    test_team_G, normalized_test_f, test_team_labels_dict, test_team_features_dict= prepare_features_labels(test_team_G, test_record, test_labels, test_team_features, coach_feature_names, team_feature_names, "failure", feature_set)

    ########################################################
    ### Hierarchical team embedding using attention mechanism.
    ########################################################
    print("Training model...")
    loss = nn.BCEWithLogitsLoss() # includes sigmoid layer
    epochs = 3000

    torch.manual_seed(seed)
    model = HierGATTeamEmb(in_dim=normalized_train_f.shape[1],
            emb_dim=emb_dim, num_heads=num_heads, 
            num_team_features=len(team_feature_names),
            merge="cat",
            pos_connect=pos_connect)

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Early stopping
    stopper = EarlyStopping(patience=50)

    train_loss_arr = np.zeros((epochs))
    valid_loss_arr = np.zeros((epochs))
    test_loss_arr = np.zeros((epochs))
    train_auc_arr = np.zeros((epochs))
    valid_auc_arr = np.zeros((epochs))
    test_auc_arr = np.zeros((epochs))

    for epoch in tqdm(range(epochs)):
        print("Epoch {}".format(epoch))
        model.train()

        optimizer.zero_grad()
        
        train_y_hat, train_labels = model(train_team_G, normalized_train_f, train_team_features_dict, train_team_labels_dict)
        train_loss = loss(train_y_hat, train_labels)
        train_loss_arr[epoch] = train_loss

        # get the train predictions
        train_prob = torch.sigmoid(train_y_hat)
        train_pred = torch.round(train_prob)
        train_auc = round(roc_auc_score(train_labels.detach().numpy(), train_prob.detach().numpy()), 3)
        train_auc_arr[epoch] = train_auc

        train_loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()

            # predict on the valid set
            valid_y_hat, valid_labels = model(valid_team_G, normalized_valid_f, valid_team_features_dict, valid_team_labels_dict)
            valid_loss = loss(valid_y_hat, valid_labels)
            valid_loss_arr[epoch] = valid_loss

            valid_prob = torch.sigmoid(valid_y_hat)
            valid_pred = torch.round(valid_prob)
            valid_auc = round(roc_auc_score(valid_labels.detach().numpy(), valid_prob.detach().numpy()), 3)
            valid_auc_arr[epoch] = valid_auc
            
            # predict on the test set
            test_y_hat, test_labels = model(test_team_G, normalized_test_f, test_team_features_dict, test_team_labels_dict)
            test_loss = loss(test_y_hat, test_labels)
            test_loss_arr[epoch] = test_loss

            test_prob = torch.sigmoid(test_y_hat)
            test_pred = torch.round(test_prob)
            test_auc = round(roc_auc_score(test_labels.detach().numpy(), test_prob.detach().numpy()), 3)
            test_auc_arr[epoch] = test_auc

            print("Train Loss: {:.3f}, auc: {:.3f}\nValid Loss: {:.3f}, auc: {:.3f}\nTest Loss: {:.3f}, auc: {:.3f}".format(train_loss,train_auc, valid_loss, valid_auc, test_loss, test_auc))
            counter, stop = stopper.step(valid_loss, model)
            if counter == 1:
                remember_epoch = epoch - 1
            if stop:
                break

    print("Performance summary: (feature set {}, seed {})".format(feature_set, seed))
    print("*Stopped at epoch {}".format(remember_epoch))
    print("Train Loss: {:.3f}, AUC: {:.3f}\nValid Loss: {:.3f}, AUC: {:.3f}\nTest Loss: {:.3f}, AUC: {:.3f}".format(train_loss_arr[remember_epoch], train_auc_arr[remember_epoch], valid_loss_arr[remember_epoch], valid_auc_arr[remember_epoch], test_loss_arr[remember_epoch], test_auc_arr[remember_epoch]))

