
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
from attn_aggregation import *
from utils import *

def prepare_features_labels(team_G, coach_features_df, team_labels_df, team_features_df, coach_feature_names, team_feature_names, label_name):

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

            # insert team features
            team_feature = team_features_df[(team_features_df.Year == year) & (team_features_df.Team==team)][team_feature_names]
            team_features_dict[(year, team)] = torch.Tensor(team_feature.values)

            team_idx += 1

    return team_G, coach_features, team_labels_dict, team_features_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-collab', '--collab', default=False, type=bool, help="if the coach previous collaborations (node embedding) should be considered as the features")
    parser.add_argument('-diversity', '--diversity', default=False, type=bool)
    parser.add_argument('-emb_dim', '--emb_dim', type=int)
    parser.add_argument('-repeats', '--repeats', default=1, type=int)
    parser.add_argument('-seed', '--seed', default=1, type=int)

    args = parser.parse_args()
    collab = args.collab
    diversity = args.diversity
    emb_dim = args.emb_dim
    repeats = args.repeats
    seed = args.seed

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
    cumul_emb_features = NFL_record_df.columns[NFL_record_df.columns.str.contains("cumul_emb")].tolist()

    if collab == True:
        coach_feature_names = basic_features + cumul_emb_features
    else:
        coach_feature_names = basic_features

    #########################################################
    ## Generating team features
    #########################################################
    if collab == True:
        team_diversity_df = generate_team_diversity_feature(NFL_record_df, cumul_emb_features)
        team_features = team_salary_df.merge(team_diversity_df, how="left", on=["Team", "Year"])
        if diversity == True:
            team_feature_names = ["Salary_Rank", "Max_Emb_Similarity", "Mean_Emb_Similarity"]
        else:
            team_feature_names = ["Salary_Rank"]
    else:
        team_features = team_salary_df
        team_feature_names = ["Salary_Rank"]

    #################################################################
    ## Construct team network
    #################################################################
    # Separate train and test datasets
    # Train
    train_split_year, valid_split_year = 2015, 2017
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

    # Construct fully connected network of coaches in the same team
    # No hierarchical. No direction
    print("** Constructing train team network... **")
    train_team_G = construct_seasonal_colleague_network(train_record, 2002, 2015)
    print(nx.info(train_team_G))
    num_train_components = nx.number_strongly_connected_components(train_team_G)
    print("The number of train teams (components): {}".format(num_train_components))

    valid_team_G = construct_seasonal_colleague_network(valid_record, 2016, 2017)
    print(nx.info(valid_team_G))
    num_valid_components = nx.number_strongly_connected_components(valid_team_G)
    print("The number of valid teams (components): {}".format(num_valid_components))

    test_team_G = construct_seasonal_colleague_network(test_record, 2018, 2019)
    print(nx.info(test_team_G))
    num_test_components = nx.number_strongly_connected_components(test_team_G)
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
    stacked_features = np.vstack((train_normalized_feature_extracted,valid_normalized_feature_extracted,test_normalized_feature_extracted))
    means = stacked_features.mean(axis=0)
    stds = stacked_features.std(axis=0)

    train_normalized_feature = (train_normalized_feature_extracted - means) / stds
    valid_normalized_feature = (valid_normalized_feature_extracted - means) / stds
    test_normalized_feature = (test_normalized_feature_extracted - means) / stds

    train_record.loc[:,normalize_features] = train_normalized_feature
    valid_record.loc[:,normalize_features] = valid_normalized_feature
    test_record.loc[:,normalize_features] = test_normalized_feature

    print("Normalizing team features...")
    train_team_feature_extracted = np.array(train_team_features[team_feature_names])
    valid_team_feature_extracted = np.array(valid_team_features[team_feature_names])
    test_team_feature_extracted = np.array(test_team_features[team_feature_names])

    stacked_team_features = np.vstack((train_team_feature_extracted, valid_team_feature_extracted, test_team_feature_extracted))
    means = stacked_team_features.mean(axis=0)
    stds = stacked_team_features.std(axis=0)

    train_normalized_team_features = (train_team_feature_extracted - means) / stds
    valid_normalized_team_features = (valid_team_feature_extracted - means) / stds
    test_normalized_team_features = (test_team_feature_extracted - means) / stds

    train_team_features.loc[:,team_feature_names] = train_normalized_team_features
    valid_team_features.loc[:,team_feature_names] = valid_normalized_team_features
    test_team_features.loc[:,team_feature_names] = test_normalized_team_features

    # generate features, labels, and salary vectors for training set
    # in the same order of the train team graph
    train_team_G, normalized_train_f, train_team_labels_dict, train_team_features_dict = prepare_features_labels(train_team_G, train_record, train_labels, train_team_features, coach_feature_names, team_feature_names, "failure")

    # generate features, labels, and salary vectors for valid set
    # in the same order of the valid team graph
    valid_team_G, normalized_valid_f, valid_team_labels_dict, valid_team_features_dict= prepare_features_labels(valid_team_G, valid_record, valid_labels, valid_team_features, coach_feature_names, team_feature_names, "failure")

    # generate features, labels, and salary vectors for test set
    # in the same order of the test team graph
    test_team_G, normalized_test_f, test_team_labels_dict, test_team_features_dict= prepare_features_labels(test_team_G, test_record, test_labels, test_team_features, coach_feature_names, team_feature_names, "failure")

    ########################################################
    ### Hierarchical team embedding using attention mechanism.
    ########################################################
    print("Training model...")
    loss = nn.BCEWithLogitsLoss()
    epochs = 3000

    patience = 7
    train_loss_arr = np.zeros((epochs))
    valid_loss_arr = np.zeros((epochs))
    test_loss_arr = np.zeros((epochs))
    train_accuracy_arr = np.zeros((epochs))
    valid_accuracy_arr = np.zeros((epochs))
    test_accuracy_arr = np.zeros((epochs))
    train_auc_arr = np.zeros((epochs))
    valid_auc_arr = np.zeros((epochs))
    test_auc_arr = np.zeros((epochs))

    torch.manual_seed(seed)
    auc_dict = {"train":[], "valid":[], "test":[]}
    accuracy_dict = {"train":[], "valid":[], "test":[]}
    emb_dims = [emb_dim]
    for emb_dim in emb_dims:
        print("Embedding dimension: {}".format(emb_dim))
        repeat_performances = {"train":{"loss":[], "accuracy": [], "auc":[]},\
                                "valid": {"loss":[], "accuracy": [], "auc":[]},\
                                "test": {"loss":[], "accuracy": [], "auc":[]}}
        for repeat in range(repeats):
            model = AttnAggLayer(in_dim=normalized_train_f.shape[1],
                                emb_dim=emb_dim,
                                num_team_features=len(team_feature_names))

            # create optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            # Early stopping
            stopper = EarlyStopping(patience=100)
            for epoch in tqdm(range(epochs)):
                print("Epoch {}".format(epoch))
                model.train()

                optimizer.zero_grad()

                train_g, train_y_hat, train_labels = model(train_team_G, normalized_train_f, train_team_features_dict, train_team_labels_dict, num_train_components)
                train_loss = loss(train_y_hat, train_labels)
                train_loss_arr[epoch] = train_loss

                # get the train predictions
                train_prob = torch.sigmoid(train_y_hat)
                train_pred = torch.round(train_prob)
                train_accuracy = (train_pred == train_labels).sum().item() / train_labels.shape[0]
                train_auc = round(roc_auc_score(train_labels.detach().numpy(), train_prob.detach().numpy()), 3)
                train_accuracy_arr[epoch] = train_accuracy
                train_auc_arr[epoch] = train_auc

                train_loss.backward()
                optimizer.step()

                with torch.no_grad():
                    model.eval()

                    # predict on the valid set
                    valid_g, valid_y_hat, valid_labels = model(valid_team_G, normalized_valid_f, valid_team_features_dict, valid_team_labels_dict, num_valid_components)
                    valid_loss = loss(valid_y_hat, valid_labels)
                    valid_loss_arr[epoch] = valid_loss

                    valid_prob = torch.sigmoid(valid_y_hat)
                    valid_pred = torch.round(valid_prob)
                    valid_accuracy = (valid_pred == valid_labels).sum().item() / valid_labels.shape[0]
                    valid_auc = round(roc_auc_score(valid_labels.detach().numpy(), valid_prob.detach().numpy()), 3)
                    valid_accuracy_arr[epoch] = valid_accuracy
                    valid_auc_arr[epoch] = valid_auc
                    
                    # predict on the test set
                    test_g, test_y_hat, test_labels = model(test_team_G, normalized_test_f, test_team_features_dict, test_team_labels_dict, num_test_components)
                    test_loss = loss(test_y_hat, test_labels)
                    test_loss_arr[epoch] = test_loss

                    test_prob = torch.sigmoid(test_y_hat)
                    test_pred = torch.round(test_prob)
                    test_accuracy = (test_pred == test_labels).sum().item() / test_labels.shape[0]
                    test_auc = round(roc_auc_score(test_labels.detach().numpy(), test_prob.detach().numpy()), 3)
                    test_accuracy_arr[epoch] = test_accuracy
                    test_auc_arr[epoch] = test_auc

                    print("Train Loss: {:.3f}, auc: {:.3f}\nValid Loss: {:.3f}, auc: {:.3f}\nTest Loss: {:.3f}, auc: {:.3f}".format(train_loss,train_auc, valid_loss, valid_auc, test_loss, test_auc))

                # if (epoch > patience and np.argmin(valid_loss_arr[epoch-patience:epoch]) == 0) or (epoch == epochs-1):
                if stopper.step(valid_loss, model):
                    break

            repeat_performances["train"]["loss"].append(float(train_loss))
            repeat_performances["train"]["accuracy"].append(train_accuracy)
            repeat_performances["train"]["auc"].append(train_auc)
            repeat_performances["valid"]["loss"].append(float(valid_loss))
            repeat_performances["valid"]["accuracy"].append(valid_accuracy)
            repeat_performances["valid"]["auc"].append(valid_auc)
            repeat_performances["test"]["loss"].append(float(test_loss))
            repeat_performances["test"]["accuracy"].append(test_accuracy)
            repeat_performances["test"]["auc"].append(test_auc)

        avg_train_loss = statistics.mean(repeat_performances["train"]["loss"])
        avg_train_accuracy = statistics.mean(repeat_performances["train"]["accuracy"])
        avg_train_auc = statistics.mean(repeat_performances["train"]["auc"])
        avg_valid_loss = statistics.mean(repeat_performances["valid"]["loss"])
        avg_valid_accuracy = statistics.mean(repeat_performances["valid"]["accuracy"])
        avg_valid_auc = statistics.mean(repeat_performances["valid"]["auc"])
        avg_test_loss = statistics.mean(repeat_performances["test"]["loss"])
        avg_test_accuracy = statistics.mean(repeat_performances["test"]["accuracy"])
        avg_test_auc = statistics.mean(repeat_performances["test"]["auc"])

        auc_dict["train"].append(avg_train_auc)
        auc_dict["valid"].append(avg_valid_auc)
        auc_dict["test"].append(avg_test_auc)




