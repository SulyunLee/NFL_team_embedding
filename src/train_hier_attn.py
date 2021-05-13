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
from construct_team_network_func import *
from hier_gat import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-collab', '--collab', default=False, type=bool, help="if the coach previous collaborations (node embedding) should be considered as the features")
    parser.add_argument('-team_emb_size', '--team_emb_size', default=100, type=int, help="the number of dimensions for the team embedding")
    parser.add_argument('-epochs', '--epochs', default=100, type=int, help="the number of epochs during training")
    # parser.add_argument('-dropout', '--dropout', default=False, type=bool, help="if the model uses dropout during training")

    args = parser.parse_args()
    team_emb_size = args.team_emb_size
    collab = args.collab
    epochs = args.epochs
    # dropout = args.dropout
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
    team_salary_df = team_salary_df.dropna(subset=["Salary_Rank"])

    # only select NFL records with salary data available
    salary_avail_arr = np.zeros((NFL_record_df.shape[0])).astype(int)
    for idx, row in NFL_record_df.iterrows():
        year = row.Year
        team = row.Team
        salary_data = team_salary_df[(team_salary_df.Team == team.replace('(NFL)','').strip()) & (team_salary_df.Year == year)]
        if salary_data.shape[0] > 0:
            salary_avail_arr[idx] = 1

    NFL_record_df = NFL_record_df.assign(salary_avail = salary_avail_arr)

    NFL_record_df = NFL_record_df[NFL_record_df.salary_avail == 1]
    NFL_record_df.reset_index(drop=True, inplace=True)


    # Define column names to be used as coach features
    basic_features = ["TotalYearsInNFL", "Past5yrsWinningPerc_best", "Past5yrsWinningPerc_avg", "HC", "Coord"]
    cumul_emb_features = NFL_record_df.columns[NFL_record_df.columns.str.contains("cumul_emb")].tolist()


    #################################################################
    ### Construct team network using all seasons.
    #################################################################
    # Separate train and test datasets
    train_split_year, valid_split_year = 2015, 2017
    train_record = NFL_record_df[NFL_record_df.Year <= train_split_year]
    train_record.reset_index(drop=True, inplace=True)

    valid_record = NFL_record_df[(NFL_record_df.Year > train_split_year) & (NFL_record_df.Year <= valid_split_year)]
    valid_record.reset_index(drop=True, inplace=True)

    test_record = NFL_record_df[(NFL_record_df.Year > valid_split_year)]
    test_record.reset_index(drop=True, inplace=True)
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
    train_id_record_dict, train_team_G = construct_fullseason_mentorship_network(train_record)
    print(nx.info(train_team_G))
    train_components = nx.weakly_connected_components(train_team_G)
    num_train_components = len(list(train_components))
    print("The number of train teams (components): {}".format(num_train_components))

    print("** Constructing validation team network... **")
    valid_id_record_dict, valid_team_G = construct_fullseason_mentorship_network(valid_record)
    print(nx.info(valid_team_G))
    valid_components = nx.weakly_connected_components(valid_team_G)
    num_valid_components = len(list(valid_components))
    print("The number of valid teams (components): {}".format(num_valid_components))

    print("** Constructing test team network... **")
    test_id_record_dict, test_team_G = construct_fullseason_mentorship_network(test_record)
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
    # adding feature vectors as the node attributes in entire_team_G
    if collab == True:
        feature_names = basic_features + cumul_emb_features
    else:
        feature_names = basic_features

    train_feature_extracted = np.array(train_record[feature_names])
    valid_feature_extracted = np.array(valid_record[feature_names])
    test_feature_extracted = np.array(test_record[feature_names])

    # generate features, labels, and salary vectors for training set
    # in the same order of the train team graph
    train_features = torch.zeros((train_feature_extracted.shape[0], train_feature_extracted.shape[1]))
    train_labels = torch.zeros((num_train_components, 1))
    train_salary = torch.zeros((num_train_components, 1))
    team_idx = 0
    for idx, node in enumerate(train_team_G.nodes()):
        # save the raw featuresr of each coach as the node attribute "f"
        train_team_G.nodes[node]["f"] = torch.Tensor(train_feature_extracted[node,:])
        # The order of features is the same as the order of graph nodes.
        train_features[idx,:] = torch.Tensor(train_feature_extracted[node,:])
        if train_team_G.nodes[node]["final_position"] == "HC":
            # insert labels
            year = train_team_G.nodes[node]["Year"]
            team = train_team_G.nodes[node]["Team"]
            label = team_labels_df[(team_labels_df.Year==year) & (team_labels_df.Team==team.replace('(NFL)','').strip())].failure
            train_labels[team_idx] = int(label)

            # insert salary
            salary_rank = team_salary_df[(team_salary_df.Year == year) & (team_salary_df.Team==team.replace('(NFL)','').strip())].Salary_Rank
            train_salary[team_idx] = int(salary_rank)
            team_idx += 1

    # generate features, labels, and salary vectors for valid set
    # in the same order of the valid team graph
    valid_features = torch.zeros((valid_feature_extracted.shape[0], valid_feature_extracted.shape[1]))
    valid_labels = torch.zeros((num_valid_components, 1))
    valid_salary = torch.zeros((num_valid_components, 1))
    team_idx = 0
    for idx, node in enumerate(valid_team_G.nodes()):
        # save the raw featuresr of each coach as the node attribute "f"
        valid_team_G.nodes[node]["f"] = torch.Tensor(valid_feature_extracted[node,:])
        # The order of features is the same as the order of graph nodes.
        valid_features[idx,:] = torch.Tensor(valid_feature_extracted[node,:])
        if valid_team_G.nodes[node]["final_position"] == "HC":
            # insert labels
            year = valid_team_G.nodes[node]["Year"]
            team = valid_team_G.nodes[node]["Team"]
            label = team_labels_df[(team_labels_df.Year==year) & (team_labels_df.Team==team.replace('(NFL)','').strip())].failure
            valid_labels[team_idx] = int(label)

            # insert salary
            salary_rank = team_salary_df[(team_salary_df.Year == year) & (team_salary_df.Team==team.replace('(NFL)','').strip())].Salary_Rank
            valid_salary[team_idx] = int(salary_rank)
            team_idx += 1

    # generate features, labels, and salary vectors for test set
    # in the same order of the test team graph
    test_features = torch.zeros((test_feature_extracted.shape[0], test_feature_extracted.shape[1]))
    test_labels = torch.zeros((num_test_components, 1))
    test_salary = torch.zeros((num_test_components, 1))
    team_idx = 0
    for idx, node in enumerate(test_team_G.nodes()):
        # save the raw featuresr of each coach as the node attribute "f"
        test_team_G.nodes[node]["f"] = torch.Tensor(test_feature_extracted[node,:])
        # The order of features is the same as the order of graph nodes.
        test_features[idx,:] = torch.Tensor(test_feature_extracted[node,:])
        if test_team_G.nodes[node]["final_position"] == "HC":
            # insert labels
            year = test_team_G.nodes[node]["Year"]
            team = test_team_G.nodes[node]["Team"]
            label = team_labels_df[(team_labels_df.Year==year) & (team_labels_df.Team==team.replace('(NFL)','').strip())].failure
            test_labels[team_idx] = int(label)

            # insert salary
            salary_rank = team_salary_df[(team_salary_df.Year == year) & (team_salary_df.Team==team.replace('(NFL)','').strip())].Salary_Rank
            test_salary[team_idx] = int(salary_rank)
            team_idx += 1
    
    ### Normalize the input node features
    num_train_record = train_features.shape[0]
    num_valid_record = valid_features.shape[0]
    num_test_record = test_features.shape[0]

    # select only numerical features
    numerical_features = [x for x in feature_names if (x != "HC") and (x != "Coord")]
    numerical_feature_idx = [i for i, x in enumerate(feature_names) if x in numerical_features]

    combined_features = torch.zeros((num_train_record + num_valid_record + num_test_record, len(numerical_feature_idx)))
    combined_features[:num_train_record,:] = train_features[:,numerical_feature_idx]
    combined_features[num_train_record:num_train_record + num_valid_record,:] = valid_features[:,numerical_feature_idx]
    combined_features[num_train_record+num_valid_record:,:] = test_features[:,numerical_feature_idx]

    means = combined_features.mean(dim=0, keepdim=True)
    stds = combined_features.std(dim=0, keepdim=True)


    # train set
    normalized_train_f = copy.deepcopy(train_features)
    normalized_train_f[:,numerical_feature_idx] = (train_features[:,numerical_feature_idx] - means) / stds 
    # valid set 
    normalized_valid_f = copy.deepcopy(valid_features)
    normalized_valid_f[:,numerical_feature_idx] = (valid_features[:,numerical_feature_idx] - means) / stds 
    # test set 
    normalized_test_f = copy.deepcopy(test_features)
    normalized_test_f[:,numerical_feature_idx] = (test_features[:,numerical_feature_idx] - means) / stds 

    ### Normalize the salary data features
    # train set salary
    num_train_salary = train_salary.shape[0]
    num_valid_salary = valid_salary.shape[0]
    num_test_salary = test_salary.shape[0]

    combined_salary = torch.zeros((num_train_salary + num_valid_salary + num_test_salary, 1))
    combined_salary[:num_train_salary,:] = train_salary
    combined_salary[num_train_salary:num_train_salary + num_valid_salary,:] = valid_salary
    combined_salary[num_train_salary+num_valid_salary:,:] = test_salary

    salary_means = combined_salary.mean(dim=0, keepdim=True)
    salary_stds = combined_salary.std(dim=0, keepdim=True)

    # train salary
    normalized_train_salary = (train_salary - salary_means) / salary_stds
    # valid set salary 
    normalized_valid_salary = (valid_salary - salary_means) / salary_stds
    # test set salary 
    normalized_test_salary = (test_salary - salary_means) / salary_stds

    ########################################################
    ### Hierarchical team embedding using attention mechanism.
    ########################################################
    print("Training model...")
    loss = nn.BCELoss()
    # create the model
    model = HierGATLayer(in_dim=normalized_train_f.shape[1],
                        emb_dim=team_emb_size, dropout=dropout)
    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


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

    for epoch in tqdm(range(epochs)):
        model.train()
        train_g, train_y_hat = model(train_team_G, normalized_train_f, normalized_train_salary)
        train_loss = loss(train_y_hat[:,1], train_labels.view(train_y_hat[:,1].shape))
        train_loss_arr[epoch] = train_loss

        # _, train_pred = torch.max(train_y_hat, dim=1)
        # train_pred = train_pred.type('torch.FloatTensor')
        # train_pred = train_pred.view(train_labels.size())
        # train_accuracy = (train_pred == train_labels).sum().item() / train_labels.shape[0]

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            # get the train predictions
            _, train_pred = torch.max(train_y_hat, dim=1)
            train_pred = train_pred.type('torch.FloatTensor')
            train_pred = train_pred.view(train_labels.size())
            train_accuracy = (train_pred == train_labels).sum().item() / train_labels.shape[0]
            train_auc = round(roc_auc_score(train_labels.detach().numpy(), train_y_hat[:,1].detach().numpy()), 3)

            train_accuracy_arr[epoch] = train_accuracy
            train_auc_arr[epoch] = train_auc

            # predict on the valid set
            valid_g, valid_y_hat = model(valid_team_G, normalized_valid_f, normalized_valid_salary)
            valid_loss = loss(valid_y_hat[:,1], valid_labels.view(valid_y_hat[:,1].shape))
            valid_loss_arr[epoch] = valid_loss

            _, valid_pred = torch.max(valid_y_hat, dim=1)
            valid_pred = valid_pred.type('torch.FloatTensor')
            valid_pred = valid_pred.view(valid_labels.size())
            valid_accuracy = (valid_pred == valid_labels).sum().item() / valid_labels.shape[0]
            valid_auc = round(roc_auc_score(valid_labels, valid_y_hat[:,1]), 3)
            
            valid_accuracy_arr[epoch] = valid_accuracy
            valid_auc_arr[epoch] = valid_auc


            # predict on the test set
            test_g, test_y_hat = model(test_team_G, normalized_test_f, normalized_test_salary)
            test_loss = loss(test_y_hat[:,1], test_labels.view(test_y_hat[:,1].shape))
            test_loss_arr[epoch] = test_loss

            _, test_pred = torch.max(test_y_hat, dim=1)
            test_pred = test_pred.type('torch.FloatTensor')
            test_pred = test_pred.view(test_labels.size())
            test_accuracy = (test_pred == test_labels).sum().item() / test_labels.shape[0]
            test_auc = round(roc_auc_score(test_labels, test_y_hat[:,1]), 3)

            test_accuracy_arr[epoch] = test_accuracy
            test_auc_arr[epoch] = test_auc

        if epoch > patience and np.argmin(valid_loss_arr[epoch-patience:epoch]) == 0:
            break
            
    print("Epoch {}:\n, Train L: {:.4f}, acc: {:.3f}, auc: {}\n Val L: {:.4f}, acc: {:.3f}, auc: {}\n Test L: {:.4f}, acc: {:.3f}, auc: {}".format(epoch, train_loss, train_accuracy, train_auc, valid_loss, valid_accuracy, valid_auc, test_loss, test_accuracy, test_auc))
        
    # save losses
    np.savez("temp_data/losses_team_emb_hierattn_collab{}_tesize{}.npz".format(collab, team_emb_size), train_loss_arr=train_loss_arr, train_accuracy_arr=train_accuracy_arr, train_auc_arr=train_auc_arr, valid_loss_arr=valid_loss_arr, valid_accuracy_arr=valid_accuracy_arr, valid_auc_arr=valid_auc_arr, test_loss_arr=test_loss_arr, test_accuracy_arr=test_accuracy_arr, test_auc_arr=test_auc_arr)

    # save model
    torch.save(model.state_dict(), "pytorch_models/team_emb_hierattn_collab{}_tesize{}_checkpoint.pth".format(collab, team_emb_size))

    
