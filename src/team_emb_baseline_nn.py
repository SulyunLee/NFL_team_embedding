'''
Aggregate coach features non-hierarchically using neural network aggregator (fully-connected)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-feature_set', '--feature_set', type=int, help="Feature set number\n(0: Basic features, 1: Basic features + salary, 2: Basic features & node embedding + salary, 3: Basic features & node embedding + salary + diversity)")
    parser.add_argument('-train_split_year', '--train_split_year', type=int, help="Maximum year for training set")
    parser.add_argument('-valid_split_year', '--valid_split_year', type=int, help="Maximum year for validation set")
    parser.add_argument('-seed', '--seed', default=0, type=int, help="Random seed")

    args = parser.parse_args()
    feature_set = args.feature_set
    train_split_year = args.train_split_year
    valid_split_year = args.valid_split_year
    seed = args.seed

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

    #########################################################
    ## Generate team embeddings...
    #########################################################
    print("Generating team embedding, team_features, and labels...")
    # aggregate coaches' features in the same team (avg).
    print("Train")
    train_team_emb, train_team_feature, train_labels = simple_aggregate_features(train_record, train_team_features, train_labels, coach_feature_names, team_feature_names, "failure")
    print("Validation")
    valid_team_emb, valid_team_feature, valid_labels = simple_aggregate_features(valid_record, valid_team_features, valid_labels, coach_feature_names, team_feature_names, "failure")
    print("Test")
    test_team_emb, test_team_feature, test_labels = simple_aggregate_features(test_record, test_team_features, test_labels, coach_feature_names, team_feature_names, "failure")

    ## normalize
    # averaged feature
    normalized_train_x, normalized_valid_x, normalized_test_x = normalize(train_team_emb, valid_team_emb, test_team_emb)

    # team feature
    if feature_set != 0:
        normalized_train_team_feature, normalized_valid_team_feature, normalized_test_team_feature = normalize(train_team_feature, valid_team_feature, test_team_feature)
    else:
        normalized_train_team_feature = normalized_valid_team_feature = normalized_test_team_feature = torch.Tensor()


    # Convert labels to tensors
    train_labels = torch.Tensor(train_labels).view(train_labels.shape[0], 1)
    valid_labels = torch.Tensor(valid_labels).view(valid_labels.shape[0], 1)
    test_labels = torch.Tensor(test_labels).view(test_labels.shape[0],1)

    # Modeling
    print("Training model...")
    loss = nn.BCEWithLogitsLoss()
    epochs = 100000

    # dictionaries that store average auc and accuracy for each hidden node
    torch.manual_seed(seed)
    model = Nonhier_NN(coach_feature_dim=len(coach_feature_names),
                        team_feature_dim=len(team_feature_names),
                        output_dim=1,
                        feature_set=feature_set)

    # optimizer
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

        train_y_hat = model(normalized_train_x, normalized_train_team_feature)
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

            # predict on the validation set
            valid_y_hat = model(normalized_valid_x, normalized_valid_team_feature)
            valid_loss = loss(valid_y_hat, valid_labels)
            valid_loss_arr[epoch] = valid_loss

            # get the valid predictions
            valid_prob = torch.sigmoid(valid_y_hat)
            valid_pred = torch.round(valid_prob)
            valid_auc = round(roc_auc_score(valid_labels.detach().numpy(), valid_prob.detach().numpy()), 3)
            valid_auc_arr[epoch] = valid_auc

            # predict on the test set
            test_y_hat = model(normalized_test_x, normalized_test_team_feature)
            test_loss = loss(test_y_hat, test_labels)
            test_loss_arr[epoch] = test_loss

            # get the test predictions
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

    print("Performance summary (feature set {}, seed {}):".format(feature_set, seed))
    print("*Stopped at epoch {}".format(remember_epoch))
    print("Train Loss: {:.3f}, AUC: {:.3f}\nValid Loss: {:.3f}, AUC: {:.3f}\nTest Loss: {:.3f}, AUC: {:.3f}".format(train_loss_arr[remember_epoch], train_auc_arr[remember_epoch], valid_loss_arr[remember_epoch], valid_auc_arr[remember_epoch], test_loss_arr[remember_epoch], test_auc_arr[remember_epoch]))


