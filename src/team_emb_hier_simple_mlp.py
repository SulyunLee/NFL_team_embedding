
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-collab', '--collab', default=False, type=bool, help="if the coach previous collaborations (node embedding) should be considered as the features")
    parser.add_argument('-diversity', '--diversity', default=False, type=bool)

    args = parser.parse_args()
    collab = args.collab
    diversity = args.diversity

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

    # split coach record, salary, and label into train, validation, test set
    train_split_year = 2015
    valid_split_year = 2017

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
    print("Generating team embedding, salary, and labels...")
    print("Train")
    train_team_emb, train_team_feature, train_labels = hierarchical_average_features(train_record, train_team_features, train_labels, coach_feature_names, team_feature_names, "failure")
    print("Validation")
    valid_team_emb, valid_team_feature, valid_labels = hierarchical_average_features(valid_record, valid_team_features, valid_labels, coach_feature_names, team_feature_names, "failure")
    print("Test")
    test_team_emb, test_team_feature, test_labels = hierarchical_average_features(test_record, test_team_features, test_labels, coach_feature_names, team_feature_names, "failure")

    # concatenage team features to team embedding
    train_x = np.concatenate((train_team_emb, train_team_feature), axis=1)
    valid_x = np.concatenate((valid_team_emb, valid_team_feature), axis=1)
    test_x = np.concatenate((test_team_emb, test_team_feature), axis=1)

    
    ## MLP
    print("*** MLP: ")

    # normalize
    combined_x = np.zeros((train_x.shape[0] + valid_x.shape[0] + test_x.shape[0], train_x.shape[1]))
    combined_x[:train_x.shape[0]] = train_x
    combined_x[train_x.shape[0]:train_x.shape[0]+valid_x.shape[0],:] = valid_x
    combined_x[train_x.shape[0]+valid_x.shape[0]:,:] = test_x

    means = combined_x.mean(axis=0)
    stds = combined_x.std(axis=0)

    normalized_train_x = torch.Tensor((train_x - means) / stds)
    normalized_valid_x = torch.Tensor((valid_x - means) / stds)
    normalized_test_x = torch.Tensor((test_x - means) / stds)

    train_labels = torch.Tensor(train_labels)
    valid_labels = torch.Tensor(valid_labels)
    test_labels = torch.Tensor(test_labels)

    hidden_nodes = [2, 4, 8, 16, 32]
    mlp_auc_dict = mlp(hidden_nodes, normalized_train_x, train_labels, normalized_valid_x, valid_labels, normalized_test_x, test_labels)

