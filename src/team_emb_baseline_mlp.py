'''
Aggregate coach features non-hierarchically using mean aggregator.
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

    args = parser.parse_args()
    feature_set = args.feature_set
    train_split_year = args.train_split_year
    valid_split_year = args.valid_split_year

    #################################################################
    # Load datasets
    NFL_coach_record_filename = "../datasets/NFL_Coach_Data_with_features_collaball_hierTrue.csv"
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
    # aggregate coaches' features in the same team.
    print("Generating team embedding, team_features, and labels...")
    print("Train")
    train_team_emb, train_team_feature, train_labels = simple_aggregate_features(train_record, train_team_features, train_labels, coach_feature_names, team_feature_names, "failure")
    print("Validation")
    valid_team_emb, valid_team_feature, valid_labels = simple_aggregate_features(valid_record, valid_team_features, valid_labels, coach_feature_names, team_feature_names, "failure")
    print("Test")
    test_team_emb, test_team_feature, test_labels = simple_aggregate_features(test_record, test_team_features, test_labels, coach_feature_names, team_feature_names, "failure")

    # combine aggregated features and team features
    train_x = np.concatenate((train_team_emb, train_team_feature), axis=1)
    valid_x = np.concatenate((valid_team_emb, valid_team_feature), axis=1)
    test_x = np.concatenate((test_team_emb, test_team_feature), axis=1)

    ## MLP
    print("*** MLP: ")

    # normalize
    normalized_train_x, normalized_valid_x, normalized_test_x = normalize(train_x, valid_x, test_x)

    train_labels = torch.Tensor(train_labels)
    valid_labels = torch.Tensor(valid_labels)
    test_labels = torch.Tensor(test_labels)

    if feature_set == 2 or feature_set == 3:
        hidden_nodes = [5, 10, 15, 20, 25, 30]
    elif feature_set == 0 or feature_set == 1:
        hidden_nodes = [3, 5, 7, 10]
    mlp_auc_dict = mlp(hidden_nodes, normalized_train_x, train_labels, normalized_valid_x, valid_labels, normalized_test_x, test_labels)

