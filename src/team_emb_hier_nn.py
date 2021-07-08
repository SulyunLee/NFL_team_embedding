'''
Aggregate coach features hierarchically using neural network aggregator (fully-connected)
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

def generate_feature_set(record_df, team_features_df, labels_df, coach_feature_names, team_feature_names, label_name):
    '''
    This function generates feature sets that are needed for hierarchical way of aggregating coach features using neural network for team embedding, as well as the team feature vectors (salary or diversity) and the labels.

    This function returns the concatenated vectors of averaged offensive position coaches' features and offensive coordinator feature, and the same with defensive and special teams. This also returns the head coach feature vectors.
    '''
    seasons = record_df[["Year","Team"]].drop_duplicates().to_dict('records')
    print("The number of seasons: {}".format(len(seasons)))

    print("Generating team embedding in hierarchical way...")
    # Initialize arrays that contain feature vectors for all teams.
    offensive_feature_arr = np.zeros((len(seasons), 2 * len(coach_feature_names)))
    defensive_feature_arr = np.zeros((len(seasons), 2 * len(coach_feature_names)))
    special_feature_arr = np.zeros((len(seasons), 2 * len(coach_feature_names)))
    hc_feature_arr = np.zeros((len(seasons), len(coach_feature_names)))

    team_feature_arr = np.zeros((len(seasons), len(team_feature_names)))
    team_labels_arr = np.zeros((len(seasons))).astype(int)
    
    # Iterate over each team
    for idx, season in enumerate(seasons):
        year = season["Year"]
        team = season["Team"]

        # collect season coaches
        coaches = record_df[(record_df.Team== team) & (record_df.Year == year)]

        ## aggregate position coach features - average
        ## Then, append to the first part of each position feature array
        # offensive position coaches
        offensive_position = coaches[coaches.final_position == "O"]
        if offensive_position.shape[0] != 0:
            offensive_position_emb = np.array(offensive_position[coach_feature_names].mean())
            offensive_feature_arr[idx, :len(coach_feature_names)] = offensive_position_emb
            
        # defensive position coaches
        defensive_position = coaches[coaches.final_position == "D"]
        if defensive_position.shape[0] != 0:
            defensive_position_emb = np.array(defensive_position[coach_feature_names].mean())
            defensive_feature_arr[idx, :len(coach_feature_names)] = defensive_position_emb

        # special team position coaches
        special_position = coaches[coaches.final_position == "S"]
        if special_position.shape[0] != 0:
            special_position_emb = np.array(special_position[coach_feature_names].mean())
            special_feature_arr[idx, :len(coach_feature_names)] = special_position_emb

        ## Aggregate position coach embedding and coordinator feature using NN.
        ## Then, append to the last part of each position feature array
        offensive_coord = coaches[coaches.final_position == "OC"]
        if offensive_coord.shape[0] != 0:
            offensive_coord_emb = np.array(offensive_coord[coach_feature_names].mean())
            offensive_feature_arr[idx, len(coach_feature_names):] = offensive_coord_emb

        defensive_coord = coaches[coaches.final_position == "DC"]
        if defensive_coord.shape[0] != 0:
            defensive_coord_emb = np.array(defensive_coord[coach_feature_names].mean())
            defensive_feature_arr[idx, len(coach_feature_names):] = defensive_coord_emb

        special_coord = coaches[coaches.final_position == "SC"]
        if special_coord.shape[0] != 0:
            special_coord_emb = np.array(special_coord[coach_feature_names].mean())
            special_feature_arr[idx, len(coach_feature_names):] = special_coord_emb

        # Get the head coach feature
        hc = coaches[coaches.final_position == "HC"]
        hc_feature_arr[idx,:] = np.array(hc[coach_feature_names])

        # Get team features
        team_features = np.array(team_features_df[(team_features_df.Team == team) & (team_features_df.Year == year)][team_feature_names])
        team_feature_arr[idx,:] = team_features

        # Get the label of the season
        label = labels_df[(labels_df.Team == team.replace('(NFL)', '').strip()) & (labels_df.Year == year)][label_name]
        team_labels_arr[idx] = int(label)

    return offensive_feature_arr, defensive_feature_arr, special_feature_arr, hc_feature_arr, team_feature_arr, team_labels_arr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-collab', '--collab', default=False, type=bool, help="if the coach previous collaborations (node embedding) should be considered as the features")
    parser.add_argument('-diversity', '--diversity', default=False, type=bool)
    parser.add_argument('-seed', '--seed', default=0, type=int, help="random seed")

    args = parser.parse_args()
    collab = args.collab
    diversity = args.diversity
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
    train_split_year = 2013
    valid_split_year = 2015

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
    train_offensive, train_defensive, train_special, train_hc, train_team_features_arr, train_labels_arr = generate_feature_set(train_record, train_team_features, train_labels, coach_feature_names, team_feature_names, "failure")
    print("Validation")
    valid_offensive, valid_defensive, valid_special, valid_hc, valid_team_features_arr, valid_labels_arr = generate_feature_set(valid_record, valid_team_features, valid_labels, coach_feature_names, team_feature_names, "failure")
    print("Test")
    test_offensive, test_defensive, test_special, test_hc, test_team_features_arr, test_labels_arr = generate_feature_set(test_record, test_team_features, test_labels, coach_feature_names, team_feature_names, "failure")

    ## Normalize
    print("Normalizing features...")
    # offensive
    normalized_train_offensive, normalized_valid_offensive, normalized_test_offensive = normalize(train_offensive, valid_offensive, test_offensive)
    # defensive
    normalized_train_defensive, normalized_valid_defensive, normalized_test_defensive = normalize(train_defensive, valid_defensive, test_defensive)
    # special
    normalized_train_special, normalized_valid_special, normalized_test_special = normalize(train_special, valid_special, test_special)
    # head coach
    normalized_train_hc, normalized_valid_hc, normalized_test_hc = normalize(train_hc, valid_hc, test_hc)

    # team features
    normalized_train_team_features, normalized_valid_team_features, normalized_test_team_features = normalize(train_team_features_arr, valid_team_features_arr, test_team_features_arr)

    # Convert labels to tensors
    train_labels_arr = torch.Tensor(train_labels_arr).view(train_labels_arr.shape[0], 1)
    valid_labels_arr = torch.Tensor(valid_labels_arr).view(valid_labels_arr.shape[0], 1)
    test_labels_arr = torch.Tensor(test_labels_arr).view(test_labels_arr.shape[0], 1)

    # Modeling
    print("Training model...")
    loss = nn.BCEWithLogitsLoss()
    epochs = 3000

    # dictionaries that store average auc and accuracy for each hidden node
    torch.manual_seed(seed)
    model = Hier_NN(coach_feature_dim = len(coach_feature_names),
                    team_feature_dim = len(team_feature_names),
                    output_dim = 1,
                    collab=collab)

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

        train_y_hat = model(normalized_train_offensive, normalized_train_defensive, normalized_train_special, normalized_train_hc, normalized_train_team_features)
        train_loss = loss(train_y_hat, train_labels_arr)
        train_loss_arr[epoch] = train_loss

        # get the train predictions
        train_prob = torch.sigmoid(train_y_hat)
        train_pred = torch.round(train_prob)
        train_auc = round(roc_auc_score(train_labels_arr.detach().numpy(), train_prob.detach().numpy()), 3)
        train_auc_arr[epoch] = train_auc

        train_loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()

            # predict on the validation set
            valid_y_hat = model(normalized_valid_offensive, normalized_valid_defensive, normalized_valid_special, normalized_valid_hc, normalized_valid_team_features)
            valid_loss = loss(valid_y_hat, valid_labels_arr)
            valid_loss_arr[epoch] = valid_loss

            valid_prob = torch.sigmoid(valid_y_hat)
            valid_pred = torch.round(valid_prob)
            valid_auc = round(roc_auc_score(valid_labels_arr.detach().numpy(), valid_prob.detach().numpy()), 3)
            valid_auc_arr[epoch] = valid_auc

            # predict on the test set
            test_y_hat = model(normalized_test_offensive, normalized_test_defensive, normalized_test_special, normalized_test_hc, normalized_test_team_features)
            test_loss = loss(test_y_hat, test_labels_arr)
            test_loss_arr[epoch] = test_loss

            test_prob = torch.sigmoid(test_y_hat)
            test_pred = torch.round(test_prob)
            test_auc = round(roc_auc_score(test_labels_arr.detach().numpy(), test_prob.detach().numpy()), 3)
            test_auc_arr[epoch] = test_auc
            
            print("Train Loss: {:.3f}, auc: {:.3f}\nValid Loss: {:.3f}, auc: {:.3f}\nTest Loss: {:.3f}, auc: {:.3f}".format(train_loss,train_auc, valid_loss, valid_auc, test_loss, test_auc))

            counter, stop = stopper.step(valid_loss, model)
            if counter == 1:
                remember_epoch = epoch - 1
            if stop:
                break

    print("Performance summary:")
    print("*Stopped at epoch {}".format(remember_epoch))
    print("Train Loss: {:.3f}, AUC: {:.3f}\nValid Loss: {:.3f}, AUC: {:.3f}\nTest Loss: {:.3f}, AUC: {:.3f}".format(train_loss_arr[remember_epoch], train_auc_arr[remember_epoch], valid_loss_arr[remember_epoch], valid_auc_arr[remember_epoch], test_loss_arr[remember_epoch], test_auc_arr[remember_epoch]))





    


