'''
This script generates team embedding of a team network using the baseline model.
The baseline model does not consider the hierarchical structure of a team, but simply
aggregates the team members' features.
'''
import pandas as pd
import numpy as np
import statistics
import argparse
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# def evaluate(true_y, pred_y, pred_prob):
def evaluate(true_y, pred_y):
    ## Accuracy
    accuracy = metrics.accuracy_score(true_y, pred_y)

    ## AUC
    # fpr, tpr, thresholds = metrics.roc_curve(true_y, pred_prob)
    # auc_score = metrics.auc(fpr, tpr)

    # F1 micro
    # f1_micro = metrics.f1_score(true_y, pred_y, average="micro")
    # f1_macro = metrics.f1_score(true_y, pred_y, average="macro")

    # return accuracy, auc_score, f1_micro, f1_macro
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-collab', '--collab', default=False, type=bool, help="if the coach previous collaborations (node embedding) should be considered as the features")

    args = parser.parse_args()
    collab = args.collab

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
    team_salary_df = team_salary_df.dropna()

    # only select NFL records and labels with salary data available
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

    salary_avail_arr = np.zeros((team_labels_df.shape[0])).astype(int)
    for idx, row in team_labels_df.iterrows():
        year = row.Year
        team = row.Team
        salary_data = team_salary_df[(team_salary_df.Team == team.replace('(NFL)','').strip()) & (team_salary_df.Year == year)]
        if salary_data.shape[0] > 0:
            salary_avail_arr[idx] = 1

    team_labels_df = team_labels_df.assign(salary_avail = salary_avail_arr)
    team_labels_df = team_labels_df[team_labels_df.salary_avail == 1]
    team_labels_df.reset_index(drop=True, inplace=True)


    # Define column names to be used as coach features
    basic_features = ["TotalYearsInNFL", "Past5yrsWinningPerc_best", "Past5yrsWinningPerc_avg"]
    cumul_emb_features = NFL_record_df.columns[NFL_record_df.columns.str.contains("cumul_emb")].tolist()

    # Separate train valid, and test datasets
    # train
    train_split_year, valid_split_year = 2015, 2017
    train_record = NFL_record_df[NFL_record_df.Year <= train_split_year]
    train_record.reset_index(drop=True, inplace=True)

    train_salary = team_salary_df[team_salary_df.Year <= train_split_year]
    train_salary.reset_index(drop=True, inplace=True)
    
    train_labels = team_labels_df[team_labels_df.Year <= train_split_year]
    train_labels.reset_index(drop=True, inplace=True)

    # valid
    valid_record = NFL_record_df[(NFL_record_df.Year > train_split_year) & (NFL_record_df.Year <= valid_split_year)]
    valid_record.reset_index(drop=True, inplace=True)

    valid_salary = team_salary_df[(team_salary_df.Year > train_split_year) & (team_salary_df.Year <= valid_split_year)]
    valid_salary.reset_index(drop=True, inplace=True)

    valid_labels = team_labels_df[(team_labels_df.Year > train_split_year) & (team_labels_df.Year <= valid_split_year)]
    valid_labels.reset_index(drop=True, inplace=True)

    # test
    test_record = NFL_record_df[(NFL_record_df.Year > valid_split_year)]
    test_record.reset_index(drop=True, inplace=True)

    test_salary = team_salary_df[(team_salary_df.Year > valid_split_year)]
    test_salary.reset_index(drop=True, inplace=True)

    test_labels = team_labels_df[(team_labels_df.Year > valid_split_year)]
    test_labels.reset_index(drop=True, inplace=True)

    print("Number of training records: {}, validation records: {}, testing records: {}".format(train_record.shape[0], valid_record.shape[0], test_record.shape[0]))

    print("Preparing features and labels...")
    # adding feature vectors as the node attributes in entire_team_G
    if collab == True:
        feature_names = basic_features + cumul_emb_features
    else:
        feature_names = basic_features

    train_seasons = train_record[["Year", "Team"]].drop_duplicates().to_dict('records')
    valid_seasons = valid_record[["Year", "Team"]].drop_duplicates().to_dict('records')
    test_seasons = test_record[["Year", "Team"]].drop_duplicates().to_dict('records')
    print("Number of training teams: {}, validation teams: {}, test teams: {}".format(len(train_seasons), len(valid_seasons), len(test_seasons)))

    # Aggregate coach features by simple aggregation
    print("Aggregating coach features for each season...")
    train_feature_arr = np.zeros((len(train_seasons), len(feature_names))) # array that stores features for each season
    for idx, season in enumerate(train_seasons):
        year = season["Year"]
        team = season["Team"]

        # collect season coaches
        coaches = train_record[(train_record.Team == team) & (train_record.Year == year)]
        coach_features = coaches[feature_names]

        # aggregate coach features
        #### AGGREGATE = MEAN
        aggregated_coach_feature = np.array(coach_features.mean())
        train_feature_arr[idx, :] = aggregated_coach_feature

    valid_feature_arr = np.zeros((len(valid_seasons), len(feature_names))) # array that stores features for each season
    for idx, season in enumerate(valid_seasons):
        year = season["Year"]
        team = season["Team"]

        # collect season coaches
        coaches = valid_record[(valid_record.Team == team) & (valid_record.Year == year)]
        coach_features = coaches[feature_names]

        # aggregate coach features
        #### AGGREGATE = MEAN
        aggregated_coach_feature = np.array(coach_features.mean())
        valid_feature_arr[idx, :] = aggregated_coach_feature

    test_feature_arr = np.zeros((len(test_seasons), len(feature_names))) # array that stores features for each season
    for idx, season in enumerate(test_seasons):
        year = season["Year"]
        team = season["Team"]

        # collect season coaches
        coaches = test_record[(test_record.Team == team) & (test_record.Year == year)]
        coach_features = coaches[feature_names]

        # aggregate coach features
        #### AGGREGATE = MEAN
        aggregated_coach_feature = np.array(coach_features.mean())
        test_feature_arr[idx, :] = aggregated_coach_feature

    # Prepare label array
    train_label_arr = np.array(train_labels.failure)
    valid_label_arr = np.array(valid_labels.failure)
    test_label_arr = np.array(test_labels.failure)

    # prepare salary array
    train_salary_arr = np.array(train_salary.Total_Salary)
    valid_salary_arr = np.array(valid_salary.Total_Salary)
    test_salary_arr = np.array(test_salary.Total_Salary)

    # combine aggregated features and salary
    train_x = np.concatenate((train_feature_arr, train_salary_arr.reshape(-1,1)), axis=1)
    valid_x = np.concatenate((valid_feature_arr, valid_salary_arr.reshape(-1,1)), axis=1)
    test_x = np.concatenate((test_feature_arr, test_salary_arr.reshape(-1,1)), axis=1)

    ## Random Forest Classifier
    print("*** Random Forest Classifier: ")
    accuracy_dict = {"valid": [], "test": []}
    auc_list = []
    f1_micro_list = []
    f1_macro_list = []

    clf = RandomForestClassifier(n_estimators=30)
    clf.fit(train_x, train_label_arr)

    valid_pred = clf.predict(valid_x)
    valid_accuracy = evaluate(valid_label_arr, valid_pred)
    accuracy_dict["valid"].append(valid_accuracy)

    test_pred = clf.predict(test_x)
    test_accuracy = evaluate(test_label_arr, test_pred)
    accuracy_dict["test"].append(test_accuracy)




    # print("AUC: {:.2f}\n F1 (micro): {:.2f}\n F1 (macro): {:.2f}".format(statistics.mean(auc_list), statistics.mean(f1_micro_list), statistics.mean(f1_macro_list)))




    





