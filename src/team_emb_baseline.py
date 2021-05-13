'''
This script generates team embedding of a team network using the baseline model.
The baseline model does not consider the hierarchical structure of a team, but simply
aggregates the team members' features.
The following classifiers are used:
    1) Random forest
    2) SVM
    3) MLP (team_emb_baseline_mlp.py)
'''
import pandas as pd
import numpy as np
import statistics
import argparse
import tqdm
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics

def aggregate_features(record_df, salary_df, labels_df, feature_names):
    seasons = record_df[["Year", "Team"]].drop_duplicates().to_dict('records')
    print("Number of seasons: {}".format(len(seasons)))

    print("Aggregating coach features for each season...")
    team_feature_arr = np.zeros((len(seasons), len(feature_names)))
    team_salary_arr = np.zeros((len(seasons))).astype(int)
    team_labels_arr = np.zeros((len(seasons))).astype(int)
    
    for idx, season in enumerate(seasons):
        year = season["Year"]
        team = season["Team"]

        # collect season coaches
        coaches = record_df[(record_df.Team == team) & (record_df.Year == year)]
        coach_features = coaches[feature_names]

        # aggregate coach features
        #### AGGREGATE = MEAN
        aggregated_coach_feature = np.array(coach_features.mean())
        team_feature_arr[idx, :] = aggregated_coach_feature

        # collect salary data of the season
        salary = salary_df[(salary_df.Team == team.replace('(NFL)', '').strip()) & (salary_df.Year == year)].Salary_Rank
        team_salary_arr[idx] = int(salary)

        # collect label of the season
        label = labels_df[(labels_df.Team == team.replace('(NFL)', '').strip()) & (labels_df.Year == year)].failure
        team_labels_arr[idx] = int(label)

    return team_feature_arr, team_salary_arr, team_labels_arr



def evaluate(true_y, pred_y, pred_prob):
    ## Accuracy
    accuracy = metrics.accuracy_score(true_y, pred_y)
    accuracy = round(accuracy, 3)

    ## AUC
    fpr, tpr, thresholds = metrics.roc_curve(true_y, pred_prob)
    auc_score = metrics.auc(fpr, tpr)
    auc_score = round(auc_score, 3)

    # F1 micro
    # f1_micro = metrics.f1_score(true_y, pred_y, average="micro")
    # f1_macro = metrics.f1_score(true_y, pred_y, average="macro")

    # return accuracy, auc_score, f1_micro, f1_macro
    return accuracy, auc_score


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
    team_salary_df = team_salary_df.dropna(subset=["Total_Salary"])

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

    # adding feature vectors as the node attributes in entire_team_G
    if collab == True:
        feature_names = basic_features + cumul_emb_features
    else:
        feature_names = basic_features

    # Split coach record, salary, and label into train, validation, and test set
    train_split_year = 2015
    valid_split_year = 2017

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

    # aggregate coaches' features in the same team.
    print("Train")
    train_feature_arr, train_salary_arr, train_labels_arr = aggregate_features(train_record, train_salary, train_labels, feature_names)
    print("Validation")
    valid_feature_arr, valid_salary_arr, valid_labels_arr = aggregate_features(valid_record, valid_salary, valid_labels, feature_names)
    print("Test")
    test_feature_arr, test_salary_arr, test_labels_arr = aggregate_features(test_record, test_salary, test_labels, feature_names)

    # combine aggregated features and salary
    train_x = np.concatenate((train_feature_arr, train_salary_arr.reshape(-1,1)), axis=1)
    valid_x = np.concatenate((valid_feature_arr, valid_salary_arr.reshape(-1,1)), axis=1)
    test_x = np.concatenate((test_feature_arr, test_salary_arr.reshape(-1,1)), axis=1)


    ## Random Forest Classifier
    print("*** Random Forest Classifier: ")
    rf_auc_dict = {"train": [], "valid": [], "test": []}
    max_depths = [1, 2, 3]
    max_depths = []
    for max_depth in max_depths:
        print("Max depth: {}".format(max_depth))
        repeat_performances = {"train":{"accuracy": [], "auc":[]},\
                                "valid": {"accuracy": [], "auc":[]},\
                                "test": {"accuracy": [], "auc":[]}}
        for repeat in tqdm(range(100)):
            clf = RandomForestClassifier(n_estimators=1000, max_depth=max_depth)
            clf.fit(train_x, train_labels_arr)
            train_pred = clf.predict(train_x)
            train_pred_prob = clf.predict_proba(train_x)[:,np.where(clf.classes_==1)[0]]
            train_accuracy, train_auc = evaluate(train_labels_arr, train_pred, train_pred_prob)

            repeat_performances["train"]["accuracy"].append(train_accuracy)
            repeat_performances["train"]["auc"].append(train_auc)

            valid_pred = clf.predict(valid_x)
            valid_pred_prob = clf.predict_proba(valid_x)[:,np.where(clf.classes_==1)[0]]
            valid_accuracy, valid_auc = evaluate(valid_labels_arr, valid_pred, valid_pred_prob)

            repeat_performances["valid"]["accuracy"].append(valid_accuracy)
            repeat_performances["valid"]["auc"].append(valid_auc)

            test_pred = clf.predict(test_x)
            test_pred_prob = clf.predict_proba(test_x)[:,np.where(clf.classes_==1)[0]]
            test_accuracy, test_auc = evaluate(test_labels_arr, test_pred, test_pred_prob)
            
            repeat_performances["test"]["accuracy"].append(test_accuracy)
            repeat_performances["test"]["auc"].append(test_auc)

        avg_train_accuracy = statistics.mean(repeat_performances["train"]["accuracy"])
        avg_train_auc = statistics.mean(repeat_performances["train"]["auc"])
        avg_valid_accuracy = statistics.mean(repeat_performances["valid"]["accuracy"])
        avg_valid_auc = statistics.mean(repeat_performances["valid"]["auc"])
        avg_test_accuracy = statistics.mean(repeat_performances["test"]["accuracy"])
        avg_test_auc = statistics.mean(repeat_performances["test"]["auc"])

        rf_auc_dict["train"].append(avg_train_auc)
        rf_auc_dict["valid"].append(avg_valid_auc)
        rf_auc_dict["test"].append(avg_test_auc)

        print("Max depth={}: \n \
        Train accuracy: {:.3f}, auc: {:.3f}\n \
        Valid accuracy: {:.3f}, auc: {:.3f}\n \
        Test accuracy: {:.3f}, auc: {:.3f}\n".format(max_depth, avg_train_accuracy, avg_train_auc, avg_valid_accuracy, avg_valid_auc, avg_test_accuracy, avg_test_auc))

    # select the hyper-parameter with the highest valid accuracy
    # max_idx = rf_auc_dict["valid"].index(max(rf_auc_dict["valid"]))

    # print("Best model performances: max_depth={}".format(max_depths[max_idx]))

    ## Support vector machines
    print("*** SVM: ")
    svm_auc_dict = {"train": [], "valid": [], "test": []}
    # cs = [1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16]
    cs = [1/4]
    for c in tqdm(cs):
        print("C: {}".format(c))
        repeat_performances = {"train":{"accuracy": [], "auc":[]},\
                                "valid": {"accuracy": [], "auc":[]},\
                                "test": {"accuracy": [], "auc":[]}}
        for repeat in tqdm(range(100)):
            clf = SVC(C=c, kernel="linear", gamma="scale", probability=True)
            clf.fit(train_x, train_labels_arr)
            train_pred = clf.predict(train_x)
            train_pred_prob = clf.predict_proba(train_x)[:,np.where(clf.classes_==1)[0]]
            train_accuracy, train_auc = evaluate(train_labels_arr, train_pred, train_pred_prob)
            repeat_performances["train"]["accuracy"].append(train_accuracy)
            repeat_performances["train"]["auc"].append(train_auc)

            valid_pred = clf.predict(valid_x)
            valid_pred_prob = clf.predict_proba(valid_x)[:,np.where(clf.classes_==1)[0]]
            valid_accuracy, valid_auc = evaluate(valid_labels_arr, valid_pred, valid_pred_prob)
            repeat_performances["valid"]["accuracy"].append(valid_accuracy)
            repeat_performances["valid"]["auc"].append(valid_auc)

            test_pred = clf.predict(test_x)
            test_pred_prob = clf.predict_proba(test_x)[:,np.where(clf.classes_==1)[0]]
            test_accuracy, test_auc = evaluate(test_labels_arr, test_pred, test_pred_prob)
            repeat_performances["test"]["accuracy"].append(test_accuracy)
            repeat_performances["test"]["auc"].append(test_auc)

        np.savez("temp_data/baseline2_svm_test_auc.npz", baseline2=np.array(repeat_performances["test"]["auc"]))

        avg_train_accuracy = statistics.mean(repeat_performances["train"]["accuracy"])
        avg_train_auc = statistics.mean(repeat_performances["train"]["auc"])
        avg_valid_accuracy = statistics.mean(repeat_performances["valid"]["accuracy"])
        avg_valid_auc = statistics.mean(repeat_performances["valid"]["auc"])
        avg_test_accuracy = statistics.mean(repeat_performances["test"]["accuracy"])
        avg_test_auc = statistics.mean(repeat_performances["test"]["auc"])

        svm_auc_dict["train"].append(avg_train_auc)
        svm_auc_dict["valid"].append(avg_valid_auc)
        svm_auc_dict["test"].append(avg_test_auc)

        print("C={}: \n \
        Train accuracy: {:.3f}, auc: {:.3f}\n \
        Valid accuracy: {:.3f}, auc: {:.3f}\n \
        Test accuracy: {:.3f}, auc: {:.3f}\n".format(c, avg_train_accuracy, avg_train_auc, avg_valid_accuracy, avg_valid_auc, avg_test_accuracy, avg_test_auc))

    # select the hyper-parameter with the highest valid accuracy
    max_idx = svm_auc_dict["valid"].index(max(svm_auc_dict["valid"]))

    print("Best model performances: C={}".format(cs[max_idx]))

