'''
This script generates team embedding of a team network using the baseline model.
The baseline model does not consider the hierarchical structure of a team, but simply
aggregates the team members' features.
'''
import pandas as pd
import numpy as np
import statistics
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def evaluate(true_y, pred_y, pred_prob):
    ## AUC
    fpr, tpr, thresholds = metrics.roc_curve(true_y, pred_prob)
    auc_score = metrics.auc(fpr, tpr)

    # F1 micro
    f1_micro = metrics.f1_score(true_y, pred_y, average="micro")
    f1_macro = metrics.f1_score(true_y, pred_y, average="macro")

    return auc_score, f1_micro, f1_macro


if __name__ == "__main__":
    #################################################################
    # Load datasets
    NFL_coach_record_filename = "../datasets/2002-2019_NFL_Coach_Data_with_features.csv"
    team_labels_filename = "../datasets/team_labels.csv"

    NFL_record_df = pd.read_csv(NFL_coach_record_filename)
    team_labels_df = pd.read_csv(team_labels_filename)
    #################################################################
    basic_features = ["TotalYearsInNFL", "Past5yrsWinningPerc_best", "Past5yrsWinningPerc_avg", "HC", "Coord"]

    # exclude interim head coaches
    NFL_record_df = NFL_record_df[NFL_record_df.final_position != "iHC"]
    NFL_record_df.reset_index(drop=True, inplace=True)

    df = team_labels_df[["Team", "Year", "failure"]]

    #######################################################
    ### Erase this code block after the data is complete ##
    NFL_record_df = NFL_record_df.dropna()
    NFL_record_df.reset_index(drop=True, inplace=True)
    df = df[df.Year != 2002]
    df.reset_index(drop=True, inplace=True)
    #######################################################
    #######################################################

    # Aggregate coach features by simple aggregation
    print("Aggregating coach features for each season...")
    feature_arr = np.zeros((df.shape[0], len(basic_features))) # array that stores features for each season
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        team = row.Team
        year = row.Year

        # collect season coaches
        coaches = NFL_record_df[(NFL_record_df.Team == "{} (NFL)".format(team)) & (NFL_record_df.Year == year)]
        coach_features = coaches[basic_features]

        # aggregate coach features
        #### AGGREGATE = MEAN
        aggregated_coach_feature = np.array(coach_features.mean())
        feature_arr[idx, :] = aggregated_coach_feature

    # Prepare label array
    label_arr = np.array(df.failure)
    # Classification
    # Features: season aggregated coach feature
    # Labels: Team failure

    # K-fold cross validation
    kf = KFold(n_splits=5)
    
    ## Random Forest Classifier
    print("*** Random Forest Classifier: ")
    auc_list = []
    f1_micro_list = []
    f1_macro_list = []
    for train_idx, test_idx in kf.split(feature_arr):
        X_train, X_test = feature_arr[train_idx], feature_arr[test_idx]
        y_train, y_test = label_arr[train_idx], label_arr[test_idx]

        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        pred_prob = clf.predict_proba(X_test)[:,clf.classes_==1]

        # evaluate
        auc_score, f1_micro, f1_macro = evaluate(y_test, pred, pred_prob)
        auc_list.append(auc_score)
        f1_micro_list.append(f1_micro)
        f1_macro_list.append(f1_macro)

    print("AUC: {:.2f}\n F1 (micro): {:.2f}\n F1 (macro): {:.2f}".format(statistics.mean(auc_list), statistics.mean(f1_micro_list), statistics.mean(f1_macro_list)))




    





