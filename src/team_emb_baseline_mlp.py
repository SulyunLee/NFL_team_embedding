import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse
import statistics
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(x, dim=1)

        return x

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


def split_train_valid_test(train_split_year, valid_split_year, feature_names, NFL_record_df, team_salary_df, team_labels_df):
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


    # Aggregate coach features by simple aggregation
    print("Aggregating coach features for each season and preparing labels...")
    train_feature_arr, train_salary_arr, train_labels_arr = aggregate_features(train_record, train_salary, train_labels, feature_names)
    valid_feature_arr, valid_salary_arr, valid_labels_arr = aggregate_features(valid_record, valid_salary, valid_labels, feature_names)
    test_feature_arr, test_salary_arr, test_labels_arr = aggregate_features(test_record, test_salary, test_labels, feature_names)

    train = [train_feature_arr, train_salary_arr, train_labels_arr]
    valid = [valid_feature_arr, valid_salary_arr, valid_labels_arr]
    test = [test_feature_arr, test_salary_arr, test_labels_arr]

    return train, valid, test


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

    train, valid, test = split_train_valid_test(2015, 2017, feature_names, NFL_record_df, team_salary_df, team_labels_df)

    train_feature_arr, train_salary_arr, train_labels_arr = train[0], train[1], train[2]
    valid_feature_arr, valid_salary_arr, valid_labels_arr = valid[0], valid[1], valid[2]
    test_feature_arr, test_salary_arr, test_labels_arr = test[0], test[1], test[2]

    # combine aggregated features and salary
    train_x = np.concatenate((train_feature_arr, train_salary_arr.reshape(-1,1)), axis=1)
    valid_x = np.concatenate((valid_feature_arr, valid_salary_arr.reshape(-1,1)), axis=1)
    test_x = np.concatenate((test_feature_arr, test_salary_arr.reshape(-1,1)), axis=1)

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

    train_labels = torch.Tensor(train_labels_arr)
    valid_labels = torch.Tensor(valid_labels_arr)
    test_labels = torch.Tensor(test_labels_arr)

    loss = nn.BCELoss()
    epochs = 300


    patience = 7
    train_loss_arr = np.zeros((epochs))
    valid_loss_arr = np.zeros((epochs))
    test_loss_arr = np.zeros((epochs))

    # dictionaries that store average auc and accuracy for each hidden node
    mlp_auc_dict = {"train":[], "valid":[], "test":[]}
    mlp_accuracy_dict = {"train":[], "valid":[], "test":[]}
    hidden_nodes = [5, 10, 15, 20, 25, 30, 35, 40]
    hidden_nodes = [40]

    for hidden_node in hidden_nodes:
        print("Hidden node: {}".format(hidden_node))
        # dictionaries that store repeated performances
        repeat_performances = {"train":{"loss":[], "accuracy": [], "auc":[]},\
                                "valid": {"loss":[], "accuracy": [], "auc":[]},\
                                "test": {"loss":[], "accuracy": [], "auc":[]}}
        for repeat in tqdm(range(100)):
            model = MLPClassifier(input_dim=normalized_train_x.shape[1], hidden_dim=hidden_node, output_dim=2)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            for epoch in range(epochs):
                model.train()

                train_prob = model(normalized_train_x)
                train_loss = loss(train_prob[:,1].view(train_prob.shape[0],1), train_labels.view(train_labels.shape[0],1))


                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                # predict on validation and test sets
                with torch.no_grad():
                    model.eval()
                    train_prob = model(normalized_train_x)
                    train_loss = loss(train_prob[:,1].view(train_prob.shape[0],1), train_labels.view(train_labels.shape[0],1))
                    # train_loss_arr[epoch] = train_loss

                    # get the train predictions
                    _, train_pred = torch.max(train_prob, dim=1)
                    train_pred = train_pred.type('torch.FloatTensor')
                    train_pred = train_pred.view(train_labels.size())
                    train_accuracy = (train_pred == train_labels).sum().item() / train_labels.shape[0]
                    train_auc = roc_auc_score(train_labels, train_prob[:,1])


                    # # predict on the valid set
                    valid_prob = model(normalized_valid_x)
                    valid_loss = loss(valid_prob[:,1].view(valid_prob.shape[0],1), valid_labels.view(valid_labels.shape[0],1))
                    valid_loss_arr[epoch] = valid_loss

                    _, valid_pred = torch.max(valid_prob, dim=1)
                    valid_pred = valid_pred.type('torch.FloatTensor')
                    valid_pred = valid_pred.view(valid_labels.size())
                    valid_accuracy = (valid_pred == valid_labels).sum().item() / valid_labels.shape[0]
                    valid_auc = roc_auc_score(valid_labels, valid_prob[:,1])

                    # predict on the test set
                    test_prob = model(normalized_test_x)
                    test_loss = loss(test_prob[:,1].view(test_prob.shape[0],1), test_labels.view(test_labels.shape[0],1))
                    test_loss_arr[epoch] = test_loss

                    _, test_pred = torch.max(test_prob, dim=1)
                    test_pred = test_pred.type('torch.FloatTensor')
                    test_pred = test_pred.view(test_labels.size())
                    test_accuracy = (test_pred == test_labels).sum().item() / test_labels.shape[0]
                    test_auc = roc_auc_score(test_labels, test_prob[:,1])


                if epoch > patience and np.argmin(valid_loss_arr[epoch-patience:epoch]) == 0:
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

            # save 100 repeated performance (test)
        np.savez("temp_data/baseline1_mlp40_test_auc.npz", baseline1=np.array(repeat_performances["test"]["auc"]))

            
    # print("Epoch {}:\n, Train L: {:.4f}, acc: {:.3f}, auc: {}\n Val L: {:.4f}, acc: {:.3f}, auc: {}\n Test L: {:.4f}, acc: {:.3f}, auc:{}".format(epoch, train_loss, train_accuracy, train_auc, valid_loss, valid_accuracy, valid_auc, test_loss, test_accuracy, test_auc))

    # torch.save(model.state_dict(), 'pytorch_models/team_emb_baseline_mlp_{}hn_checkpoint.pth')

        avg_train_loss = statistics.mean(repeat_performances["train"]["loss"])
        avg_train_accuracy = statistics.mean(repeat_performances["train"]["accuracy"])
        avg_train_auc = statistics.mean(repeat_performances["train"]["auc"])
        avg_valid_loss = statistics.mean(repeat_performances["valid"]["loss"])
        avg_valid_accuracy = statistics.mean(repeat_performances["valid"]["accuracy"])
        avg_valid_auc = statistics.mean(repeat_performances["valid"]["auc"])
        avg_test_loss = statistics.mean(repeat_performances["test"]["loss"])
        avg_test_accuracy = statistics.mean(repeat_performances["test"]["accuracy"])
        avg_test_auc = statistics.mean(repeat_performances["test"]["auc"])

        mlp_auc_dict["train"].append(avg_train_auc)
        mlp_auc_dict["valid"].append(avg_valid_auc)
        mlp_auc_dict["test"].append(avg_test_auc)

        print("Hidden node={}:\n Train Loss: {:.3f}, accuracy: {:.3f}, auc: {:.3f}\n \
        Valid Loss: {:.3f}, accuracy: {:.3f}, auc: {:.3f}\n\
        Test Loss: {:.3f}, accuracy: {:.3f}, auc: {:.3f}\n".format(hidden_node, avg_train_loss, avg_train_accuracy, avg_train_auc, avg_valid_loss, avg_valid_accuracy, avg_valid_auc, avg_test_loss, avg_test_accuracy, avg_test_auc))

    # find the best hidden node
    max_idx = mlp_auc_dict["valid"].index(max(mlp_auc_dict["valid"]))
    print("Best model performances: hidden node={}".format(hidden_nodes[max_idx]))


