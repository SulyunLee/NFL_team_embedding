
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
    parser.add_argument('-collab_type', '--collab_type', default='all', type=str, help="Collaboration type: NFL or all")
    parser.add_argument('-hier', '--hier', default=False, type=bool, help="Mentorship network or not")
    parser.add_argument('-biased', '--biased', default=False, type=bool, help="Random walks are biased or not")
    parser.add_argument('-prob', '--prob', type=int)
    parser.add_argument('-w', '--w', default=3, type=int, help="window size")
    parser.add_argument('-emb_size', '--emb_size', type=int, help="node embedding size")
    parser.add_argument('-mode', '--mode', default='expanded', type=str, help="simple for only considering teams with 8 position coach titles or expanded for considering all teams with qualified position coach titles")

    args = parser.parse_args()
    feature_set = args.feature_set
    train_split_year = args.train_split_year
    valid_split_year = args.valid_split_year
    collab_type = args.collab_type
    hier = args.hier
    biased = args.biased
    prob = args.prob
    w = args.w
    emb_size = args.emb_size
    mode = args.mode

    #################################################################
    # Load datasets
    NFL_coach_record_filename = "../datasets/NFL_Coach_Data_with_features_{}_size{}_collab{}_hier{}_biased{}_selectionprob{}_w{}.csv".format(mode, emb_size, collab_type, hier, biased, prob, w)
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
    elif feature_set == 21:
        coach_feature_names = basic_features + cumul_emb_features
        team_feature_names = ["Salary_Rank"]
    elif feature_set == 22:
        coach_feature_names = basic_features + cumul_emb_features
        team_feature_names = []
    elif feature_set == 31:
        coach_feature_names = basic_features + cumul_emb_features
        team_feature_names = ["Salary_Rank", "Max_Emb_Similarity", "Mean_Emb_Similarity"]
    elif feature_set == 32:
        coach_feature_names = basic_features + cumul_emb_features
        team_feature_names = ["Max_Emb_Similarity", "Mean_Emb_Similarity"]

    print("Feature set {}".format(feature_set))

    if mode == "simple":
        print("Extracting complete teams...")
        off_position_titles = ["QB", "RB", "OL", "WR", "TE"] 
        def_position_titles = ["LB", "DL", "Sec"]
        complete_year_team_pairs = get_complete_teams(NFL_record_df, off_position_titles + def_position_titles)
        year_team_tuple = NFL_record_df[["Year","Team"]].apply(tuple, axis=1)
        NFL_record_df = NFL_record_df.assign(year_team = year_team_tuple)
        NFL_record_df = NFL_record_df[NFL_record_df.year_team.isin(complete_year_team_pairs)]
        NFL_record_df.reset_index(drop=True, inplace=True)
        print("Number of teams with complete position coaches: {}".format(NFL_record_df[["Year","Team"]].drop_duplicates().shape[0]))
    
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
    # Required inputs: averaged position coaches, OC, DC, HC
    print("Generating team embedding, salary, and labels...")
    print("Train")
    train_offensive_position, train_defensive_position, train_offensive_coord, train_defensive_coord, train_hc, train_team_features, train_labels_arr = generate_coach_features_by_hierarchy(train_record, train_team_features, train_labels, coach_feature_names, team_feature_names, "failure")

    print("Validation")
    valid_offensive_position, valid_defensive_position, valid_offensive_coord, valid_defensive_coord, valid_hc, valid_team_features, valid_labels_arr = generate_coach_features_by_hierarchy(valid_record, valid_team_features, valid_labels, coach_feature_names, team_feature_names, "failure")

    print("Test")
    test_offensive_position, test_defensive_position, test_offensive_coord, test_defensive_coord, test_hc, test_team_features, test_labels_arr = generate_coach_features_by_hierarchy(test_record, test_team_features, test_labels, coach_feature_names, team_feature_names, "failure")

    ## Normalize
    print("Normalizing features...")
    # offensive position
    normalized_train_offensive_position, normalized_valid_offensive_position, normalized_test_offensive_position = normalize(train_offensive_position, valid_offensive_position, test_offensive_position)
    # defensive position
    normalized_train_defensive_position, normalized_valid_defensive_position, normalized_test_defensive_position = normalize(train_defensive_position, valid_defensive_position, test_defensive_position)
    # offensive coord
    normalized_train_offensive_coord, normalized_valid_offensive_coord, normalized_test_offensive_coord = normalize(train_offensive_coord, valid_offensive_coord, test_offensive_coord)
    # defensive coord
    normalized_train_defensive_coord, normalized_valid_defensive_coord, normalized_test_defensive_coord = normalize(train_defensive_coord, valid_defensive_coord, test_defensive_coord)

    # head coach
    normalized_train_hc, normalized_valid_hc, normalized_test_hc = normalize(train_hc, valid_hc, test_hc)

    # team features
    if len(team_feature_names) != 0:
        normalized_train_team_features, normalized_valid_team_features, normalized_test_team_features = normalize(train_team_features_arr, valid_team_features_arr, test_team_features_arr)
    else:
        normalized_train_team_features = normalized_valid_team_features = normalized_test_team_features = torch.Tensor()

    # Convert labels to tensors
    train_labels_arr = torch.Tensor(train_labels_arr).view(train_labels_arr.shape[0], 1)
    valid_labels_arr = torch.Tensor(valid_labels_arr).view(valid_labels_arr.shape[0], 1)
    test_labels_arr = torch.Tensor(test_labels_arr).view(test_labels_arr.shape[0], 1)

    # Modeling
    print("Training model...")
    loss = nn.BCEWithLogitsLoss()
    epochs = 100000

    repeated_train_loss_arr = np.zeros((11))
    repeated_valid_loss_arr = np.zeros((11))
    repeated_test_loss_arr = np.zeros((11))
    repeated_train_auc_arr = np.zeros((11))
    repeated_valid_auc_arr = np.zeros((11))
    repeated_test_auc_arr = np.zeros((11))
    for seed in range(0,11):
        torch.manual_seed(seed)
        model = Hier_NN_V2(coach_features_dim = len(coach_feature_names),
                            emb_dim = int(len(coach_feature_names)),
                            team_feature_dim = len(team_feature_names),
                            output_dim = 1,
                            feature_set = feature_set)
        print(model)

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

        for epoch in range(epochs):
            model.train()

            optimizer.zero_grad()

            train_y_hat = model(normalized_train_offensive_position, normalized_train_defensive_position, normalized_train_offensive_coord, normalized_train_defensive_coord, normalized_train_hc, normalized_train_team_features)
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
                valid_y_hat = model(normalized_valid_offensive_position, normalized_valid_defensive_position, normalized_valid_offensive_coord, normalized_valid_defensive_coord, normalized_valid_hc, normalized_valid_team_features)
                valid_loss = loss(valid_y_hat, valid_labels_arr)
                valid_loss_arr[epoch] = valid_loss

                valid_prob = torch.sigmoid(valid_y_hat)
                valid_pred = torch.round(valid_prob)
                valid_auc = round(roc_auc_score(valid_labels_arr.detach().numpy(), valid_prob.detach().numpy()), 3)
                valid_auc_arr[epoch] = valid_auc

                # predict on the test set
                test_y_hat = model(normalized_test_offensive_position, normalized_test_defensive_position, normalized_test_offensive_coord, normalized_test_defensive_coord, normalized_test_hc, normalized_test_team_features)
                test_loss = loss(test_y_hat, test_labels_arr)
                test_loss_arr[epoch] = test_loss

                test_prob = torch.sigmoid(test_y_hat)
                test_pred = torch.round(test_prob)
                test_auc = round(roc_auc_score(test_labels_arr.detach().numpy(), test_prob.detach().numpy()), 3)
                test_auc_arr[epoch] = test_auc
                
                counter, stop = stopper.step(valid_loss, model)
                if counter == 1:
                    remember_epoch = epoch - 1
                if stop:
                    break

        # print("Train Loss: {:.3f}, AUC: {:.3f}\nValid Loss: {:.3f}, AUC: {:.3f}\nTest Loss: {:.3f}, AUC: {:.3f}".format(train_loss_arr[remember_epoch], train_auc_arr[remember_epoch], valid_loss_arr[remember_epoch], valid_auc_arr[remember_epoch], test_loss_arr[remember_epoch], test_auc_arr[remember_epoch]))
        repeated_train_loss_arr[seed] = train_loss_arr[remember_epoch]
        repeated_valid_loss_arr[seed] = valid_loss_arr[remember_epoch]
        repeated_test_loss_arr[seed] = test_loss_arr[remember_epoch]
        repeated_train_auc_arr[seed] = train_auc_arr[remember_epoch]
        repeated_valid_auc_arr[seed] = valid_auc_arr[remember_epoch]
        repeated_test_auc_arr[seed] = test_auc_arr[remember_epoch]

    # print("Feature set: {}\nTrain: 2002-{}, Validation: {}-{}, Test: {}-2019\nCollaboration type: {}, Network type: hierarchy {}, Biased random walk: {} (selection prob {})".format(feature_set, train_split_year, train_split_year+1, valid_split_year, valid_split_year+1, collab_type, hier, biased, prob))
    print("Train loss \t Valid loss \t Test loss \t Train AUC \t Valid AUC \t Test AUC")
    print(round(repeated_train_loss_arr.mean(),3), round(repeated_valid_loss_arr.mean(),3), round(repeated_test_loss_arr.mean(),3), round(repeated_train_auc_arr.mean(),3), round(repeated_valid_auc_arr.mean(),3), round(repeated_test_auc_arr.mean(),3))





