import pandas as pd
import numpy as np

def simple_aggregate_features(record_df, team_features_df, labels_df, coach_feature_names, team_feature_names, label_name):
    seasons = record_df[["Year", "Team"]].drop_duplicates().to_dict('records')
    print("Number of seasons: {}".format(len(seasons)))

    print("Aggregating coach features for each season...")
    team_emb_arr = np.zeros((len(seasons), len(coach_feature_names)))
    team_feature_arr = np.zeros((len(seasons), len(team_feature_names)))
    team_labels_arr = np.zeros((len(seasons))).astype(int)
    
    for idx, season in enumerate(seasons):
        year = season["Year"]
        team = season["Team"]

        # collect season coaches
        coaches = record_df[(record_df.Team == team) & (record_df.Year == year)]
        coach_features = coaches[coach_feature_names]

        # aggregate coach features
        #### AGGREGATE = MEAN
        aggregated_coach_feature = np.array(coach_features.mean())
        team_emb_arr[idx, :] = aggregated_coach_feature

        # collect salary data of the season
        team_feature = np.array(team_features_df[(team_features_df.Team == team) & (team_features_df.Year == year)][team_feature_names])
        team_feature_arr[idx, :] = team_feature

        # collect label of the season
        label = labels_df[(labels_df.Team == team.replace('(NFL)', '').strip()) & (labels_df.Year == year)][label_name]
        team_labels_arr[idx] = int(label)

    return team_emb_arr, team_feature_arr, team_labels_arr

def hierarchical_average_features(record_df, team_features_df, labels_df, coach_feature_names, team_feature_names, label_name):
    seasons = record_df[["Year", "Team"]].drop_duplicates().to_dict('records')
    print("The number of seasons: {}".format(len(seasons)))

    print("Generating team embedding in hierarchical way...")
    team_emb_arr = np.zeros((len(seasons), len(coach_feature_names)))
    team_feature_arr = np.zeros((len(seasons), len(team_feature_names)))
    team_labels_arr = np.zeros((len(seasons))).astype(int)

    for idx, season in enumerate(seasons):
        year = season["Year"]
        team = season["Team"]

        # collect season coaches
        coaches = record_df[(record_df.Team== team) & (record_df.Year == year)]

        sums = []
        ## aggregate position coach features - average
        # offensive position coaches
        offensive_position = coaches[coaches.final_position == "O"]
        if offensive_position.shape[0] != 0:
            offensive_position_emb = np.array(offensive_position[coach_feature_names].mean())
        # defensive position coaches
        defensive_position = coaches[coaches.final_position == "D"]
        if defensive_position.shape[0] != 0:
            defensive_position_emb = np.array(defensive_position[coach_feature_names].mean())
        # special team position coaches
        special_position = coaches[coaches.final_position == "S"]
        if special_position.shape[0] != 0:
            special_position_emb = np.array(special_position[coach_feature_names].mean())

        ## aggregate position coach embedding and coordinator features
        # offensive position + offensive coordinator
        offensive_coord = coaches[coaches.final_position == "OC"]
        if offensive_coord.shape[0] != 0:
            offensive_coord_emb = np.array(offensive_coord[coach_feature_names].mean())
            if offensive_position.shape[0] != 0:
                offensive_emb = (offensive_position_emb + offensive_coord_emb)/2
                sums.append(offensive_emb)
            else:
                offensive_emb = offensive_coord_emb
                sums.append(offensive_emb)
        elif offensive_position.shape[0] != 0:
            offensive_emb = offensive_position_emb
            sums.append(offensive_emb)
        # defensive position + defensive coordinator
        defensive_coord = coaches[coaches.final_position == "DC"]
        if defensive_coord.shape[0] != 0:
            defensive_coord_emb = np.array(defensive_coord[coach_feature_names].mean())
            if defensive_position.shape[0] != 0:
                defensive_emb = (defensive_position_emb + defensive_coord_emb)/2
                sums.append(defensive_emb)
            else:
                defensive_emb = defensive_coord_emb
                sums.append(defensive_emb)
        elif defensive_position.shape[0] != 0:
            defensive_emb = defensive_position_emb
            sums.append(defensive_emb)
        # special team position + special team coordinator
        special_coord = coaches[coaches.final_position == "SC"]
        if special_coord.shape[0] != 0:
            special_coord_emb = np.array(special_coord[coach_feature_names].mean())
            if special_position.shape[0] != 0:
                special_emb = (special_position_emb + special_coord_emb)/2
                sums.append(special_emb)
            else:
                special_emb = special_coord_emb
                sums.append(special_emb)
        elif special_position.shape[0] != 0:
            special_emb = special_position_emb
            sums.append(special_emb)
        
        ## aggregate offensive, defensive, special, and head coach features
        hc = coaches[coaches.final_position == "HC"]
        hc_f = np.array(hc[coach_feature_names])
        sums.append(hc_f)

        # team embedding
        team_emb = sum(sums) / len(sums)
        team_emb_arr[idx,:] = team_emb

        # collect salary data of the season
        team_features = np.array(team_features_df[(team_features_df.Team == team) & (team_features_df.Year == year)][team_feature_names])
        team_feature_arr[idx,:] = team_features

        # collect label of the season
        label = labels_df[(labels_df.Team == team.replace('(NFL)', '').strip()) & (labels_df.Year == year)][label_name]
        team_labels_arr[idx] = int(label)

    return team_emb_arr, team_feature_arr, team_labels_arr
