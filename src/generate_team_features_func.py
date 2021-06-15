import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def generate_team_diversity_feature(NFL_record_df, feature_names):
    # Diversity features
    # Maximum pairwise similarity of coaches' embeddings in the same team (non-zero)
    seasons = NFL_record_df[["Team","Year"]].drop_duplicates()
    seasons.reset_index(drop=True, inplace=True)

    emb_sim_max_arr = np.zeros((len(seasons)))
    emb_sim_mean_arr = np.zeros((len(seasons)))
    for idx, season in seasons.iterrows():
        team = season.Team
        year = season.Year
        coach_emb = np.array(NFL_record_df[(NFL_record_df.Team == team) & (NFL_record_df.Year == year) & (NFL_record_df.no_node_emb == 0)][feature_names])

        pairwise_emb_sim = cosine_similarity(coach_emb)
        iu = np.triu_indices(pairwise_emb_sim.shape[0], 1)
        emb_sim_max = pairwise_emb_sim[iu].max()
        emb_sim_mean = pairwise_emb_sim[iu].mean()

        emb_sim_max_arr[idx] = emb_sim_max
        emb_sim_mean_arr[idx] = emb_sim_mean

    seasons = seasons.assign(Max_Emb_Similarity = emb_sim_max_arr)
    seasons = seasons.assign(Mean_Emb_Similarity = emb_sim_mean_arr)

    return seasons
