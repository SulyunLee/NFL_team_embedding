import numpy as np
import pandas as pd

def dict_of_dict_to_dataframe(coach_dict, emb_dim):
    total_instances = sum([len(coach_dict[coach].keys()) for coach in coach_dict.keys()])

    coach_array = np.zeros((total_instances)).astype(object)
    year_array = np.zeros((total_instances)).astype(int)
    emb_array = np.zeros((total_instances, emb_dim))

    idx = 0
    for coach in coach_dict:
        for year in coach_dict[coach]:
            coach_array[idx] = coach
            year_array[idx] = year
            emb_array[idx,:] = coach_dict[coach][year]
            idx += 1

    df_coach_year_embedding = pd.DataFrame(data=emb_array, columns=["cumul_emb{}".format(i) for i in range(emb_dim)])
    df_coach_year_embedding.insert(value=coach_array, loc=0, column="Name") 
    df_coach_year_embedding.insert(value=year_array, loc=1, column="Year") 
    return df_coach_year_embedding

# emb_dim = 10
# coach_dict = {"Hankyu": {2002: np.random.random(emb_dim), 2003: np.random.random(emb_dim)},
            # "Banggu": {2003: np.random.random(emb_dim), 2004: np.random.random(emb_dim)},
            # "Sultan": {2003: np.random.random(emb_dim), 2004: np.random.random(emb_dim), 2005: np.random.random(emb_dim)}
            # }

# df_coach_year_embedding = dict_of_dict_to_dataframe(coach_dict, emb_dim)
