
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # Load datasets
    team_salary_filename = "../datasets/Total_Salary.csv"
    team_salary_df = pd.read_csv(team_salary_filename)

    seasons = team_salary_df[["Team", "Year"]].to_dict('records')
    years = team_salary_df.Year.unique()

    salary_rank_arr = np.zeros((team_salary_df.shape[0]))
    for year in years:
        salary = team_salary_df[team_salary_df.Year == year].Total_Salary
        # salary.reset_index(drop=True, inplace=True)
        salary_rank_dict = dict(zip(salary.sort_values(ascending=False), range(1, 33)))
        mapped_rank = salary.map(salary_rank_dict)
        salary_rank_arr[salary.index] = mapped_rank

    team_salary_df = team_salary_df.assign(Salary_Rank = salary_rank_arr)
    team_salary_df.to_csv(team_salary_filename, index=False, encoding="utf-8-sig")


