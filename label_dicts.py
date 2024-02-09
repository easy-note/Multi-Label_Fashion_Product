import pandas as pd
import joblib

from utils import clean_data


def save_label_dicts(df):
    # remove rows from the DataFrame which do not have corresponding images
    df = clean_data(df)

    # we will use the 'gender', 'articleType', 'season', and `usage` labels
    # mapping `gender` to numerical values
    cat_list_gender = df['gender'].unique()
    # 5 unique categories for gender
    num_list_gender = {cat:i for i, cat in enumerate(cat_list_gender)}
    
    # mapping `articleType` to numerical values
    cat_list_articletype = df['articleType'].unique()
    # 5 unique categories for articleType
    num_list_articletype = {cat:i for i, cat in enumerate(cat_list_articletype)}
    
    # mapping `season` to numerical values
    cat_list_season = df['season'].unique()
    # 5 unique categories for season
    num_list_season = {cat:i for i, cat in enumerate(cat_list_season)}
    
    # mapping `usage` to numerical values
    cat_list_usage = df['usage'].unique()
    # 5 unique categories for usage
    num_list_usage = {cat:i for i, cat in enumerate(cat_list_usage)}
    
    print(num_list_gender)
    print(num_list_articletype)
    print(num_list_season)
    print(num_list_usage)
    
    joblib.dump(num_list_gender, './num_list_gender.pkl')
    joblib.dump(num_list_articletype, './num_list_articletype.pkl')
    joblib.dump(num_list_season, './num_list_season.pkl')
    joblib.dump(num_list_usage, './num_list_usage.pkl')
    

df = pd.read_csv('./data/styles.csv', usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], keep_default_na=False)
save_label_dicts(df)