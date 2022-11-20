"""
Init date: 17th Nov 2022
Update date:
Description: This script is an analysis for the SkillCorner dataset found in the repo
SkillCorner/opendata: SkillCorner Open Data with 9 matches of broadcast tracking data. (github.com)
Github link: https://github.com/SkillCorner/opendata
Author: @sijielim
"""

import json
import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv("blue-crow-sports/.env")

ROOT = os.getenv("ROOT")
DATA_DIR = os.path.join(ROOT, "opendata", "data")
INPUT_JSON_PATH = os.path.join(DATA_DIR, "matches.json")

file_df = pd.read_json(INPUT_JSON_PATH)

# for match_metadata in file_df.values.tolist():
match_metadata = file_df.values.tolist()[0]
match_status, match_dt, home_team, away_team, match_id = match_metadata
home_team = home_team["short_name"]
away_team = away_team["short_name"]

match_data_dir = os.path.join(DATA_DIR, "matches", str(match_id))
match_data_json_path = os.path.join(match_data_dir, "match_data.json")
match_structured_data_json_path = os.path.join(match_data_dir, "structured_data.json")

with open(match_data_json_path, "r") as f:
    match_info_dict = json.load(f)
match_struc_data_df = pd.read_json(match_structured_data_json_path)
match_struc_data_df[["trackable_object", "group"]] = pd.json_normalize(match_struc_data_df["possession"])
match_struc_data_df.drop(["possession"], axis=1, inplace=True)

## There are certain frames where the group is None. Drop those rows where time == None
match_struc_data_df = match_struc_data_df[~match_struc_data_df["time"].isna()]
match_struc_data_df = match_struc_data_df.reset_index(drop=True)

match_struc_data_df["group"] = match_struc_data_df["group"].replace("home team", home_team)
match_struc_data_df["group"] = match_struc_data_df["group"].replace("away team", away_team)

match_struc_data_df["data_length"] = match_struc_data_df["data"].apply(lambda x: len(x))

match_struc_data_df["group"].value_counts()
match_struc_data_df["time"].value_counts().sort_index()


def explode_data(df: pd.DataFrame,
                 row_idx: int,
                 track_list: list):
    """
    Extract
    """
    for tracked in track_list:
        if tracked.get("trackable_object"):
            track_obj = tracked.get("trackable_object")
            if track_obj == "55":
                df.loc[row_idx, f"{track_obj}_z"] = tracked.get("z")
        else:
            track_obj = tracked.get("group_name").replace(" ", "_").lower()

        df.loc[row_idx, f"{track_obj}_x"] = tracked.get("x")
        df.loc[row_idx, f"{track_obj}_y"] = tracked.get("y")
        df.loc[row_idx, f"{track_obj}_track_id"] = tracked.get("track_id")

        print(f'{df.loc[row_idx, f"{track_obj}_track_id"]}, {df.loc[row_idx, f"{track_obj}_x"]}, {df.loc[row_idx, f"{track_obj}_y"]}')
    return df

## Processing using a list comprehension to run it faster
for idx, track_list in enumerate(match_struc_data_df["data"]):
    match_explode_data_df = explode_data(df=match_struc_data_df, row_idx=idx, track_list=track_list)

match_explode_data_df.columns.values

match_struc_data_df[match_struc_data_df["55_x"].isna()]
match_struc_data_df[match_struc_data_df["home_team_x"].isna()]
match_struc_data_df[~match_struc_data_df["Manchester City_track_id"].isna()]
match_struc_data_df[~match_struc_data_df["Liverpool_track_id"].isna()]

match_info_dict.keys()
match_info_dict["ball"]
match_info_dict["players"]

## Assumptions:
## 1. Intent of the player is not captured.
## 2. The ball is not in every frame
## 3. The tracking is also not continuous. Assuming
## 4. Stationary/slow movement may not be because they are not moving fast. Maybe they are just defending
## On-ball movement vs Off-ball movement
