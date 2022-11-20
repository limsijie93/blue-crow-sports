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

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from blue_crow_sports.utils import explode_data, extract_home_away_player_id

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
match_struc_data_df[["possession_player_id", "possession_homeaway"]
                    ] = pd.json_normalize(match_struc_data_df["possession"])
match_struc_data_df.drop(["possession"], axis=1, inplace=True)
match_struc_data_df["possession_homeaway"] = match_struc_data_df["possession_homeaway"].apply(
    lambda x: x.replace(" team", "") if x else x)

## There are certain frames where the group is None. Drop those rows where time == None
match_struc_data_df = match_struc_data_df[~match_struc_data_df["time"].isna()]
match_struc_data_df = match_struc_data_df.reset_index(drop=True)

match_struc_data_df["group"] = match_struc_data_df["group"].replace("home team", home_team)
match_struc_data_df["group"] = match_struc_data_df["group"].replace("away team", away_team)

match_struc_data_df["data_length"] = match_struc_data_df["data"].apply(lambda x: len(x))

match_struc_data_df["group"].value_counts()
match_struc_data_df["time"].value_counts().sort_index()

def extract_home_away_player_id(match_info: dict):
    """
    Function to extract a list of the player ids and
    whether they belong to the home or away team
    """
    home_team_id_list, away_team_id_list = [], []
    home_team_id = match_info["home_team"]["id"]
    away_team_id = match_info["away_team"]["id"]

    for player in match_info["players"]:
        player_id = player["id"]
        if player_id == home_team_id:
            home_team_id_list.append(player_id)
        elif player_id == away_team_id:
            away_team_id_list.append(player_id)
    return home_team_id_list, away_team_id_list

def explode_data(df: pd.DataFrame,
                 match_info: dict,
                 row_idx: int,
                 track_list: list):
    """
    Explode the list of dictionaries that are in the "data" column in the dataframe
    """
    home_player_id_list, away_player_id_list = extract_home_away_player_id(match_info)

    for tracked in track_list:
        if tracked.get("trackable_object"):
            player_id = tracked.get("trackable_object")
            if player_id == str(match_info["ball"]["trackable_object"]):
                df.loc[row_idx, f"{player_id}_z"] = tracked.get("z")
        else:
            player_id = tracked.get("group_name").replace(" ", "_").lower()

        x = tracked.get("x")
        y = tracked.get("y")
        track_id = tracked.get("track_id")

        df.loc[row_idx, f"{player_id}_x"] = x
        df.loc[row_idx, f"{player_id}_y"] = y
        df.loc[row_idx, f"{player_id}_track_id"] = track_id
        if player_id in home_player_id_list:
            home_away_none = "home"
        elif player_id in away_player_id_list:
            home_away_none = "away"
        else:
            home_away_none = np.nan
        df.loc[row_idx, f"{player_id}_homeaway"] = home_away_none

        print(f'{track_id}, {x}, {y}')
    return df


## Processing using a list comprehension to run it faster
for idx, track_list in enumerate(match_struc_data_df["data"]):
    match_explode_data_df = explode_data(df=match_struc_data_df,
                                         match_info=match_info_dict,
                                         row_idx=idx, track_list=track_list)
match_explode_data_df = match_explode_data_df.reindex(
    sorted(match_explode_data_df.columns), axis=1)



match_explode_data_df.columns.values

match_struc_data_df.loc[58416, "data"][0]
match_struc_data_df[~match_struc_data_df["possession"].isna()]
match_struc_data_df[match_struc_data_df["55_x"].isna()]
match_struc_data_df[~match_struc_data_df["home_team_x"].isna()]
match_struc_data_df[~match_struc_data_df["Manchester City_track_id"].isna()]
match_struc_data_df[~match_struc_data_df["Liverpool_track_id"].isna()]


match_info_dict.keys()
match_info_dict["home_team"]
match_info_dict["ball"]
match_info_dict["players"][0]

## Assumptions:
## 1. Intent of the player is not captured.
## 2. The ball is not in every frame
## 3. The tracking is also not continuous. Assuming
## 4. Stationary/slow movement may not be because they are not moving fast. Maybe they are just defending
## On-ball movement vs Off-ball movement
