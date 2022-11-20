"""
Init date: 17th Nov 2022
Update date:
Description: This script is an analysis for the SkillCorner dataset found in the repo
SkillCorner/opendata: SkillCorner Open Data with 9 matches of broadcast tracking data. (github.com)
Github link: https://github.com/SkillCorner/opendata
Author: @sijielim
"""

import math
import json
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from blue_crow_sports.utils import explode_data, summarise_distance_time

load_dotenv("blue_crow_sports/.env")

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

match_struc_data_df["data_length"] = match_struc_data_df["data"].apply(lambda x: len(x))
match_struc_data_df["player_id_captured"] = [[]] * len(match_struc_data_df)

## Processing using a list comprehension to run it faster
for idx, track_list in enumerate(match_struc_data_df["data"]):
    match_explode_data_df = explode_data(df=match_struc_data_df,
                                         match_info=match_info_dict,
                                         row_idx=idx, track_list=track_list)
match_explode_data_df["num_player_captured"] = match_explode_data_df["player_id_captured"].apply(lambda x: len(x))
match_explode_data_df = match_explode_data_df.reindex(
    sorted(match_explode_data_df.columns), axis=1)

frame_threshold = 10 # Threshold number of frames to consider as continous movement


match_player_stats_data_df = summarise_distance_time(df=match_explode_data_df, frame_threshold=10)
match_player_stats_data_df = match_player_stats_data_df.reindex(
    sorted(match_player_stats_data_df.columns), axis=1)

##################### WORKINGS #####################

match_player_stats_data_df.columns.values

len(match_struc_data_df.at[time_idx, "player_id_captured"])
len(set(match_struc_data_df.at[time_idx, "player_id_captured"]))
match_explode_data_df.columns.values

match_struc_data_df["group"].value_counts()
match_struc_data_df["time"].value_counts().sort_index()

match_struc_data_df.at[58416, "data"][0]

player_id = "2792"
match_struc_data_df[~match_struc_data_df[f"{player_id}_dist"].isna()][f"{player_id}_dist"].sum()

match_struc_data_df[match_struc_data_df["possession_homeaway"].isna()]
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
## 5. There are some stoppage time moments that are also captured. In those moments, the players are moving, but they are not in a competitive mode.
## This obscures their speed because there's no real intention to be fast.
