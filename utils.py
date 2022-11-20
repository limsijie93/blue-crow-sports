"""
Init date: 17th Nov 2022
Update date: 20th Nov 2022
Description: This script is for the utility functions used in the analysis for SkillCorner dataset found in the repo
SkillCorner/opendata: SkillCorner Open Data with 9 matches of broadcast tracking data. (github.com)
Github link: https://github.com/SkillCorner/opendata
Author: @sijielim
"""

import math
import numpy as np
import pandas as pd

player_match_stat_template = {
    "home": {},
    "away": {}
}

player_stat_template = {
    "dist": 0, "time": 0, "speed": 0,
    "dist_onball": 0, "time_onball": 0, "speed_onball": 0,
    "dist_offball": 0, "time_offball": 0, "speed_offball": 0,
    "dist_teampos": 0, "time_teampos": 0, "speed_teampos": 0,
    "dist_teampos_onball": 0, "time_teampos_onball": 0, "speed_teampos_onball": 0,
    "dist_teampos_offball": 0, "time_teampos_offball": 0, "speed_teampos_offball": 0,
    "dist_teamnopos": 0, "time_teamnopos": 0, "speed_teamnopos": 0,
    "dist_teamnopos_offball": 0, "time_teamnopos_offball": 0, "speed_teamnopos_offball": 0
}

def extract_home_away_player_trackobj(match_info: dict):
    """
    Function to extract a list of the player ids and
    whether they belong to the home or away team
    """
    home_team_trackobj_list, away_team_trackobj_list = [], []
    home_team_trackobj = match_info["home_team"]["id"]
    away_team_trackobj = match_info["away_team"]["id"]
    trackobj_mapping_dict = {}

    for player in match_info["players"]:
        player_trackobj = player["trackable_object"]
        trackobj_mapping_dict[player_trackobj] = player["first_name"].lower() + "_" + player["last_name"].lower()
        player_team_trackobj = player["team_id"]
        if player_team_trackobj == home_team_trackobj:
            home_team_trackobj_list.append(player_trackobj)
        elif player_team_trackobj == away_team_trackobj:
            away_team_trackobj_list.append(player_trackobj)
    return home_team_trackobj_list, away_team_trackobj_list, trackobj_mapping_dict

def explode_data(df: pd.DataFrame,
                 match_info: dict,
                 row_idx: int,
                 track_list: list):
    """
    Explode the list of dictionaries that are in the "data" column in the dataframe
    """
    home_player_trackobj_list, away_player_trackobj_list, _ = extract_home_away_player_trackobj(match_info)
    full_player_trackobj_list = sorted(home_player_trackobj_list + away_player_trackobj_list)
    player_trackobj_in_frame_list = []

    for tracked in track_list:
        player_trackobj = tracked.get("trackable_object")
        if player_trackobj in full_player_trackobj_list:
            if player_trackobj == str(match_info["ball"]["trackable_object"]):
                df.at[row_idx, f"{player_trackobj}_z"] = tracked.get("z")
            player_trackobj_in_frame_list.append(player_trackobj)
        elif not player_trackobj:
            player_trackobj = tracked.get("group_name").replace(" ", "_").lower()

        x = tracked.get("x")
        y = tracked.get("y")
        track_id = tracked.get("track_id")

        df.at[row_idx, f"{player_trackobj}_x"] = x
        df.at[row_idx, f"{player_trackobj}_y"] = y
        df.at[row_idx, f"{player_trackobj}_track_id"] = track_id
        if player_trackobj in home_player_trackobj_list:
            home_away_none = "home"
        elif player_trackobj in away_player_trackobj_list:
            home_away_none = "away"
        else:
            home_away_none = np.nan
        df.at[row_idx, f"{player_trackobj}_homeaway"] = home_away_none
        df.at[row_idx, "player_trackobj_captured"] = list(set(player_trackobj_in_frame_list))

        print(f'{track_id}, {x}, {y}, {player_trackobj_in_frame_list}')
    return df

def calc_dist(x1: float,
              y1: float,
              x2: float,
              y2: float):
    """
    Function to calculated the distance travelled from frame to frame based on
    """
    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return distance

def summarise_distance_time(df: pd.DataFrame,
                            frame_threshold: int=10):
    """
    Summarise the distance ran by the player on a frame to frame basis
    Setting the default frame_threshold to be 10

    Returns:
        Copy of the input dataframe with the following additional columns:
            1. {player_trackobj}_dist: Distance travelled
            2. {player_trackobj}_time: Number of seconds travelled
    """
    copy_df = df.copy()
    total_time_record = len(df)

    for time_idx, time in enumerate(copy_df["time"]):
        if time_idx < (total_time_record - frame_threshold):
            print(f"Frame {time_idx} / {total_time_record} @ time {time}")
            print("^" * 20)
            player_trackobj_in_frame_list = copy_df.at[time_idx, "player_trackobj_captured"]
            num_players_in_frame = len(player_trackobj_in_frame_list)
            for player_idx, player_trackobj in enumerate(player_trackobj_in_frame_list):
                if player_trackobj in copy_df.at[time_idx + frame_threshold, "player_trackobj_captured"]:
                    print(f"Frame {time_idx}: Player count {player_idx} / {num_players_in_frame} : {player_trackobj}")
                    print("*" * 5)
                    x1 = copy_df.at[time_idx, f"{player_trackobj}_x"]
                    x2 = copy_df.at[time_idx + frame_threshold, f"{player_trackobj}_x"]
                    y1 = copy_df.at[time_idx, f"{player_trackobj}_y"]
                    y2 = copy_df.at[time_idx + frame_threshold, f"{player_trackobj}_y"]
                    distance = calc_dist(x1=x1, y1=y1, x2=x2, y2=y2)
                    copy_df.at[time_idx, f"{player_trackobj}_dist"] = distance
                    copy_df.at[time_idx, f"{player_trackobj}_time"] = frame_threshold * 0.10
    return copy_df
