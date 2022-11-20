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
    "home_team": {},
    "away_team": {}
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
    player_mapping_list = []

    for player in match_info["players"]:
        player_mapping = []
        player_trackobj = player["trackable_object"]
        player_mapping.append(player_trackobj)

        first_name = player["first_name"].lower()
        last_name = player["last_name"].lower()
        if first_name:
            player_mapping.append(f"{first_name}_{last_name}")
        else:
            player_mapping.append(f"{last_name}")

        player_id = player["id"]
        player_mapping.append(player_id)

        player_team_trackobj = player["team_id"]
        if player_team_trackobj == home_team_trackobj:
            home_team_trackobj_list.append(player_trackobj)
        elif player_team_trackobj == away_team_trackobj:
            away_team_trackobj_list.append(player_trackobj)
        player_mapping_list.append(player_mapping)

    return home_team_trackobj_list, away_team_trackobj_list, player_mapping_list

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
            home_away_none = "home_team"
        elif player_trackobj in away_player_trackobj_list:
            home_away_none = "away_team"
        else:
            home_away_none = np.nan
        df.at[row_idx, f"{player_trackobj}_homeaway"] = home_away_none
        df.at[row_idx, "player_trackobj_captured"] = list(set(player_trackobj_in_frame_list))

        print(f'{track_id}, {x}, {y}, {player_trackobj_in_frame_list}')
    return df

def mt_to_sec(match_time: str):
    """
    Convert the match time from format (XX:XX.XX) to seconds
    """
    mins, secs_micro = match_time.split(":")
    secs, micro = secs_micro.split(".")
    total_secs = float(mins) * 60 + float(secs) + float(micro)/100
    return total_secs

def calc_dist(x1: float,
              y1: float,
              x2: float,
              y2: float):
    """
    Function to calculated the distance travelled from frame to frame based on
    """
    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return distance

def get_team_name(team: str,
                  match_info: dict):
    """
    Convert string of home_team / away_team to actual name of the team
    """
    assert team in ["home_team", "away_team"], f'{team} not in ["hohome_teamme", "away_team"]'
    return match_info[team]["short_name"].lower().replace(" ", "_")

def summarise_distance_time(df: pd.DataFrame,
                            frame_rate_smoothing_threshold: int=10,
                            time_per_frame_rate: float=0.10):
    """
    Summarise the distance ran by the player on a frame to frame basis
    Setting the default frame_smoothing_threshold to be 10

    Input:
        df: DataFrame that consists of each player's x, y coordinates
        frame_smoothing_threshold: Number of frames to smooth in the calculation
        frame_rate: Number of seconds for each frame

    Returns:
        Copy of the input dataframe with the following additional columns:
            1. {player_trackobj}_dist: Distance travelled
            2. {player_trackobj}_time: Number of seconds travelled
    """
    summary_df = pd.DataFrame()
    for period in [1, 2]:
        copy_df = df[df["period"] == period]
        total_time_record = len(df)

        for time_idx, time in zip(copy_df.index, copy_df["time"]):
            if time_idx < (total_time_record - frame_rate_smoothing_threshold):
                print(f"Frame {time_idx} / {total_time_record} @ time {time}")
                print("^" * 20)
                player_trackobj_in_frame_list = copy_df.at[time_idx, "player_trackobj_captured"]
                num_players_in_frame = len(player_trackobj_in_frame_list)
                for player_idx, player_trackobj in enumerate(player_trackobj_in_frame_list):
                    current_track_id = copy_df.at[time_idx, f"{player_trackobj}_track_id"]
                    forward_track_id = copy_df.at[time_idx + frame_rate_smoothing_threshold, f"{player_trackobj}_track_id"]
                    current_seconds = copy_df.at[time_idx, "time_seconds"]
                    forward_seconds = copy_df.at[time_idx + frame_rate_smoothing_threshold, "time_seconds"]

                    track_id_same_flag = current_track_id == forward_track_id
                    time_smoothing_same_flag = current_seconds + frame_rate_smoothing_threshold * time_per_frame_rate <= forward_seconds
                    ## Assumption: In some situation, the AI may not be able to capture the player properly.
                    ## However, from the continuation of motion, we know that it is still the same player
                    ## Thus, if the player is within the frame_rate_smoothing_threshold, we will still calculate it even if his/her track_id changes

                    if (player_trackobj in copy_df.at[time_idx + frame_rate_smoothing_threshold, "player_trackobj_captured"]) & \
                        (track_id_same_flag or time_smoothing_same_flag):
                        print(f"Frame {time_idx}: Player count {player_idx} / {num_players_in_frame} : {player_trackobj}")
                        print("*" * 5)
                        x1 = copy_df.at[time_idx, f"{player_trackobj}_x"]
                        x2 = copy_df.at[time_idx + frame_rate_smoothing_threshold, f"{player_trackobj}_x"]
                        y1 = copy_df.at[time_idx, f"{player_trackobj}_y"]
                        y2 = copy_df.at[time_idx + frame_rate_smoothing_threshold, f"{player_trackobj}_y"]
                        distance = calc_dist(x1=x1, y1=y1, x2=x2, y2=y2) / frame_rate_smoothing_threshold
                        copy_df.at[time_idx, f"{player_trackobj}_dist"] = distance
                        copy_df.at[time_idx, f"{player_trackobj}_time"] = time_per_frame_rate
        summary_df = pd.concat([summary_df, copy_df], axis=0).reset_index()

    return summary_df
