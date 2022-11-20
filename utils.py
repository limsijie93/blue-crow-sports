"""
Init date: 17th Nov 2022
Update date: 19th Nov 2022
Description: This script is for the utility functions used in the analysis for SkillCorner dataset found in the repo
SkillCorner/opendata: SkillCorner Open Data with 9 matches of broadcast tracking data. (github.com)
Github link: https://github.com/SkillCorner/opendata
Author: @sijielim
"""

import pandas as pd

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
    for tracked in track_list:
        if tracked.get("trackable_object"):
            track_obj = tracked.get("trackable_object")
            if track_obj == str(match_info["ball"]["trackable_object"]):
                df.loc[row_idx, f"{track_obj}_z"] = tracked.get("z")
        else:
            track_obj = tracked.get("group_name").replace(" ", "_").lower()

        df.loc[row_idx, f"{track_obj}_x"] = tracked.get("x")
        df.loc[row_idx, f"{track_obj}_y"] = tracked.get("y")
        df.loc[row_idx, f"{track_obj}_track_id"] = tracked.get("track_id")

        print(f'{df.loc[row_idx, f"{track_obj}_track_id"]}, {df.loc[row_idx, f"{track_obj}_x"]}, {df.loc[row_idx, f"{track_obj}_y"]}')
    return df
