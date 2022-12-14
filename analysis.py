"""
Init date: 17th Nov 2022
Update date: 20th Nov 2022
Description: This script is an analysis for the SkillCorner dataset found in the repo
SkillCorner/opendata: SkillCorner Open Data with 9 matches of broadcast tracking data. (github.com)
Github link: https://github.com/SkillCorner/opendata
Author: @sijielim
"""

##################### ASSUMPTIONS #####################

## Assumptions:
## 1. Intent of the player is not captured.
## 2. The ball is not in every frame
## 3. The tracking is also not continuous. Assuming
## 4. Stationary/slow movement may not be because they are not moving fast. Maybe they are just defending
## On-ball movement vs Off-ball movement
## 5. There are some stoppage time moments that are also captured. In those moments, the players are moving, but they are not in a competitive mode.
## This obscures their speed because there's no real intention to be fast.
## Do we need to account for the focal length of the camera? Are we making too much assumptions about the data?
## 6. Sideway movements are not well-captured


import json
import os

import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

from blue_crow_sports.utils import (explode_data,
                                    extract_home_away_player_trackobj,
                                    get_team_name, mt_to_sec,
                                    player_match_stat_template,
                                    player_stat_template,
                                    summarise_distance_time)

load_dotenv("blue_crow_sports/.env")

ROOT = os.getenv("ROOT")
IN_DATA_DIR = os.path.join(ROOT, "opendata", "data")
INPUT_JSON_PATH = os.path.join(IN_DATA_DIR, "matches.json")
OUT_DATA_DIR = os.path.join(ROOT, "blue_crow_sports", "plotly")
os.makedirs(OUT_DATA_DIR, exist_ok=True)

# Threshold number of frames to consider as continous movement
FRAME_RATE_SMOOTHING_THRESHOLD = 1
TIME_PER_FRAME_RATE = 0.10

file_df = pd.read_json(INPUT_JSON_PATH)

all_player_stat_summary_df = pd.DataFrame()
for match_metadata in file_df.values.tolist():
    print(f"Processing {match_metadata}")
    # match_metadata = file_df.values.tolist()[0]
    match_status, match_dt, home_team, away_team, match_id = match_metadata
    home_team = home_team["short_name"]
    away_team = away_team["short_name"]

    match_data_dir = os.path.join(IN_DATA_DIR, "matches", str(match_id))
    match_data_json_path = os.path.join(match_data_dir, "match_data.json")
    match_structured_data_json_path = os.path.join(match_data_dir, "structured_data.json")

    with open(match_data_json_path, "r") as f:
        match_info_dict = json.load(f)
    match_struc_data_df = pd.read_json(match_structured_data_json_path)
    match_struc_data_df[["possession_player_trackobj", "possession_homeaway"]] = pd.json_normalize(match_struc_data_df["possession"])
    match_struc_data_df.drop(["possession"], axis=1, inplace=True)
    match_struc_data_df["possession_homeaway"] = match_struc_data_df["possession_homeaway"].apply(lambda x: x.replace(" ", "_") if x else x)

    match_struc_data_df[~match_struc_data_df["possession_player_trackobj"].isna()]["possession_player_trackobj"].value_counts()

    ## There are certain frames where the group is None. Drop those rows where time == None
    match_struc_data_df = match_struc_data_df[~match_struc_data_df["time"].isna()]
    match_struc_data_df = match_struc_data_df.reset_index(drop=True)

    match_struc_data_df["time_seconds"] = match_struc_data_df["time"].apply(mt_to_sec)
    match_struc_data_df["data_length"] = match_struc_data_df["data"].apply(lambda x: len(x))
    match_struc_data_df["player_trackobj_captured"] = [[]] * len(match_struc_data_df)

    home_player_trackobj_list, away_player_trackobj_list, player_mapping_list = extract_home_away_player_trackobj(match_info=match_info_dict)

    ## Explode the data column into individual column for each player using the trackable object id
    for idx, track_list in enumerate(match_struc_data_df["data"]):
        match_explode_data_df = explode_data(df=match_struc_data_df,
                                            match_info=match_info_dict,
                                            row_idx=idx, track_list=track_list)
    match_explode_data_df["num_player_captured"] = match_explode_data_df["player_trackobj_captured"].apply(lambda x: len(x))
    match_explode_data_df = match_explode_data_df.reindex(
        sorted(match_explode_data_df.columns), axis=1)

    ## There are some duplicated rows in the data. We will remove those rows
    # match_explode_data_df[match_explode_data_df["time"].duplicated()]
    # match_explode_data_df[match_explode_data_df["time"] == "45:00.00"]

    ## Summarise the distance travelled by each player from frame to frame
    match_player_stats_data_df = summarise_distance_time(
        df=match_explode_data_df,
        frame_rate_smoothing_threshold=FRAME_RATE_SMOOTHING_THRESHOLD,
        time_per_frame_rate=TIME_PER_FRAME_RATE)
    match_player_stats_data_df = match_player_stats_data_df.reindex(
        sorted(match_player_stats_data_df.columns), axis=1)

    ## Calculate:
    ## 1. Distance: Total, Onball, Offball
    ## 2. Distance when team in possession: Total, Onball, Offball
    ## 3. Distance when team NOT in possession: Total, Offball

    player_match_stat = dict(player_match_stat_template)
    stat_summary_df = pd.DataFrame()

    for player_idx, (home_player_trackobj, away_player_trackobj) in enumerate(
        zip(home_player_trackobj_list, away_player_trackobj_list)):

        player_match_stat["home_team"][home_player_trackobj] = player_stat_template
        player_match_stat["away_team"][away_player_trackobj] = player_stat_template

        for team_pos, player_trackobj in zip(["home_team", "away_team"], [home_player_trackobj, away_player_trackobj]):
            if f"{player_trackobj}_dist" in match_player_stats_data_df.columns:
                # print(f"Process player_trackobj: {player_trackobj}")

                ## Filter for subset where player is in view
                player_match_stat_df = match_player_stats_data_df[
                    ~match_player_stats_data_df[f"{player_trackobj}_dist"].isna()]

                ############################################################

                ## 1a. Distance travelled: Total

                dist = player_match_stat_df[f"{player_trackobj}_dist"].sum()
                time = player_match_stat_df[f"{player_trackobj}_time"].sum()

                player_match_stat[team_pos][player_trackobj]["dist"] = dist
                player_match_stat[team_pos][player_trackobj]["time"] = time
                player_match_stat[team_pos][player_trackobj]["speed"] = dist / time

                ## 1b, c. Distance travelled: Onball, Offball
                player_match_stat_df[f"{player_trackobj}_onball"] = player_match_stat_df["possession_player_trackobj"].apply(
                    lambda x, player_trackobj=player_trackobj: 1 if x == player_trackobj else 0)

                #### Onball movement
                player_match_stat_df[f"{player_trackobj}_dist_onball"] = player_match_stat_df[f"{player_trackobj}_onball"] * player_match_stat_df[f"{player_trackobj}_dist"]
                player_match_stat_df[f"{player_trackobj}_time_onball"] = player_match_stat_df[f"{player_trackobj}_onball"] * player_match_stat_df[f"{player_trackobj}_time"]

                dist_onball = player_match_stat_df[f"{player_trackobj}_dist_onball"].sum()
                time_onball = player_match_stat_df[f"{player_trackobj}_time_onball"].sum()

                player_match_stat[team_pos][player_trackobj]["dist_onball"] = dist_onball
                player_match_stat[team_pos][player_trackobj]["time_onball"] = time_onball
                player_match_stat[team_pos][player_trackobj]["speed_onball"] = dist_onball / time_onball

                #### Offball movement
                player_match_stat_df[f"{player_trackobj}_dist_offball"] = player_match_stat_df[f"{player_trackobj}_dist"] - player_match_stat_df[f"{player_trackobj}_dist_onball"].fillna(0)
                player_match_stat_df[f"{player_trackobj}_time_offball"] = player_match_stat_df[f"{player_trackobj}_time"] - player_match_stat_df[f"{player_trackobj}_time_onball"].fillna(0)

                dist_offball = player_match_stat_df[f"{player_trackobj}_dist_offball"].sum()
                time_offball = player_match_stat_df[f"{player_trackobj}_time_offball"].sum()

                player_match_stat[team_pos][player_trackobj]["dist_offball"] = dist_offball
                player_match_stat[team_pos][player_trackobj]["time_offball"] = time_offball
                player_match_stat[team_pos][player_trackobj]["speed_offball"] = dist_offball / time_offball

                #### Counter check
                assert abs(dist - dist_onball - dist_offball) <= 1, f"Distance for {player_trackobj} doesn't tally: {dist_onball} vs {dist_offball} vs {dist}"
                assert abs(time - time_onball - time_offball) <= 1, f"Time for {player_trackobj} doesn't tally: {time_onball} vs {time_offball} vs {time}"
                # print(f"Player: {player_trackobj}. Dist: {dist}. Time {time}. Speed {dist/time}")

                ############################################################

                ## Distance travelled when the player's team is in possession
                player_match_stat_df[f"{team_pos}_possession"] = player_match_stat_df["possession_homeaway"].apply(
                    lambda x, team_pos=team_pos: 1 if x == team_pos else 0)
                player_match_stat_df[f"{player_trackobj}_{team_pos}"] = player_match_stat_df[f"{player_trackobj}_homeaway"].apply(
                    lambda x, team_pos=team_pos: 1 if x == team_pos else 0)

                ## 2a. Distance travelled while team in posession: Total
                player_match_stat_df[f"{player_trackobj}_dist_teampos"] = player_match_stat_df[f"{team_pos}_possession"] * \
                    player_match_stat_df[f"{player_trackobj}_{team_pos}"] * player_match_stat_df[f"{player_trackobj}_dist"]

                player_match_stat_df[f"{player_trackobj}_time_teampos"] = player_match_stat_df[f"{team_pos}_possession"] * \
                    player_match_stat_df[f"{player_trackobj}_{team_pos}"] * player_match_stat_df[f"{player_trackobj}_time"]

                dist_teampos = player_match_stat_df[f"{player_trackobj}_dist_teampos"].sum()
                time_teampos = player_match_stat_df[f"{player_trackobj}_time_teampos"].sum()

                player_match_stat[team_pos][player_trackobj]["dist_teampos"] = dist_teampos
                player_match_stat[team_pos][player_trackobj]["time_teampos"] = time_teampos
                player_match_stat[team_pos][player_trackobj]["speed_teampos"] = dist_teampos / time_teampos

                ## 2b, c. Distance travelled while team in posession: Onball, Offball
                for metric in ["dist", "time"]:
                    player_match_stat_df[f"{player_trackobj}_{metric}_teampos_onball"] = player_match_stat_df[
                        [f"{team_pos}_possession", f"{player_trackobj}_{team_pos}", f"{player_trackobj}_onball", f"{player_trackobj}_{metric}"]
                    ].prod(axis=1)

                    player_match_stat_df[f"{player_trackobj}_{metric}_teampos_offball"] = player_match_stat_df[f"{player_trackobj}_{metric}_teampos"] - \
                        player_match_stat_df[f"{player_trackobj}_{metric}_teampos_onball"].fillna(0)

                dist_teampos_onball = player_match_stat_df[f"{player_trackobj}_dist_teampos_onball"].sum()
                time_teampos_onball = player_match_stat_df[f"{player_trackobj}_time_teampos_onball"].sum()

                player_match_stat[team_pos][player_trackobj]["dist_teampos_onball"] = dist_teampos_onball
                player_match_stat[team_pos][player_trackobj]["time_teampos_onball"] = time_teampos_onball
                player_match_stat[team_pos][player_trackobj]["speed_teampos_onball"] = dist_teampos_onball / time_teampos_onball

                dist_teampos_offball = player_match_stat_df[f"{player_trackobj}_dist_teampos_offball"].sum()
                time_teampos_offball = player_match_stat_df[f"{player_trackobj}_time_teampos_offball"].sum()

                player_match_stat[team_pos][player_trackobj]["dist_teampos_offball"] = dist_teampos_offball
                player_match_stat[team_pos][player_trackobj]["time_teampos_offball"] = time_teampos_offball
                player_match_stat[team_pos][player_trackobj]["speed_teampos_offball"] = dist_teampos_offball / time_teampos_offball

                #### Counter check
                dist_diff = abs(dist_teampos - dist_teampos_onball - dist_teampos_offball)
                time_diff = abs(time_teampos - time_teampos_onball - time_teampos_offball)

                assert dist_diff <= 1 , f"Distance in possession for {player_trackobj} doesn't tally: {dist_diff}. {dist_teampos_onball} vs {dist_teampos_offball} vs {dist_teampos}"
                assert time_diff <= 1, f"Time in possession for {player_trackobj} doesn't tally: {time_diff}. {time_teampos_onball} vs {time_teampos_offball} vs {time_teampos}"

                # ####################################################################################

                ## 3. Calculate distance travelled when the player's team is NOT in possession
                #### Note: Under this scenario, player can only be offball

                dist_teamnopos = dist - dist_teampos
                time_teamnopos = time - time_teampos
                player_match_stat[team_pos][player_trackobj]["dist_teamnopos"] = dist_teamnopos
                player_match_stat[team_pos][player_trackobj]["time_teamnopos"] = time_teamnopos
                player_match_stat[team_pos][player_trackobj]["speed_teamnopos"] = dist_teamnopos / time_teamnopos

                player_match_stat[team_pos][player_trackobj]["dist_teamnopos_offball"] = dist_teamnopos
                player_match_stat[team_pos][player_trackobj]["time_teamnopos_offball"] = time_teamnopos
                player_match_stat[team_pos][player_trackobj]["speed_teamnopos_offball"] = dist_teamnopos / time_teamnopos

                assert  abs(dist - dist_teamnopos - dist_teampos) <= 1e-1, f"Distance when team not in posession is wrong. {player_trackobj}: {dist} {dist_teamnopos} + {dist_teampos}"
                assert  abs(time - time_teamnopos - time_teampos) <= 1e-1, f"Time when team not in posession is wrong. {player_trackobj}: {time} {time_teamnopos} + {time_teampos}"
                # ############################################################

                player_summarised_stat_df = pd.DataFrame(player_match_stat[team_pos][player_trackobj], index=[player_trackobj])
                player_summarised_stat_df["team"] = team_pos
                stat_summary_df = pd.concat([stat_summary_df, player_summarised_stat_df], axis=0)
    stat_summary_df.sort_values(["speed"], ascending=False, inplace=True)

    player_map_df = pd.DataFrame(player_mapping_list, columns=["track_id", "name", "player_id"])
    player_map_df.set_index("track_id", inplace=True)
    player_stats_summary_df = stat_summary_df.merge(player_map_df, left_index=True, right_index=True)

    player_stats_summary_df["team"] = player_stats_summary_df["team"].apply(get_team_name, match_info=match_info_dict)

    all_player_stat_summary_df = pd.concat([all_player_stat_summary_df, player_stats_summary_df], axis=0).reset_index(drop=True)
    print(f"*" * 50)

##################### SUMMARISE #####################

## Summarise the all_player_stat_summary_df as some players can be found in multiple matches
charts_to_plot_list = []
col_set = []
for idx, col in enumerate(player_stat_template.keys()):
    if idx % 3 == 0 and idx > 0:
        charts_to_plot_list.append(col_set)
        col_set = []
    col_set.append(col)

## NOTE: As some players played 2 matches, we will find the average distance they covered across the 2 matches
## The code below is to account for such cases
dedup_player_stat_summary_df = all_player_stat_summary_df.groupby(["player_id", "name", "team"]).sum().reset_index()
match_per_player_df = all_player_stat_summary_df.groupby(["player_id", "name", "team"])["dist"].count().reset_index()
match_per_player_df.rename(columns={"dist": "match_count"}, inplace=True)

dedup_player_stat_summary_df = dedup_player_stat_summary_df.merge(match_per_player_df, how="inner", on=["player_id", "name", "team"])

for col_set in charts_to_plot_list:
    dist, time, speed = col_set
    dedup_player_stat_summary_df[dist] = dedup_player_stat_summary_df[dist] / dedup_player_stat_summary_df["match_count"]
    dedup_player_stat_summary_df[time] = dedup_player_stat_summary_df[time] / dedup_player_stat_summary_df["match_count"]
    dedup_player_stat_summary_df[speed] = dedup_player_stat_summary_df[dist] / dedup_player_stat_summary_df[time]


##################### VISUALISATION #####################
for col_set in charts_to_plot_list:
    x, z, y = col_set
    fig = px.scatter(dedup_player_stat_summary_df,
                     x=x, y=y, size=z, color="team",
                     title=f"{y} (y-axis) vs {x} (x-axis)",
                     text="name", symbol="match_count",
                     hover_data={
                        "name": True, "player_id": True, "team": True,
                        x:":.2f", y:":.2f", z:":.2f",
                        "match_count": True
                     })
    fig.update_traces(textposition='middle right', textfont={"size": 8})
    fig.update_layout(hoverlabel={"bgcolor": "white",
                                  "font_size": 12})
    fig.write_html(os.path.join(OUT_DATA_DIR, f"{y}-vs-{x}.html"))
    fig.show()


##################### CHARTS (WORKINGS) #####################

# ax = sns.scatterplot(data=player_stats_summary_df, x=x, y=y, hue="team")
# plt.title(f"{y} (y-axis) vs {x} (x-axis)")
# for i, point in player_stats_summary_df.iterrows():
#     player_name = point['name']
#     player_id = point['player_id']
#     point_y = point[y]
#     point_x = point[x] / 1000
#     ax.text(point_x * 1000 * 1.01, point_y * 1.01,
#             f"{player_name} ({player_id})\n{point_y:.2f} m/s, \n{point_x:.0f} km", size=5)
# ax.legend()
# plt.show()
