##############################################################################
##############################################################################
# Check the quality of the buoy data observations at each point 
# we use in the manuscript 
##############################################################################
##############################################################################

# Import modules
import os
import glob as glob
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import timedelta
import math
import itertools
import seaborn as sns
from matplotlib import pyplot as plt
import scipy
from rasterio.enums import Resampling
from scipy.stats import mannwhitneyu
import rioxarray
from geocube.vector import vectorize
import xarray
from rasterio.enums import Resampling

home = "C:/Users/kmcquil/Documents/SWOT_WIND/"

# wind_df = pd.read_csv(os.path.join(home, "Data/Wind_Direction/wdir_rana_df.csv"))
wind_df = pd.read_csv(
    os.path.join(home, "Data/Wind_Direction/wdir_rana_df_updated.csv")
)

# Subset to observations less than 1 hour between buoy and satellite observation
wind_df["time_diff"] = pd.to_timedelta(wind_df["time_diff"])
wind_df = wind_df.dropna(subset=["wdir_corrected", "wdir_buoy"])
wind_df = wind_df[abs(wind_df["time_diff"]) < timedelta(hours=1)]

# Select wind direction estiamted at 1km resolutioni
wind_df = wind_df[wind_df["wdir_id"] == "wdir_10pixels_"]

# Don't include the cal/val swot observations
wind_df = wind_df[wind_df["satellite"] != "SWOT_L2_HR_Raster_1.1"]

# Bring in the wind streak presence/absence and merge
wind_streak_check = pd.read_csv(
    os.path.join(home, "Data/Wind_Streaks/wind_streak_check.csv")
)
wind_df = wind_df.merge(
    wind_streak_check[["buoy_id", "image_id", "buoy_image_id", "wind_streak"]],
    on=["buoy_id", "image_id"],
    how="left",
)

# Drop when the lake had ice
lake_ice = pd.read_csv(os.path.join(home, "Data/Lake_Ice/lake_ice_date_ranges.csv"))
lake_ice["lake_id"] = lake_ice["lake_id"].astype("Int64").astype(str)
buoy_info = gpd.read_file(os.path.join(home, "Data/Buoy/buoy_info.shp"))
wind_df = wind_df.merge(
    buoy_info[["id", "pld_id"]].rename(columns={"id": "buoy_id", "pld_id": "lake_id"}),
    on="buoy_id",
)
wind_df = wind_df.merge(lake_ice, on="lake_id")
idx = []
for i in range(0, wind_df.shape[0]):
    row = wind_df.iloc[i]
    if (
        pd.isnull(row.start_2023)
        & pd.isnull(row.end_2023)
        & pd.isnull(row.start_2024)
        & pd.isnull(row.end_2024)
    ):
        idx.append(i)
    elif pd.isnull(row.start_2023) & pd.isnull(row.end_2023):
        if (row.overpass_datetime < row.start_2024) | (
            row.overpass_datetime > row.end_2024
        ):
            idx.append(i)
    elif pd.isnull(row.start_2024) & pd.isnull(row.end_2024):
        if (row.overpass_datetime < row.start_2023) | (
            row.overpass_datetime > row.end_2023
        ):
            idx.append(i)
    elif (
        (row.overpass_datetime < row.start_2023)
        | (row.overpass_datetime > row.end_2023)
    ) & (
        (row.overpass_datetime < row.start_2024)
        | (row.overpass_datetime > row.end_2024)
    ):
        idx.append(i)
wind_df = wind_df.iloc[idx]
wind_df = wind_df.drop(
    columns=[
        "Unnamed: 0_x",
        "Unnamed: 0_y",
        "start_2023",
        "end_2023",
        "start_2024",
        "end_2024",
    ]
).reset_index(drop=True)
# Dropped 32 observations because of ice

# convert distance and fetch to km
wind_df["distance"] = wind_df["distance"] / 1000
wind_df["fetch_lake"] = wind_df["fetch_lake"] / 1000
wind_df["fetch_buoy"] = wind_df["fetch_buoy"] / 1000

# Rename attributes for prettier plotting
wind_df["wind_streak"] = wind_df["wind_streak"].map({0: "No", 1: "Yes"})
wind_df["satellite"] = wind_df["satellite"].map(
    {"Sentinel1": "S1", "SWOT_L2_HR_Raster_2.0": "SWOT"}
)

# drop  buchillon station
wind_df = wind_df[wind_df["buoy_id"] != "buchillonfieldstation"]

# drop rows that are s1 but don't have wind speed estiamted.
wind_df = wind_df[
    ~((wind_df["wspd_sat_cmod5n"].isnull()) & (wind_df["satellite"] == "S1"))
]


buoys_to_check = ['Sammamish', 'Washington', 'lakegreifenmeteostation', 'lexploremeteostation','meteostationlakeaegeri', 'meteostationlakemurten']
wind_df_subset = wind_df[wind_df['buoy_id'].isin(buoys_to_check)]
wind_df_subset['id'] = range(0, wind_df_subset.shape[0])

for i in range(0, wind_df_subset.shape[0]):
    id = wind_df_subset['id'].iloc[i]
    buoy_id = wind_df_subset['buoy_id'].iloc[i]
    buoy_dt = wind_df_subset['buoy_datetime'].iloc[i]
    bouy_dt_fixed = buoy_dt[0:10] + "T" + buoy_dt[11:19] + "Z"
    buoy_df = pd.read_csv(os.path.join(home, "Data/Buoy/Processed", buoy_id + ".csv"))
    buoy_df['datetime'] = pd.to_datetime(buoy_df['datetime'], format='mixed').dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    match_index = buoy_df.index[buoy_df['datetime'] == bouy_dt_fixed].tolist()[0]
    buoy_df_subset = buoy_df.iloc[match_index-5:match_index+5, :]
    
    fig, ((ax1, ax2)) = plt.subplots(2, 1, figsize=(10, 10))
    sns.lineplot(
        ax=ax1,
        x="datetime",
        y="wspd",
        data=buoy_df_subset,
    )
    ax1.tick_params(axis='x', rotation=15)
    ax1.set_xlabel("Date", size=12)
    ax1.set_ylabel("Wind speed (m/s)", size=12)

    sns.lineplot(
        ax=ax2,
        x="datetime",
        y="wdir",
        data=buoy_df_subset,
    )
    ax2.tick_params(axis='x', rotation=15)
    ax2.set_xlabel("Date", size=12)
    ax2.set_ylabel("Wind direction (degrees)", size=12)
    plt.savefig(os.path.join(home, "Data/Buoy/QC_Checks/" + str(id) +  ".png"), dpi=300)