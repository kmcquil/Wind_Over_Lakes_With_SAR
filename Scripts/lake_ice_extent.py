#################################################################################################
# Katie McQuillan
# Create dataset of lake ice presence/absence at lake daily from 2023- 2024
# https://land.copernicus.eu/en/products/water-bodies/lake-ice-extent-northern-hemisphere-500m#download
#################################################################################################

##################################################################################################
# Download the dataset from 2023-01-01 to 2024-06
# There was an api but it was giving very odd errors.
# Using the GUI only allowed download of 7 days at at time - outrageous
# So use the FTP with FileZilla (instructions at bottom of link below)
# https://land.copernicus.eu/en/products/water-bodies/lake-ice-extent-northern-hemisphere-500m#download
##################################################################################################

# Import modules
import os
import glob as glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray
from rasterstats import zonal_stats
import gc

# Set home directory
home = "C:/Users/kmcquil/Documents/SWOT_WIND/"

#################################################################################################
# Calculate categorical zonal stats of lake ice extent for each lake for each day
#################################################################################################

# Open the lake shapefile
pld = gpd.read_file(os.path.join(home, "Data/Buoy/pld_with_buoys.shp"))

# Loop through each lake extent nc file and extract percentage of each pixel catoegry in each lake
files = glob.glob(os.path.join(home, "Data/Lake_Ice/*.nc"))
# Create list to store lake ice extent info
zs_list = []
for file in files:
    # Open file and set crs
    ds = rioxarray.open_rasterio(file)
    ds.rio.write_crs("epsg:4326", inplace=True)
    # Select the Lake Ice Extent (LIE) layer
    lie = ds["LIE"]
    # Extract categorical zonal stats
    aff = lie.rio.transform()
    lie_np = lie.to_numpy()[0, :, :]
    stats = zonal_stats(pld, lie_np, affine=aff, categorical=True, nodata=0)
    # Convert to a df
    df = pd.DataFrame(stats)
    df["date"] = ds.attrs["time_coverage_start"][0:10]
    df["lake_id"] = pld["lake_id"]

    # Add to the list
    zs_list.append(df)

    # stop the memory from exploding
    ds.close()
    gc.collect()

# Create a df of the ice presence zonal stats and save
zs_df_list = [pd.DataFrame(i) for i in zs_list]
zs_df = pd.concat(zs_df_list, axis=0, join="outer").reset_index(drop=True)
zs_df.to_csv(os.path.join(home, "Data/Lake_Ice/lake_ice_pld.csv"))

#################################################################################################
# Calculate the percentage of each land cover
#################################################################################################

zs_df.columns = zs_df.columns.astype("str")
zs_df["total"] = zs_df[["50", "70", "10", "30", "40", "60"]].sum(axis=1)
# Lake Ice Extent (LIE): Ice cover = 10; Open water = 30; Cloud = 40; No data = 50; Sea pixel = 60; land pixel = 70
zs_df["ice_pct"] = zs_df["10"] / zs_df["total"]
zs_df["openwater_pct"] = zs_df["30"] / zs_df["total"]
zs_df["cloud_pct"] = zs_df["40"] / zs_df["total"]
zs_df["nodata_pct"] = zs_df["50"] / zs_df["total"]
zs_df["sea_pct"] = zs_df["60"] / zs_df["total"]
zs_df["land_pct"] = zs_df["70"] / zs_df["total"]
zs_df["date"] = pd.to_datetime(zs_df["date"])

#################################################################################################
# Summarize across the time period
zs_avgs = (
    zs_df[
        [
            "ice_pct",
            "openwater_pct",
            "cloud_pct",
            "nodata_pct",
            "sea_pct",
            "land_pct",
            "lake_id",
        ]
    ]
    .groupby("lake_id")
    .mean()
)

# Lake Murray never has ice and some open water. Its in SC, USA so that is fine and can stay all NAs
# Lake Greifen and Lake Ageris are both classified as land 100% of the time. Since these are in Switzerland, there could be some ice happening. Use Lake Murten as a proxy
# Lake Washington is classified as the sea. Lake Sammamish does get classified as open water but never ice. Use Lake Sammamish as a proxy for Lake Washington.

#################################################################################################
# For each lake, get the start and end of lake ice in 2023 and 2024 winters. Then calculate
# the first and last date of the winter when ice was detected as > 1%

# List the unique lakes amonst the buoys
lakes = zs_df["lake_id"].unique()

# Create an empty list to append ice info
lake_dfs = []

# Loop through lakes
for lake in lakes:

    # Subset to lake
    zs_df_lake = zs_df[zs_df["lake_id"] == lake]

    # Subset to days where ice was detected on at least 1% of the lake
    zs_df_lake_ice = zs_df_lake[zs_df_lake["ice_pct"] > 0.01]

    # Find first and last date of detected ice for jan 2023 winter
    winter_2023 = zs_df_lake_ice[zs_df_lake_ice["date"] < "2023-08-01"]
    if winter_2023.shape[0] == 0:
        winter_2023_start = np.nan
        winter_2023_end = np.nan
    else:
        winter_2023_start = winter_2023["date"].iloc[0]
        winter_2023_end = winter_2023["date"].iloc[-1]
    # Find first and last date of detected ice for Jan 2024 winter
    winter_2024 = zs_df_lake_ice[
        (
            (zs_df_lake_ice["date"] < "2024-08-01")
            & (zs_df_lake_ice["date"] >= "2023-08-01")
        )
    ]
    if winter_2024.shape[0] == 0:
        winter_2024_start = np.nan
        winter_2024_end = np.nan
    else:
        winter_2024_start = winter_2024["date"].iloc[0]
        winter_2024_end = winter_2024["date"].iloc[-1]
    df = pd.DataFrame(
        {
            "lake_id": [lake],
            "start_2023": [winter_2023_start],
            "end_2023": [winter_2023_end],
            "start_2024": [winter_2024_start],
            "end_2024": [winter_2024_end],
        }
    )
    # Add dates to list
    lake_dfs.append(df)

# Combine list to one df of ice ranges for each lake
lake_dfs = pd.concat(lake_dfs)

#################################################################################################
# Assign the lakes without info the closest lake info
# Lake Washington gets Lake Sammamish dates. Lake Sammish never has ice, so both Washingotn and Sammamish are NA across
# Lake Murray never has ice and can be left alone
# Lake Greifen and Aegeris get Lake Murten
# murten: 2320148382
# greifen: 2320146072
# aegeris: 2320145612
lake_dfs.loc[
    lake_dfs["lake_id"] == 2320146072,
    ["start_2023", "end_2023", "start_2024", "end_2024"],
] = lake_dfs.loc[
    lake_dfs["lake_id"] == 2320148382,
    ["start_2023", "end_2023", "start_2024", "end_2024"],
]  # lake greifen
lake_dfs.loc[
    lake_dfs["lake_id"] == 2320145612,
    ["start_2023", "end_2023", "start_2024", "end_2024"],
] = lake_dfs.loc[
    lake_dfs["lake_id"] == 2320148382,
    ["start_2023", "end_2023", "start_2024", "end_2024"],
]  # lake aegeris

# Save as csv
lake_dfs.to_csv(os.path.join(home, "Data/Lake_Ice/lake_ice_date_ranges.csv"))
