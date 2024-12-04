##########################################################################################
# Katie Mcquillan
# 04/30/2023
# Performance stats of wind field at buoys
##########################################################################################

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

###############################################################################################
# Function to calculate performance metrics
###############################################################################################

def calc_performance_stats_180(df):
    """
    Calculate bias, mae, rmse for wind direction and wind speed
    df: pd df
    return df of metrics
    """
    def calc_bias(o, p, m):
        o_upper = (o + 180) % 360
        if o_upper < o:
            if (p > o) | (p < o_upper):
                return m
            else:
                return m * -1
        if o_upper > o:
            if (p > o) & (p < o_upper):
                return m
            else:
                return m * -1
    # LG Wind Direction with aliased wind direction by using era5
    sub = df.dropna(subset=["wdir_corrected", "wdir_buoy"])
    if sub.shape[0] < 3:
        row1 = pd.DataFrame(
            [["WDIR-LG", sub.shape[0], np.nan, np.nan, np.nan]],
            columns=["ID", "N", "Bias", "MAE", "RMSE"],
        )
    else:
        obs = sub["wdir_buoy"].tolist()
        pred = sub["wdir_corrected"].tolist()
        # MAE. Get the smallest difference
        d1 = [(p_i - o_i) % 360 for p_i, o_i in zip(pred, obs)]
        d2 = [360 - i for i in d1]
        mae = [min(i, j) for i, j in zip(d1, d2)]
        # Bias. If the predicted is within + 180 from the observed, it is positive.
        # If the predicted is within - 180 from the observed, it is negative
        bias = [calc_bias(o, p, m) for o, p, m in zip(obs, pred, mae)]
        # Root mean square error
        rmse = math.sqrt(np.square(bias).mean())
        bias = np.mean(bias)
        mae = np.mean(mae)
        row1 = pd.DataFrame(
            [["WDIR-LG", sub.shape[0], bias, mae, rmse]],
            columns=["ID", "N", "Bias", "MAE", "RMSE"],
        )
    # LG Wind direction just seeing how far apart they are in terms of 180
    if sub.shape[0] < 3:
        row6 = pd.DataFrame(
            [["WDIR-LG-180", sub.shape[0], np.nan, np.nan, np.nan]],
            columns=["ID", "N", "Bias", "MAE", "RMSE"],
        )
    else:
        obs = sub["wdir_buoy"].tolist()
        pred = sub["wdir"].tolist()
        pred_180 = [(p + 180) % 360 for p in pred]
        # MAE. Get the smallest difference using pred and pred_180
        d1 = [(p_i - o_i) % 360 for p_i, o_i in zip(pred, obs)]
        d2 = [360 - i for i in d1]
        d1_180 = [(p_i - o_i) % 360 for p_i, o_i in zip(pred_180, obs)]
        d2_180 = [360 - i for i in d1_180]
        mae = [min([i, j, x, y]) for i, j, x, y in zip(d1, d2, d1_180, d2_180)]
        mae = np.mean(mae)
        row6 = pd.DataFrame(
            [["WDIR-LG-180", sub.shape[0], np.nan, mae, np.nan]],
            columns=["ID", "N", "Bias", "MAE", "RMSE"],
        )
    # ERA5 Wind Direction. Use the same subset as for the LG wind direction so it's a direct comparison
    if sub.shape[0] < 3:
        row2 = pd.DataFrame(
            [["WDIR-ERA5", sub.shape[0], np.nan, np.nan, np.nan]],
            columns=["ID", "N", "Bias", "MAE", "RMSE"],
        )
    else:
        obs = sub["wdir_buoy"].tolist()
        pred = sub["wdir_era5"].tolist()
        # MAE. Get the smallest difference
        d1 = [(p_i - o_i) % 360 for p_i, o_i in zip(pred, obs)]
        d2 = [360 - i for i in d1]
        mae = [min(i, j) for i, j in zip(d1, d2)]
        # Bias. If the predicted is within + 180 from the observed, it is positive.
        # If the predicted is within - 180 from the observed, it is negative
        bias = [calc_bias(o, p, m) for o, p, m in zip(obs, pred, mae)]
        # Root mean square error
        rmse = math.sqrt(np.square(bias).mean())
        bias = np.mean(bias)
        mae = np.mean(mae)
        row2 = pd.DataFrame(
            [["WDIR-ERA5", sub.shape[0], bias, mae, rmse]],
            columns=["ID", "N", "Bias", "MAE", "RMSE"],
        )
    # LG Wind direction just seeing how far apart they are in terms of 180
    if sub.shape[0] < 3:
        row7 = pd.DataFrame(
            [["WDIR-ERA5-180", sub.shape[0], np.nan, np.nan, np.nan]],
            columns=["ID", "N", "Bias", "MAE", "RMSE"],
        )
    else:
        obs = sub["wdir_buoy"].tolist()
        pred = sub["wdir_era5"].tolist()
        pred_180 = [(p + 180) % 360 for p in pred]
        # MAE. Get the smallest difference using pred and pred_180
        d1 = [(p_i - o_i) % 360 for p_i, o_i in zip(pred, obs)]
        d2 = [360 - i for i in d1]
        d1_180 = [(p_i - o_i) % 360 for p_i, o_i in zip(pred_180, obs)]
        d2_180 = [360 - i for i in d1_180]
        mae = [min([i, j, x, y]) for i, j, x, y in zip(d1, d2, d1_180, d2_180)]
        mae = np.mean(mae)
        row7 = pd.DataFrame(
            [["WDIR-ERA5-180", sub.shape[0], np.nan, mae, np.nan]],
            columns=["ID", "N", "Bias", "MAE", "RMSE"],
        )
    # CMOD5.n LG wind speed
    #sub = df.dropna(subset=["wspd_sat_cmod5n", "wspd_buoy"])
    sub = df.dropna(subset=["wspd_sat_cmod5n", "buoy_wspd_10m"])
    # Don't include buchillon field station
    sub = sub[sub["buoy_id"] != "buchillonfieldstation"]
    if sub.shape[0] < 3:
        row3 = pd.DataFrame(
            [["WSPD-CMOD5-SAR", sub.shape[0], np.nan, np.nan, np.nan]],
            columns=["ID", "N", "Bias", "MAE", "RMSE"],
        )
    else:
        #obs = sub["wspd_buoy"].tolist()
        obs = sub["buoy_wspd_10m"].tolist()
        pred = sub["wspd_sat_cmod5n"].tolist()
        bias = np.mean(np.subtract(pred, obs))
        mae = np.mean(np.abs(np.subtract(pred, obs)))
        rmse = math.sqrt(np.square(np.subtract(pred, obs)).mean())
        row3 = pd.DataFrame(
            [["WSPD-CMOD5-SAR", sub.shape[0], bias, mae, rmse]],
            columns=["ID", "N", "Bias", "MAE", "RMSE"],
        )
    # CMOD5.n ERA5 wind speed
    if sub.shape[0] < 3:
        row4 = pd.DataFrame(
            [["WSPD-CMOD5-ERA5", sub.shape[0], np.nan, np.nan, np.nan]],
            columns=["ID", "N", "Bias", "MAE", "RMSE"],
        )
    else:
        #obs = sub["wspd_buoy"].tolist()
        obs = sub["buoy_wspd_10m"].tolist()
        pred = sub["wspd_era5_cmod5n"].tolist()
        bias = np.mean(np.subtract(pred, obs))
        mae = np.mean(np.abs(np.subtract(pred, obs)))
        rmse = math.sqrt(np.square(np.subtract(pred, obs)).mean())
        row4 = pd.DataFrame(
            [["WSPD-CMOD5-ERA5", sub.shape[0], bias, mae, rmse]],
            columns=["ID", "N", "Bias", "MAE", "RMSE"],
        )
    # ERA5 wind speed
    if sub.shape[0] < 3:
        row5 = pd.DataFrame(
            [["WSPD-ERA5", sub.shape[0], np.nan, np.nan, np.nan]],
            columns=["ID", "N", "Bias", "MAE", "RMSE"],
        )
    else:
        #obs = sub["wspd_buoy"].tolist()
        obs = sub["buoy_wspd_10m"].tolist()
        pred = sub["wspd_era5"].tolist()
        bias = np.mean(np.subtract(pred, obs))
        mae = np.mean(np.abs(np.subtract(pred, obs)))
        rmse = math.sqrt(np.square(np.subtract(pred, obs)).mean())
        row5 = pd.DataFrame(
            [["WSPD-ERA5", sub.shape[0], bias, mae, rmse]],
            columns=["ID", "N", "Bias", "MAE", "RMSE"],
        )
    df = pd.concat([row1, row6, row2, row7, row3, row4, row5], axis=0)
    return df


###############################################################################################
# Open the df with wind, sigma0, buoy, and lake/ satellite attributes
###############################################################################################

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

# Create df with just entries with wind streaks
wind_df_ws = wind_df[wind_df["wind_streak"] == "Yes"]

# Create df with no wind streaks
wind_df_nows = wind_df[wind_df["wind_streak"] == "No"]

# Open the buoy and lake info shapefiles
pld = gpd.read_file(os.path.join(home, "Data/Buoy/pld_with_buoys.shp"))
# Fix the pld ref area to be km2
pld["ref_area"] = pld["ref_area"] / 100  # for some reason this is necessary
pld["lake_id"] = pld["lake_id"].astype(np.int64).astype(str)

# Min and max dates of each satelite included
wind_df.groupby("satellite").agg({"overpass_datetime": [np.min, np.max]})

###############################################################################################
# Supplementary Table 1. A table with buoy name, source, water body name,
# temporal sampling, if seen by swot or s-1
###############################################################################################

buoy_df = buoy_info[["id", "name", "source", "height", "type", "pld_id"]]
buoy_df["Longitude"] = buoy_info.geometry.apply(lambda p: p.x)
buoy_df["Latitude"] = buoy_info.geometry.apply(lambda p: p.y)
buoy_df = buoy_df.rename(
    columns={
        "id": "ID",
        "name": "Name",
        "source": "Source",
        "height": "Height",
        "type": "Observation Frequency",
        "pld_id": "PLD ID",
    }
)
buoy_df["Height"] = buoy_df["Height"].round(3)


# Add which satellite
def which_satellite(id):
    sub = wind_df[wind_df["buoy_id"] == id]
    return ", ".join(list(set(sub["satellite"])))


buoy_df["Satellite"] = list(map(which_satellite, buoy_df["ID"].tolist()))
buoy_df = buoy_df.sort_values("ID")

# Drop buchillon station
buoy_df = buoy_df[buoy_df["ID"] != "buchillonfieldstation"]

# How many observations of S1 and SWOT at each buoy
sat_count = (
    wind_df[["wdir", "satellite", "buoy_id"]]
    .groupby(["satellite", "buoy_id"])
    .count()
    .reset_index()
)
sat_count_wide = sat_count.pivot(
    index="buoy_id", columns="satellite", values="wdir"
).reset_index()
sat_count_wide["S1"].min()
sat_count_wide["S1"].max()
sat_count_wide["SWOT"].min()
sat_count_wide["SWOT"].max()

buoy_df = buoy_df.merge(sat_count_wide.rename(columns={"buoy_id": "ID"}))
buoy_df.groupby("Source").count()

# Save the table
buoy_df.to_csv(os.path.join(home, "Data/Figures/buoy_attributes_table.csv"))

###############################################################################################
# Figure 1. A map of the study area including the lakes and buoys
# Color the buoys according to seen by SWOT, S1, or both.
# A histogram inset into the map shows the distribution of lake area.
###############################################################################################

# Add the geometry back to buoy df and save and create the map in arcpro
buoy_df = gpd.GeoDataFrame(
    buoy_df,
    geometry=gpd.points_from_xy(buoy_df.Longitude, buoy_df.Latitude),
    crs="EPSG:4326",
)

buoy_df.to_file(
    os.path.join(home, "Data/Outputs/buoy_attributes_table.shp"),
)

# Make the histogram of pld area
fig = plt.figure(figsize=(5, 3))
sns.histplot(data=pld, x="ref_area", color="#002673", bins="auto", kde=True)
plt.xlabel("Lake Area (km$^{2}$)", size=12)
plt.ylabel("Count", size=14)
plt.xscale("log")
# plt.show()
plt.savefig(os.path.join(home, "Data/Figures/pld_area_hist.png"), dpi=1000)

# Make CDF of the pld area for all buoys grouped by sensor
# Merge in the pld area to buoy_df
buoy_df = buoy_df.merge(
    pld[["lake_id", "ref_area"]].rename(columns={"lake_id": "PLD ID"}), on="PLD ID"
)

fig = plt.figure(figsize=(6, 3))
b = sns.ecdfplot(
    data=buoy_df,
    x="ref_area",
    hue="Satellite",
    palette=["#6ccfed", "#f0f000"],
    linewidth=6,
    legend=False,
)
b.set_yticklabels(np.round(b.get_yticks(), 1), size=20)
b.set_xticklabels([int(i) for i in b.get_xticks()], size=20)
sns.set_context("paper", rc={"figure.figsize": (6, 3)})
plt.xlabel("Lake Area (km$^{2}$)", size=25)
plt.ylabel("CDF", size=25)
# plt.show()
plt.savefig(
    os.path.join(home, "Data/Figures/pld_area_cdf.png"), dpi=1000, bbox_inches="tight"
)

# Number of buoys seen by each satellite
buoy_df.groupby(["Satellite"]).size()

# Size of PLD lakes with a buoy
pld.shape[0]
pld["ref_area"].min()
pld["ref_area"].max()
pld["ref_area"].mean()
pld["ref_area"].median()

###############################################################################################
# Figure 2. Visuzliae wind streaks and wind direction over lake ontario.
# First panel is just the sigma0 with an arrow at the buoy showing the real wind direction.
# Second panel is wind direction arrows colored according to marginal error (degrees)
###############################################################################################

def save_examples(row):
    """
    Take the row from wind_df corresponding to the image and buoy.
    Save the sigma0 image, wind direction and me at 1km resolution as a geopandas df
    """
    image_id = row["image_id"].iloc[0]
    if row.satellite.iloc[0] == "SWOT":
        satellite = "SWOT_L2_HR_Raster/SWOT_L2_HR_Raster_2.0"
    if row.satellite.iloc[0] == "S1":
        satellite = "Sentinel1/Processed"

    folder_out = os.path.join(home, "Data/Outputs/", image_id)
    isExist = os.path.exists(folder_out)
    if not isExist:
        os.makedirs(folder_out)

    # Save the sigma0 raster
    fp = os.path.join(home, "Data", satellite, os.path.basename(image_id) + ".nc")
    src = rioxarray.open_rasterio(fp)
    sig0 = src.sig0

    if row.satellite.iloc[0] == "SWOT":
        # Covnert no data to na
        sig0 = sig0.where(sig0 != sig0.rio.nodata)
        # Filter sig0 observations. Only keep observations that corresond to 0 (good), 1(suspect) and 2 (degraded). 3 = bad
        sig0 = sig0.where(src.sig0_qual <= 2)
        # Get the epsg string and update the crs
        crs_wkt = sig0.crs.attrs["crs_wkt"]
        crs_wkt_split = crs_wkt.split(",")
        epsg = crs_wkt_split[len(crs_wkt_split) - 1].split('"')[1]
        sig0.rio.write_crs("epsg:" + epsg, inplace=True)

    boundary_reproj = pld.to_crs(sig0.rio.crs)
    sig0_clipped = sig0.rio.clip(boundary_reproj.geometry.values, boundary_reproj.crs)
    sig0_clipped = sig0_clipped.where(sig0_clipped != sig0_clipped.rio.nodata)
    sig0_out = folder_out + "/sig0_" + image_id + ".tif"
    # sig0_clipped.rio.to_raster(sig0_out)
    sig0.rio.to_raster(sig0_out)
    src.close()

    # Save the buoy with the wind direction
    buoy_wdir = row[["buoy_id", "wdir_buoy"]]
    buoy_wdir = gpd.GeoDataFrame(
        buoy_wdir.merge(buoy_info[["id", "geometry"]], left_on="buoy_id", right_on="id")
    )
    buoy_wdir[["buoy_id", "wdir_buoy", "geometry"]].to_file(
        os.path.join(folder_out, "buoy" + image_id + ".shp")
    )

    # Find corresponding wind direction and me rasters and combine into 1km shapefile
    if row.satellite.iloc[0] == "SWOT":
        wdir_fps = os.path.join(
            home,
            "Data/Wind_Direction/Rana/SWOT_L2_HR_Raster_2.0/WindDirection/ERA5_Corrected/corrected_wdir_10pixels_"
            + image_id
            + ".tif",
        )
        me_fps = os.path.join(
            home,
            "Data/Wind_Direction/Rana/SWOT_L2_HR_Raster_2.0/MarginalError/me_10pixels_"
            + image_id
            + ".tif",
        )
    if row.satellite.iloc[0] == "S1":
        wdir_fps = os.path.join(
            home,
            "Data/Wind_Direction/Rana/Sentinel1/WindDirection/ERA5_Corrected/corrected_wdir_10pixels_"
            + image_id
            + ".tif",
        )
        me_fps = os.path.join(
            home,
            "Data/Wind_Direction/Rana/Sentinel1/MarginalError/me_10pixels_"
            + image_id
            + ".tif",
        )

    # Convert to 1km resolution shapefile in one
    wdir = rioxarray.open_rasterio(wdir_fps)
    wdir = wdir.to_dataset(name="wdir")
    me = rioxarray.open_rasterio(me_fps)
    me = me.to_dataset(name="me")
    combo = xarray.merge([wdir, me])
    upscale_factor = 0.1
    new_width = int(combo.rio.width * upscale_factor)
    new_height = int(combo.rio.height * upscale_factor)
    combo = combo.rio.reproject(
        combo.rio.crs,
        shape=(new_height, new_width),
        resampling=Resampling.bilinear,
    )

    lons = combo.x.to_numpy().ravel()
    lats = combo.y.to_numpy().ravel()
    lons, lats = np.meshgrid(lons, lats)
    lons = lons.ravel()
    lats = lats.ravel()

    wdir = combo.wdir.to_numpy().ravel()
    me = combo.me.to_numpy().ravel()
    image_crs = combo.rio.crs
    df = pd.DataFrame({"long": lons, "lat": lats, "wdir": wdir, "me": me})
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.long, df.lat), crs=image_crs
    )
    gdf = gdf.dropna(subset=["wdir", "me"])
    gdf.to_file(os.path.join(folder_out, "wdir_" + image_id + ".shp"))

    return


# SAR image to visualize: S1A_IW_GRDH_1SDV_20230520T231641_20230520T231706_048624_05D926_1C5B
row = wind_df[
    (
        (
            wind_df["image_id"]
            == "S1A_IW_GRDH_1SDV_20230520T231641_20230520T231706_048624_05D926_1C5B"
        )
        & (wind_df["buoy_id"] == "45139")
    )
]
row.wdir_buoy
row.wdir_corrected
save_examples(row)

# Export ERA5 polygons cropped around the lake
pld_subset = gpd.GeoDataFrame(
    pld[pld["lake_id"] == buoy_info[buoy_info["id"] == "45139"]["pld_id"].iloc[0]]
)
pld_subset["geometry"] = pld_subset.buffer(0.1)
era5_fp = (
    "C:/Users/kmcquil/Documents/SWOT_WIND/Data/ERA5/Processed/wdir/20230520160000.tif"
)
era5 = rioxarray.open_rasterio(era5_fp)
era5_crop = era5.rio.clip_box(*pld_subset.total_bounds)
era5_crop = era5_crop.astype("float32")
era5_poly = vectorize(era5_crop)
era5_poly = era5_poly.rename(columns={era5_poly.columns.values[0]: "wdir"})
era5_poly.to_file(
    os.path.join(home, "Data/Outputs/", row["image_id"].iloc[0], "era5_grid.shp")
)


###############################################################################################
# Figure 3. Wind streak frequency and conditions
# First is a bar plot of % wind streaks for S1 and SWOT.
# Second subplot is wind streak vs no wind streak for lake fetch.
# Third subplot is wind speed for wind streak vs no wind streak. Put the significance on the figure.
###############################################################################################

# How many presence / absence records total
wind_df.shape[0]

# How many records for each buoy
counts = (
    wind_df[["buoy_id", "satellite", "wind_streak"]]
    .groupby(["buoy_id", "satellite"])
    .count()
    .reset_index()
)

counts[counts["satellite"] == "S1"]["wind_streak"].min()
counts[counts["satellite"] == "S1"]["wind_streak"].max()
counts[counts["satellite"] == "S1"]["wind_streak"].median()

counts[counts["satellite"] == "SWOT"]["wind_streak"].min()
counts[counts["satellite"] == "SWOT"]["wind_streak"].max()
counts[counts["satellite"] == "SWOT"]["wind_streak"].median()

# What fraction of rows had a wind streak = 13.6%
wind_df_ws.shape[0] / wind_df.shape[0]
wind_df_ws[wind_df_ws["satellite"] == "S1"].shape[0] / wind_df[
    wind_df["satellite"] == "S1"
].shape[0]
wind_df_ws[wind_df_ws["satellite"] == "SWOT"].shape[0] / wind_df[
    wind_df["satellite"] == "SWOT"
].shape[0]

# How many image-buoy pairs for each satellite
wind_df[wind_df["satellite"] == "S1"].shape[0]
wind_df[wind_df["satellite"] == "SWOT"].shape[0]

# Set up columns for the figures
ws_figs = wind_df.copy()

# What the mean and median of presence/absence wind
# streaks for wind speed and fetch
ws_figs.groupby("wind_streak")["buoy_wspd_10m"].median()
ws_figs.groupby("wind_streak")["buoy_wspd_10m"].mean()

ws_figs.groupby("wind_streak")["fetch_lake"].median()
ws_figs.groupby("wind_streak")["fetch_lake"].mean()
# ws_figs.groupby('wind_streak')['fetch_buoy'].median()

# Median wind speed and fetch for SWOT and S1
ws_figs[ws_figs["satellite"] == "S1"]["buoy_wspd_10m"].median()
ws_figs[ws_figs["satellite"] == "SWOT"]["buoy_wspd_10m"].median()

ws_figs[ws_figs["satellite"] == "S1"]["fetch_lake"].median()
ws_figs[ws_figs["satellite"] == "SWOT"]["fetch_lake"].median()

# Test if wind streak and no wind streak are sig diff for wind speed and lake fetch
wspd_ws = ws_figs[
    ((ws_figs["satellite"] == "S1") & (ws_figs["wind_streak"] == "Yes"))
].dropna(subset=["buoy_wspd_10m"])["buoy_wspd_10m"]
wspd_nows = ws_figs[
    ((ws_figs["satellite"] == "S1") & (ws_figs["wind_streak"] == "No"))
].dropna(subset=["buoy_wspd_10m"])["buoy_wspd_10m"]

fetch_ws = ws_figs[
    ((ws_figs["satellite"] == "S1") & (ws_figs["wind_streak"] == "Yes"))
].dropna(subset=["fetch_lake"])["fetch_lake"]
fetch_nows = ws_figs[
    ((ws_figs["satellite"] == "S1") & (ws_figs["wind_streak"] == "No"))
].dropna(subset=["fetch_lake"])["fetch_lake"]

U1_wspd, p_wspd = mannwhitneyu(wspd_ws, wspd_nows, method="asymptotic")
U1_fetch, p_fetch = mannwhitneyu(fetch_ws, fetch_nows, method="asymptotic")

# Fraction of buoy-image pairs that have a visible wind streak for each
ws_summary = (
    ws_figs.groupby(["satellite", "wind_streak"]).size().to_frame("size").reset_index()
)

# % of image-buoy pairs that have wind streak visible
fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(14, 4))
sns.barplot(
    ax=ax1,
    x="satellite",
    y="size",
    hue="wind_streak",
    palette=["#31688e", "#fde725"],
    data=ws_summary,
)
ax1.set_xlabel("Satellite", size=12)
ax1.set_ylabel("Buoy-image count", size=12)
ax1.legend(title="Wind Streaks")

# Box plot of the wind streak presence/absence grouped by wind speed
sns.boxplot(
    ax=ax2,
    x="satellite",
    y="buoy_wspd_10m",
    hue="wind_streak",
    palette=["#31688e", "#fde725"],
    data=ws_figs,
    legend=False,
)
ax2.set_xlabel("Satellite", size=12)
ax2.set_ylabel("Wind speed (m/s)", size=12)

# Box plot of the wind streak presence/absence grouped by fetch at each buoy
sns.boxplot(
    ax=ax3,
    x="satellite",
    y="fetch_lake",
    hue="wind_streak",
    palette=["#31688e", "#fde725"],
    data=ws_figs,
    legend=False,
)
ax3.set_xlabel("Satellite", size=12)
ax3.set_ylabel("Fetch (km)", size=12)
plt.savefig(
    os.path.join(home, "Data/Figures/wind_streak_summary.png"),
    dpi=1000,
    bbox_inches="tight",
)

###############################################################################################
# Table 1. MAE of wind direction for All, All with ME thresholds,
#  with ME thresholds Wind streaks, wind streaks
###############################################################################################

perf_overall = []
satellites = ["S1", "SWOT"]
me_limits = [360, 40, 30, 20]
for satellite in satellites:
    for limit in me_limits:
        subset = wind_df[
            ((wind_df["satellite"] == satellite) & (wind_df["me"] < limit))
        ]
        subset_ws = wind_df_ws[
            ((wind_df_ws["satellite"] == satellite) & (wind_df_ws["me"] < limit))
        ]
        subset_nows = wind_df_nows[
            ((wind_df_nows["satellite"] == satellite) & (wind_df_nows["me"] < limit))
        ]

        stats_180 = calc_performance_stats_180(subset).reset_index(drop=True)
        stats_180_ws = calc_performance_stats_180(subset_ws).reset_index(drop=True)
        stats_180_nows = calc_performance_stats_180(subset_nows).reset_index(drop=True)

        label = pd.DataFrame(
            [[satellite, "1 km", "All", limit]],
            columns=["Satellite", "Resolution", "Wind_Streak", "ME Limit"],
        )
        labelrep = label.loc[label.index.repeat(stats_180.shape[0])].reset_index(
            drop=True
        )
        perf = pd.concat([labelrep, stats_180], axis=1)

        label = pd.DataFrame(
            [[satellite, "1 km", "Wind Streaks", limit]],
            columns=["Satellite", "Resolution", "Wind_Streak", "ME Limit"],
        )
        labelrep = label.loc[label.index.repeat(stats_180_ws.shape[0])].reset_index(
            drop=True
        )
        perf_ws = pd.concat([labelrep, stats_180_ws], axis=1)

        label = pd.DataFrame(
            [[satellite, "1 km", "No Wind Streaks", limit]],
            columns=["Satellite", "Resolution", "Wind_Streak", "ME Limit"],
        )
        labelrep = label.loc[label.index.repeat(stats_180_nows.shape[0])].reset_index(
            drop=True
        )
        perf_nows = pd.concat([labelrep, stats_180_nows], axis=1)

        perf_overall.append(pd.concat([perf, perf_ws, perf_nows], axis=0))

perf_overall = pd.concat(perf_overall)

# Just look at the performance with 180 degree ambiguity
overall_table = perf_overall[perf_overall["ID"].isin(["WDIR-LG-180", "WDIR-ERA5-180"])]
# Get rid of the 'No Wind Streaks'
overall_table = overall_table[overall_table["Wind_Streak"] != "No Wind Streaks"]
# Drop 'resolution', 'bias', and 'rmse'.
# We are only looking at 1 km resolution and did not calculate bias or rmse for 180 degree ambiguous wdir
overall_table = overall_table.drop(["Resolution", "Bias", "RMSE"], axis=1)
# Pivot so that era5 and lg-mod each have their own column in the table
overall_table = (
    overall_table.pivot(
        index=["Satellite", "Wind_Streak", "ME Limit", "N"], columns="ID", values="MAE"
    )
    .round(2)
    .reset_index()
)
# Sort according to ME limit
overall_table = overall_table.sort_values(
    ["Satellite", "Wind_Streak", "ME Limit"], ascending=[True, True, False]
)
overall_table = overall_table[overall_table["N"] > 0]
# Save the wind direction performance table
overall_table.to_csv(os.path.join(home, "Data/Outputs/wdir_perf.csv"))


########################################################################################3
# What is the performance without 180 degree ambiguity
overall_table_na = perf_overall[perf_overall["ID"].isin(["WDIR-LG", "WDIR-ERA5"])]
# Get rid of the 'No Wind Streaks'
overall_table_na = overall_table_na[
    overall_table_na["Wind_Streak"] != "No Wind Streaks"
]
# Drop 'resolution', 'bias', and 'rmse'.
# We are only looking at 1 km resolution and did not calculate bias or rmse for 180 degree ambiguous wdir
overall_table_na = overall_table_na.drop(["Resolution", "Bias", "RMSE"], axis=1)
# Pivot so that era5 and lg-mod each have their own column in the table
overall_table_na = (
    overall_table_na.pivot(
        index=["Satellite", "Wind_Streak", "ME Limit", "N"], columns="ID", values="MAE"
    )
    .round(2)
    .reset_index()
)
# Sort according to ME limit
overall_table_na = overall_table_na.sort_values(
    ["Satellite", "Wind_Streak", "ME Limit"], ascending=[True, True, False]
)
overall_table_na = overall_table_na[overall_table_na["N"] > 0]
########################################################################################

# Make this into a bar plot
overall_table_wdir = overall_table.copy()
# Create a column with nicer names
overall_table_wdir["Subset"] = [
    i + ": " + j
    for i, j in zip(overall_table_wdir["Satellite"], overall_table_wdir["Wind_Streak"])
]
overall_table_wdir["Subset"] = overall_table_wdir["Subset"].map(
    {"S1: All": "S1", "S1: Wind Streaks": "S1: Wind streaks", "SWOT: All": "SWOT"}
)
# three separate panels. Each panel is colored by s1/swot vs ERA5
# so make long
overall_table_wdir = pd.melt(
    overall_table_wdir,
    id_vars=["Satellite", "Wind_Streak", "ME Limit", "N", "Subset"],
    value_vars=["WDIR-ERA5-180", "WDIR-LG-180"],
)
overall_table_wdir["ID"] = np.where(
    overall_table_wdir["ID"] == "WDIR-ERA5-180", "ERA5", overall_table_wdir["Satellite"]
)
# Change the ME thresholds to strings and order them
overall_table_wdir["ME Limit"] = overall_table_wdir["ME Limit"].astype(str)
overall_table_wdir["ME Limit"] = overall_table_wdir["ME Limit"].map(
    {"360": "All", "20": "20", "30": "30", "40": "40"}
)

fig, ((ax3), (ax1), (ax2)) = plt.subplots(1, 3, figsize=(12, 4))
palette = {
    "S1": "#5ec962",
    "ERA5": "#3b528b",
    "S1: Wind streaks": "#fde725",
    "SWOT": "#440154",
}
g1 = sns.barplot(
    ax=ax1,
    data=overall_table_wdir[overall_table_wdir["Subset"] == "S1"],
    x="ME Limit",
    y="value",
    hue="ID",
    palette=palette,
    hue_order=["S1", "ERA5"],
)
ax1.set_xlabel("ME$^{TH}$", size=12)
ax1.set_ylabel("", size=12)
ax1.set(ylim=(15, 52))
ax1.set_title("", size=12)
ax1.legend(title="")

ws_sub = overall_table_wdir[overall_table_wdir["Subset"] == "S1: Wind streaks"]
ws_sub["ID"] = ws_sub["ID"].map({"ERA5": "ERA5", "S1": "S1: Wind streaks"})
g2 = sns.barplot(
    ax=ax2,
    data=ws_sub,
    x="ME Limit",
    y="value",
    hue="ID",
    palette=palette,
    hue_order=["S1: Wind streaks", "ERA5"],
)
ax2.set_xlabel("ME$^{TH}$", size=12)
ax2.set_ylabel("", size=12)
ax2.set(ylim=(15, 52))
ax2.set_title("", size=12)
ax2.legend(title="")

g3 = sns.barplot(
    ax=ax3,
    data=overall_table_wdir[overall_table_wdir["Subset"] == "SWOT"],
    x="ME Limit",
    y="value",
    hue="ID",
    palette=palette,
    hue_order=["SWOT", "ERA5"],
)
ax3.set_xlabel("ME$^{TH}$", size=12)
ax3.set_ylabel("MAE (m/s)", size=12)
ax3.set(ylim=(15, 52))
ax3.set_title("", size=12)
ax3.legend(title="")
sns.move_legend(ax3, "upper center")
plt.savefig(
    os.path.join(home, "Data/Figures/wind_direction_performance.png"),
    dpi=1000,
    bbox_inches="tight",
)


###############################################################################################
# Table 2. Perf stats of wind speed for All, All with ME thresholds,
#  with ME thresholds Wind streaks, wind streaks
###############################################################################################
# Select the wind speed performance
overall_table = perf_overall[perf_overall["ID"].isin(["WSPD-CMOD5-SAR", "WSPD-ERA5"])]
# Get rid of the SWOT NAs for wind speed performance by selecting S1
overall_table = overall_table[overall_table["Satellite"] == "S1"]
# Get rid of 'No wind streaks'
overall_table = overall_table[
    overall_table["Wind_Streak"].isin(["All", "Wind Streaks"])
]

# Drop the resolution column since we only looked at 1km resolution
overall_table = overall_table.drop(["Resolution"], axis=1)
overall_table_wspd = overall_table.copy()
# Pivot so that cmod5.n and era5 both have a column in the table
overall_table = (
    overall_table.pivot(
        index=["Satellite", "Wind_Streak", "ME Limit", "N"],
        columns="ID",
        values=["Bias", "MAE", "RMSE"],
    )
    .round(2)
    .reset_index()
)
# Sort by the ME
overall_table = overall_table.sort_values(
    ["Satellite", "Wind_Streak", "ME Limit"], ascending=[True, True, False]
).reset_index(drop=True)
# Save the wind speed performance table
overall_table.to_csv(os.path.join(home, "Data/Outputs/wspd_perf.csv"))

# Use the copy in long form to make figures
overall_table_wspd["Subset"] = np.where(
    overall_table_wspd["Wind_Streak"] == "All", "S1", "S1: Wind streaks"
)
overall_table_wspd["ID"] = np.where(
    overall_table_wspd["ID"] == "WSPD-ERA5", "ERA5", overall_table_wspd["Subset"]
)
overall_table_wspd["ME Limit"] = overall_table_wspd["ME Limit"].astype(str)
overall_table_wspd["ME Limit"] = overall_table_wspd["ME Limit"].map(
    {"360": "All", "40": "40", "30": "30", "20": "20"}
)

# will eventually be 3, 2
fig, ((ax1, ax3), (ax2, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(6, 8))

palette = {"S1": "#5ec962", "ERA5": "#3b528b", "S1: Wind streaks": "#fde725"}

sns.barplot(
    ax=ax1,
    data=overall_table_wspd[overall_table_wspd["Subset"] == "S1"],
    x="ME Limit",
    y="Bias",
    hue="ID",
    palette=palette,
)
# ax1.set_xlabel( "ME$^{TH}$" , size = 12 )
ax1.set_xlabel("", size=12)
ax1.set_ylabel("Bias (m/s)", size=12)
ax1.set(ylim=(-0.9, 1.3))
ax1.set_title("", size=12)
ax1.legend(title="", loc="lower left")

sns.barplot(
    ax=ax3,
    data=overall_table_wspd[overall_table_wspd["Subset"] == "S1: Wind streaks"],
    x="ME Limit",
    y="Bias",
    hue="ID",
    palette=palette,
)
# ax3.set_xlabel( "ME$^{TH}$" , size = 12 )
ax3.set_xlabel("", size=12)
ax3.set_ylabel("", size=12)
ax3.set(ylim=(-0.9, 1.3))
ax3.set_title("", size=12)
ax3.legend(title="", loc="lower left")

sns.barplot(
    ax=ax2,
    data=overall_table_wspd[overall_table_wspd["Subset"] == "S1"],
    x="ME Limit",
    y="MAE",
    hue="ID",
    palette=palette,
    legend=False,
)
# ax2.set_xlabel( "ME$^{TH}$" , size = 12 )
ax2.set_xlabel("", size=12)
ax2.set_ylabel("MAE (m/s)", size=12)
ax2.set(ylim=(0, 2.5))
ax2.set_title("", size=12)
# ax2.legend(title='')

sns.barplot(
    ax=ax4,
    data=overall_table_wspd[overall_table_wspd["Subset"] == "S1: Wind streaks"],
    x="ME Limit",
    y="MAE",
    hue="ID",
    palette=palette,
    legend=False,
)
# ax4.set_xlabel( "ME$^{TH}$" , size = 12 )
ax4.set_xlabel("", size=12)
ax4.set_ylabel("", size=12)
ax4.set(ylim=(0, 2.5))
ax4.set_title("", size=12)
# ax4.legend(title='')

sns.barplot(
    ax=ax5,
    data=overall_table_wspd[overall_table_wspd["Subset"] == "S1"],
    x="ME Limit",
    y="RMSE",
    hue="ID",
    palette=palette,
    legend=False,
)
ax5.set_xlabel("ME$^{TH}$", size=12)
ax5.set_ylabel("RMSE (m/s)", size=12)
ax5.set(ylim=(0, 3))
ax5.set_title("", size=12)
# ax5.legend(title='')

sns.barplot(
    ax=ax6,
    data=overall_table_wspd[overall_table_wspd["Subset"] == "S1: Wind streaks"],
    x="ME Limit",
    y="RMSE",
    hue="ID",
    palette=palette,
    legend=False,
)
ax6.set_xlabel("ME$^{TH}$", size=12)
ax6.set_ylabel("", size=12)
ax6.set(ylim=(0, 3))
ax6.set_title("", size=12)
# ax6.legend(title='')
# plt.show()
plt.savefig(
    os.path.join(home, "Data/Figures/wind_speed_performance.png"),
    dpi=1000,
    bbox_inches="tight",
)

###############################################################################################
# Figure 4. Box plots of performance across buoys for wind direction
###############################################################################################
# Create a list to store the buoy performance
perf_buoys = []
# Create a list of the ME limits
me_limits = [360, 40, 30, 20]
# Create list of unique buoys
buoys = set(wind_df["buoy_id"])
# List the satellites
satellites = ["S1", "SWOT"]
combos = list(itertools.product(*[buoys, satellites, me_limits]))
for combo in combos:
    subset = wind_df[
        (
            (wind_df["satellite"] == combo[1])
            & (wind_df["buoy_id"] == combo[0])
            & (wind_df["me"] <= combo[2])
        )
    ]
    subset_ws = wind_df_ws[
        (
            (wind_df_ws["satellite"] == combo[1])
            & (wind_df_ws["buoy_id"] == combo[0])
            & (wind_df_ws["me"] <= combo[2])
        )
    ]
    subset_nows = wind_df_nows[
        (
            (wind_df_nows["satellite"] == combo[1])
            & (wind_df_nows["buoy_id"] == combo[0])
            & (wind_df_nows["me"] <= combo[2])
        )
    ]

    stats_180 = calc_performance_stats_180(subset).reset_index(drop=True)
    stats_180_ws = calc_performance_stats_180(subset_ws).reset_index(drop=True)
    stats_180_nows = calc_performance_stats_180(subset_nows).reset_index(drop=True)

    label = pd.DataFrame(
        [[combo[1], combo[0], "1 km", "All", combo[2]]],
        columns=["Satellite", "Buoy", "Resolution", "Wind_Streak", "ME_Limit"],
    )
    labelrep = label.loc[label.index.repeat(stats_180.shape[0])].reset_index(drop=True)
    perf = pd.concat([labelrep, stats_180], axis=1)

    label = pd.DataFrame(
        [[combo[1], combo[0], "1 km", "Wind Streaks", combo[2]]],
        columns=["Satellite", "Buoy", "Resolution", "Wind_Streak", "ME_Limit"],
    )
    labelrep = label.loc[label.index.repeat(stats_180_ws.shape[0])].reset_index(
        drop=True
    )
    perf_ws = pd.concat([labelrep, stats_180_ws], axis=1)

    label = pd.DataFrame(
        [[combo[1], combo[0], "1 km", "No Wind Streaks", combo[2]]],
        columns=["Satellite", "Buoy", "Resolution", "Wind_Streak", "ME_Limit"],
    )
    labelrep = label.loc[label.index.repeat(stats_180_nows.shape[0])].reset_index(
        drop=True
    )
    perf_nows = pd.concat([labelrep, stats_180_nows], axis=1)

    perf_buoys.append(pd.concat([perf, perf_ws, perf_nows], axis=0))

# Create df of the wind direction bouy performance
perf_buoys = pd.concat(perf_buoys)
# Only select the wind direction IDs
subset = perf_buoys[perf_buoys["ID"].isin(["WDIR-LG-180", "WDIR-ERA5-180"])]
# Get rid of NAs in the MAE column
subset = subset.dropna(subset=["MAE"])
# Not interested in looking at the subset with only No Wind Streaks
subset = subset[subset["Wind_Streak"] != "No Wind Streaks"]
# Make the ID pretty for plotting
subset["ID"] = subset["ID"].map({"WDIR-LG-180": "LG-Mod", "WDIR-ERA5-180": "ERA5"})
# For the final analysis, just look at the performance using an ME limit of 30
buoy_perf_30_wdir = subset[subset["ME_Limit"] == 30]
# Make column for IDs
buoy_perf_30_wdir["SID"] = [
    i + ": " + j
    for i, j in zip(
        buoy_perf_30_wdir["Satellite"].tolist(),
        buoy_perf_30_wdir["Wind_Streak"].tolist(),
    )
]

# Make a boxplot of the MAE across buoys
fig, ax = plt.subplots(figsize=(6, 4))
g = sns.boxplot(
    ax=ax,
    data=buoy_perf_30_wdir,
    x="SID",
    y="MAE",
    hue="ID",
    palette=["#3b528b", "#5ec962"],
    order=["S1: Wind Streaks", "S1: All", "SWOT: All"],
)
ax.set_xlabel("", size=12)
ax.set_ylabel("MAE (degrees)", size=12)
ax.set(ylim=(0, 80))
ax.legend(title="")
# plt.show()
plt.savefig(os.path.join(home, "Data/Figures/wdir_perf_boxplots.png"), dpi=1000)


###############################################################################################
# Figure 5. Box plots of performance across buoys for wind speed
###############################################################################################

# Create df of the wind speed
subset = perf_buoys[perf_buoys["ID"].isin(["WSPD-CMOD5-SAR", "WSPD-ERA5"])]
# Drop NAs of MAE
subset = subset.dropna(subset=["MAE"])
# Not interested in the subset without wind streaks
subset = subset[subset["Wind_Streak"] != "No Wind Streaks"]
# Make the names pretty for plotting
subset["ID"] = subset["ID"].map(
    {"WSPD-CMOD5-SAR": "CMOD5.N+LG-Mod", "WSPD-ERA5": "ERA5"}
)
# Only use performance with a ME limit of 30
buoy_perf_30_wspd = subset[subset["ME_Limit"] == 30]
buoy_perf_30_wspd["SID"] = [
    i + ": " + j
    for i, j in zip(
        buoy_perf_30_wspd["Satellite"].tolist(),
        buoy_perf_30_wspd["Wind_Streak"].tolist(),
    )
]

# Make a boxplot of the bias across buoys
fig, ((ax1), (ax2), (ax3)) = plt.subplots(1, 3, figsize=(12, 4))
g = sns.boxplot(
    ax=ax1,
    data=buoy_perf_30_wspd,
    x="Wind_Streak",
    y="Bias",
    hue="ID",
    palette=["#3b528b", "#5ec962"],
    width=0.7,
    showfliers=False,
)
sns.despine()
ax1.set_xlabel("", size=12)
ax1.set_ylabel("Bias (m/s)", size=12)
ax1.set_title("", size=12)
ax1.legend(title="")

# Boxplot of the MAE across buoys
g1 = sns.boxplot(
    ax=ax2,
    data=buoy_perf_30_wspd,
    x="Wind_Streak",
    y="MAE",
    hue="ID",
    palette=["#3b528b", "#5ec962"],
    width=0.7,
    legend=False,
    showfliers=False,
)
ax2.set_xlabel("", size=12)
ax2.set_ylabel("MAE (m/s)", size=12)
ax2.set_title("", size=12)

# Boxplot of the RMSE across buoys
g2 = sns.boxplot(
    ax=ax3,
    data=buoy_perf_30_wspd,
    x="Wind_Streak",
    y="RMSE",
    hue="ID",
    palette=["#3b528b", "#5ec962"],
    width=0.7,
    legend=False,
    showfliers=False,
)
ax3.set_xlabel("", size=12)
ax3.set_ylabel("RMSE (m/s)", size=12)
ax3.set_title("", size=12)
plt.savefig(os.path.join(home, "Data/Figures/wspd_perf_boxplots.png"), dpi=1000)


###############################################################################################
# Supplementary Figure 1
# Check for sig relationships between buoy/lake/attributes with error
###############################################################################################


def calc_mae_180(obs, pred):
    pred_180 = (pred + 180) % 360
    d1 = (pred - obs) % 360
    d2 = 360 - d1
    d1_180 = (pred_180 - obs) % 360
    d2_180 = 360 - d1_180
    return min([d1, d2, d1_180, d2_180])


wind_df["diff_180"] = wind_df.apply(
    lambda row: calc_mae_180(row["wdir_buoy"], row["wdir"]), axis=1
)

# Calc the time difference between buoy and overpass
wind_df["abs_time_diff_secs"] = np.abs(wind_df["time_diff"]).dt.total_seconds()

wdir_mae = wind_df[wind_df["me"] <= 30]

# Merge in the ref area
wdir_mae = wdir_mae.merge(
    buoy_info.rename(columns={"id": "buoy_id"})[["buoy_id", "type", "pld_id"]],
    on="buoy_id",
    how="left",
)
wdir_mae = wdir_mae.merge(
    pld[["lake_id", "ref_area"]], left_on="lake_id", right_on="lake_id", how="left"
)

wdir_mae[wdir_mae["satellite"] == "S1"].shape
wdir_mae[wdir_mae["satellite"] == "SWOT"].shape

# Loop through combos of satellite, wind streaks,
vars = [
    "buoy_wspd_10m",
    "wdir_buoy",
    "fetch_lake",
    "ref_area",
    "distance",
    "sig0",
    "inc_angle",
    "abs_time_diff_secs",
]
metrics = ["diff_180"]
satellites = ["SWOT", "S1"]
rels_df = []
for sat in satellites:
    for var in vars:
        for metric in metrics:
            s = wdir_mae[wdir_mae["satellite"] == sat]
            s = s.dropna(subset=[var])
            if s.shape[0] < 3:
                continue
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
                s[metric].tolist(), s[var].tolist()
            )
            rels_df.append(
                pd.DataFrame(
                    [[sat, metric, var, slope, r_value, p_value, s.shape[0]]],
                    columns=[
                        "Satellite",
                        "Metric",
                        "Variable",
                        "Slope",
                        "R",
                        "Pvalue",
                        "N",
                    ],
                )
            )

# Combine results into one df
wdir_point_rel_df = pd.concat(rels_df)
wdir_point_rel_df["Significance"] = pd.Series(dtype="str")
wdir_point_rel_df.loc[wdir_point_rel_df["Pvalue"] <= 0.05, "Significance"] = "p<=0.05"
wdir_point_rel_df.loc[wdir_point_rel_df["Pvalue"] > 0.05, "Significance"] = "p>0.05"
wdir_point_rel_df["Variable"] = wdir_point_rel_df["Variable"].map(
    {
        "buoy_wspd_10m": "Wind speed (m/s)",
        "wdir_buoy": "Wind direction \n(degrees)",
        "fetch_lake": "Lake fetch (km)",
        "ref_area": "Lake area (km2)",
        "distance": "Buoy distance \nto shore (km)",
        "sig0": "Sigma0",
        "inc_angle": "Incidence Angle \n(degrees)",
        "abs_time_diff_secs": "Time (seconds)",
    }
)

# Plot the correlation coefficient and significance of each relationship grouped by satellite
fig, ((ax2)) = plt.subplots(
    1,
    1,
    figsize=(15.5, 4),
)
markers = {"p<=0.05": "s", "p>0.05": "X"}
palette = {
    "S1": "#5ec962",
    "ERA5": "#3b528b",
    "S1: Wind streaks": "#fde725",
    "SWOT": "#440154",
}
g1 = sns.scatterplot(
    ax=ax2,
    data=wdir_point_rel_df,
    x="R",
    y="Variable",
    hue="Satellite",
    palette=palette,
    style="Significance",
    markers=markers,
    s=100,
)
ax2.axhline(
    y="Wind speed (m/s)", linewidth=1, alpha=0.5, color="grey", ls="-", zorder=1
)
ax2.axhline(
    y="Wind direction \n(degrees)",
    linewidth=1,
    alpha=0.5,
    color="grey",
    ls="-",
    zorder=1,
)
ax2.axhline(y="Lake fetch (km)", linewidth=1, alpha=0.5, color="grey", ls="-", zorder=1)
ax2.axhline(y="Lake area (km2)", linewidth=1, alpha=0.5, color="grey", ls="-", zorder=1)
ax2.axhline(
    y="Buoy distance \nto shore (km)",
    linewidth=1,
    alpha=0.5,
    color="grey",
    ls="-",
    zorder=1,
)
ax2.axhline(y="Sigma0", linewidth=1, alpha=0.5, color="grey", ls="-", zorder=1)
ax2.axhline(
    y="Incidence Angle \n(degrees)",
    linewidth=1,
    alpha=0.5,
    color="grey",
    ls="-",
    zorder=1,
)
ax2.axhline(y="Time (seconds)", linewidth=1, alpha=0.5, color="grey", ls="-", zorder=1)
ax2.set_ylabel("")
ax2.set_xlabel("Correlation Coefficient (R)", size=14)
g1.set_yticklabels(
    [
        "Wind speed (m/s)",
        "Wind direction \n(degrees)",
        "Lake fetch (km)",
        "Lake area (km2)",
        "Buoy distance \nto shore (km)",
        "Sigma0",
        "Incidence Angle \n(degrees)",
        "Time (seconds)",
    ],
    size=12,
)
g1.set_xticklabels([round(i, 1) for i in g1.get_xticks()], size=12)
plt.legend(title="", fontsize="12")
# sns.move_legend(ax2, "upper left", bbox_to_anchor=(1, 1), frameon=False)
# plt.show()
plt.savefig(
    os.path.join(home, "Data/Figures/wdir_point_var_reg.png"),
    dpi=1000,
    bbox_inches="tight",
)

###############################################################################################
# Same thing but for wind speed
###############################################################################################

# Wind speed error at each point when ME < 30
wind_df["wspd_error"] = wind_df["wspd_sat_cmod5n"] - wind_df["buoy_wspd_10m"]
# Drop the buchillon field station
wspd_error = wind_df[
    ((wind_df["me"] <= 30) & (wind_df["buoy_id"] != "buchillonfieldstation"))
]

# Merge in the ref area
wspd_error = wspd_error.merge(
    buoy_info.rename(columns={"id": "buoy_id"})[["buoy_id", "type", "pld_id"]],
    on="buoy_id",
    how="left",
)
wspd_error = wspd_error.merge(
    pld[["lake_id", "ref_area"]], left_on="lake_id", right_on="lake_id", how="left"
)

# Loop through combos of satellite, wind streaks to calculate slope, R, and pvalue
vars = [
    "buoy_wspd_10m",
    "wdir_buoy",
    "fetch_lake",
    "ref_area",
    "distance",
    "sig0",
    "inc_angle",
    "abs_time_diff_secs",
]
metrics = ["wspd_error"]
satellites = ["S1"]
rels_df = []
for sat in satellites:
    for var in vars:
        for metric in metrics:
            s = wspd_error[wspd_error["satellite"] == sat]
            s = s.dropna(subset=[var, metric])
            if s.shape[0] < 3:
                continue
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
                s[metric].tolist(), s[var].tolist()
            )
            rels_df.append(
                pd.DataFrame(
                    [[sat, metric, var, slope, r_value, p_value, s.shape[0]]],
                    columns=[
                        "Satellite",
                        "Metric",
                        "Variable",
                        "Slope",
                        "R",
                        "Pvalue",
                        "N",
                    ],
                )
            )
# Combine results to one df
wspd_point_rels_df = pd.concat(rels_df)
# Make the significance pretty for plotting
wspd_point_rels_df["Significance"] = pd.Series(dtype="str")
wspd_point_rels_df.loc[wspd_point_rels_df["Pvalue"] <= 0.05, "Significance"] = "p<=0.05"
wspd_point_rels_df.loc[wspd_point_rels_df["Pvalue"] > 0.05, "Significance"] = "p>0.05"
# Make the vars pretty for plotting
wspd_point_rels_df["Variable"] = wspd_point_rels_df["Variable"].map(
    {
        "buoy_wspd_10m": "Wind speed (m/s)",
        "wdir_buoy": "Wind direction \n(degrees)",
        "fetch_lake": "Lake fetch (km)",
        "ref_area": "Lake area (km2)",
        "distance": "Buoy distance \nto shore (km)",
        "sig0": "Sigma0",
        "inc_angle": "Incidence Angle \n(degrees)",
        "abs_time_diff_secs": "Time (seconds)",
    }
)

fig, ((ax2)) = plt.subplots(
    1,
    1,
    figsize=(15.5, 4),
)
markers = {"p<=0.05": "s", "p>0.05": "X"}
g1 = sns.scatterplot(
    ax=ax2,
    data=wspd_point_rels_df,
    x="R",
    y="Variable",
    hue="Satellite",
    palette=palette,
    style="Significance",
    markers=markers,
    s=100,
)
ax2.axhline(
    y="Wind speed (m/s)", linewidth=1, alpha=0.5, color="grey", ls="-", zorder=1
)
ax2.axhline(
    y="Wind direction \n(degrees)",
    linewidth=1,
    alpha=0.5,
    color="grey",
    ls="-",
    zorder=1,
)
ax2.axhline(y="Lake fetch (km)", linewidth=1, alpha=0.5, color="grey", ls="-", zorder=1)
ax2.axhline(y="Lake area (km2)", linewidth=1, alpha=0.5, color="grey", ls="-", zorder=1)
ax2.axhline(
    y="Buoy distance \nto shore (km)",
    linewidth=1,
    alpha=0.5,
    color="grey",
    ls="-",
    zorder=1,
)
ax2.axhline(y="Sigma0", linewidth=1, alpha=0.5, color="grey", ls="-", zorder=1)
ax2.axhline(
    y="Incidence Angle \n(degrees)",
    linewidth=1,
    alpha=0.5,
    color="grey",
    ls="-",
    zorder=1,
)
ax2.axhline(y="Time (seconds)", linewidth=1, alpha=0.5, color="grey", ls="-", zorder=1)

ax2.set_ylabel("")
ax2.set_xlabel("Correlation Coefficient (R)", size=14)
g1.set_yticklabels(
    [
        "Wind speed (m/s)",
        "Wind direction \n(degrees)",
        "Lake fetch (km)",
        "Lake area (km2)",
        "Buoy distance \nto shore (km)",
        "Sigma0",
        "Incidence Angle \n(degrees)",
        "Time (seconds)",
    ],
    size=12,
)
g1.set_xticklabels([round(i, 1) for i in g1.get_xticks()], size=12)
plt.legend(title="", fontsize="12")
# sns.move_legend(ax2, "upper left", bbox_to_anchor=(1, 1), frameon=False)
# plt.show()
plt.savefig(
    os.path.join(home, "Data/Figures/wspd_point_var_reg.png"),
    dpi=1000,
    bbox_inches="tight",
)

wspd_error.dropna(subset="wspd_error").shape[0]

###############################################################################################
# Some of the wind speed is crazy over estimated. Why/where is that happening?
###############################################################################################
# Wind speed is calcualted from inc angle, sigma0, wind direction wrt look angle
# Make a subset to explore
wspd_subset = wind_df[
    (
        (wind_df["me"] < 30)
        & (~np.isnan(wind_df["wspd_sat_cmod5n"]))
        & (wind_df["buoy_id"] != "buchillonfieldstation")
    )
]

sns.scatterplot(data=wspd_subset, x="wspd_error", y="sig0")
plt.show()

sns.scatterplot(data=wspd_subset, x="wspd_error", y="inc_angle")
plt.show()

sns.scatterplot(data=wspd_subset, x="wspd_error", y="wdir_wrt_azimuth_sat")
plt.show()

sns.histplot(data=wspd_subset, x="wspd_error")
plt.show()

# Which buoys are the huge errors coming from?
q90 = np.nanquantile(wspd_subset["wspd_error"], 0.9)
high_error_buoys = np.unique(
    wspd_subset[abs(wspd_subset["wspd_error"]) > q90]["buoy_id"]
)  # only 8 buoys

# Almost all of the errors are coming from one buoy
# That buoy is really close to the shore, which is resulting in high sigma0 and therefore wind speed
high_errors = wspd_subset[abs(wspd_subset["wspd_error"]) > q90]
high_errors.groupby("buoy_id")["buoy_id"].count()

# Drop that buoy from the analysis (buchillonfieldstation)
# Just for the wind speed because for the wind direction it is still able to be based on a 1km area within the lake


###############################################################################################
# Figure 6. Scatterplot of the observed vs predicted wind speed
###############################################################################################

wspd_subset = wind_df[
    ((wind_df["buoy_id"] != "buchillonfieldstation"))
].dropna(subset=["buoy_wspd_10m", "wspd_sat_cmod5n"])

# Make it long for plotting
wspd_subset_long = wspd_subset.melt(
    id_vars=["buoy_image_id", "buoy_wspd_10m", "wind_streak"],
    value_vars=["wspd_era5", "wspd_sat_cmod5n"],
)
# Make vars pretty for plotting
wspd_subset_long["variable"] = wspd_subset_long["variable"].map(
    {"wspd_sat_cmod5n": "CMOD5.N+LG-Mod", "wspd_era5": "ERA5"}
)

# Calculate the slope, R, and pvalue for each group
slope_sar, intercept_sar, r_value_sar, p_value_sar, std_err_sar = (
    scipy.stats.linregress(
        wspd_subset["buoy_wspd_10m"].tolist(), wspd_subset["wspd_sat_cmod5n"].tolist()
    )
)
slope_era5, intercept_era5, r_value_era5, p_value_era5, std_err_era5 = (
    scipy.stats.linregress(
        wspd_subset["buoy_wspd_10m"].tolist(), wspd_subset["wspd_era5"].tolist()
    )
)


wspd_subset_long["variable"] = wspd_subset_long["variable"].map(
    {
        "ERA5": "ERA5: Slope = "
        + str(round(slope_era5, 3))
        + " , $R^2$ = "
        + str(round(r_value_era5**2, 3)),
        "CMOD5.N+LG-Mod": "S1: Slope = "
        + str(round(slope_sar, 3))
        + " , $R^2$ = "
        + str(round(r_value_sar**2, 3)),
    }
)

# Calculate the x and y pairs for the S1 linear model
s1_x = np.linspace(0, 25, 100)
s1_y = np.array([i * slope_sar + intercept_sar for i in s1_x])
era5_x = np.linspace(0, 25, 100)
era5_y = np.array([i * slope_era5 + intercept_era5 for i in era5_x])

fig, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(
    ax=ax,
    x="buoy_wspd_10m",
    y="value",
    hue="variable",
    palette=["#3b528b", "#5ec962"],
    alpha=0.5,
    s=50,
    data=wspd_subset_long,
)
plt.plot(
    np.linspace(0, 25, 100),
    np.linspace(0, 25, 100),
    color="black",
    linestyle="--",
    linewidth=2,
)
plt.plot(s1_x, s1_y, color="#5ec962", linestyle="-", linewidth=2)
plt.plot(era5_x, era5_y, color="#3b528b", linestyle="-", linewidth=2)
ax.set_ylim(0, 25)
ax.set_xlim(0, 25)
ax.set_xlabel("Buoy wind speed (m/s)", size=12)
ax.set_ylabel("Sentinel-1 C-band CMOD5.N Modeled Wind Speed (m/s)", size=12)
ax.set_title("", size=12)
ax.legend(title="")
#plt.show()
plt.savefig(os.path.join(home, "Data/Figures/wspd_buoy_scatter.png"), dpi=1000)


###############################################################################################
# Figure 7: Image of SAR backscatter with wind direction and speed over Lake Washington
###############################################################################################

row = wind_df[
    (
        (
            wind_df["image_id"]
            == "S1A_IW_GRDH_1SDV_20230321T015425_20230321T015450_047736_05BBFD_A2DE"
        )
        & (wind_df["buoy_id"] == "Washington")
    )
]
row.wdir_buoy
row.wdir_corrected
save_examples(row)

row.buoy_wspd_10m

# Export ERA5 polygons cropped around the lake
pld_subset = gpd.GeoDataFrame(
    pld[pld["lake_id"] == buoy_info[buoy_info["id"] == "Washington"]["pld_id"].iloc[0]]
)
pld_subset.to_file(
    os.path.join(home, "Data/Outputs/", row["image_id"].iloc[0], "lake_washington.shp")
)
pld_subset["geometry"] = pld_subset.buffer(0.1)
era5_fp = (
    "C:/Users/kmcquil/Documents/SWOT_WIND/Data/ERA5/Processed/wdir/20230321160000.tif"
)
era5 = rioxarray.open_rasterio(era5_fp)
era5_crop = era5.rio.clip_box(*pld_subset.total_bounds)
era5_crop = era5_crop.astype("float32")
era5_poly = vectorize(era5_crop)
era5_poly = era5_poly.rename(columns={era5_poly.columns.values[0]: "wdir"})
era5_poly.to_file(
    os.path.join(home, "Data/Outputs/", row["image_id"].iloc[0], "era5_grid.shp")
)

###############################################################################################
# Figure of SWOT wind direction and ME over two small lakes in Switzerland
###############################################################################################

row = wind_df[
    (
        (
            wind_df["image_id"]
            == "SWOT_L2_HR_Raster_100m_UTM32T_N_x_x_x_012_307_119F_20240317T165152_20240317T165213_PIC0_01"
        )
        & (wind_df["buoy_id"] == "lakegreifenmeteostation")
    )
]
row.wdir_buoy
row.wdir_corrected
save_examples(row)

# Export ERA5 polygons cropped around the lake
pld_subset = gpd.GeoDataFrame(
    pld[
        pld["lake_id"]
        == buoy_info[buoy_info["id"] == "lakegreifenmeteostation"]["pld_id"].iloc[0]
    ]
)
pld_subset.to_file(
    os.path.join(home, "Data/Outputs/", row["image_id"].iloc[0], "lake_greifen.shp")
)
pld_subset["geometry"] = pld_subset.buffer(0.1)
era5_fp = (
    "C:/Users/kmcquil/Documents/SWOT_WIND/Data/ERA5/Processed/wdir/20240317160000.tif"
)
era5 = rioxarray.open_rasterio(era5_fp)
era5_crop = era5.rio.clip_box(*pld_subset.total_bounds)
era5_crop = era5_crop.astype("float32")
era5_poly = vectorize(era5_crop)
era5_poly = era5_poly.rename(columns={era5_poly.columns.values[0]: "wdir"})
era5_poly.to_file(
    os.path.join(home, "Data/Outputs/", row["image_id"].iloc[0], "era5_grid.shp")
)


###############################################################################################
# Look at the relationship between wind speed and sigma0 as it relates to
# wind direction relative to satellite look direction
# incidence angle
# lake size
###############################################################################################

swot_subset = wind_df.loc[wind_df["satellite"] == "SWOT"]

swot_subset["sig0_db"] = swot_subset["sig0"]
swot_subset["sig0_db"] = np.where(
    swot_subset["sig0_db"] <= 0, 0.01, swot_subset["sig0_db"]
)
swot_subset["sig0_db"] = np.log10(swot_subset["sig0_db"]) * 10


# Overall
fig, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(data=swot_subset, x="wspd_buoy", y="sig0_db", color="#3b528b")
ax.set_xlabel("Buoy wind speed (m/s)", size=12)
ax.set_ylabel("SWOT Sigma0 (dB)", size=12)
plt.savefig(
    os.path.join(home, "Data/Figures/sig_wspd_buoy_scatter.png"),
    dpi=1000,
    bbox_inches="tight",
)


slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
    swot_subset.loc[swot_subset["sig0"] < 400]["buoy_wspd_10m"].tolist(),
    swot_subset.loc[swot_subset["sig0"] < 400]["sig0"].tolist(),
)
slope
intercept
r_value
p_value

# Bin the wind speed incience angle
swot_subset["wspd_buoy_bin"] = pd.cut(
    swot_subset["wspd_buoy"],
    [0, 2, 4, 6, 8, 20],
    labels=["0-2", "2-4", "4-6", "6-8", ">8"],
)
swot_subset["inc_angle_bin"] = pd.cut(
    swot_subset["inc_angle"],
    [0.5, 1.5, 2.5, 3.5, 4.51],
    labels=["0.5-1.5", "1.5-2.5", "2.5-3.5", "3.5-4.5"],
)

fig, ax = plt.subplots(figsize=(6, 6))
sns.boxplot(
    data=swot_subset,
    x="wspd_buoy_bin",
    y="sig0_db",
    hue="inc_angle_bin",
    palette="viridis",
)
ax.set_xlabel("Buoy Wind Speed (m/s)", size=12)
ax.set_ylabel("SWOT Sigma0 (dB)", size=12)
plt.legend(title="Incidence Angle")
plt.savefig(
    os.path.join(home, "Data/Figures/sig_wspd_inc_buoy_boxplot.png"),
    dpi=1000,
    bbox_inches="tight",
)


###############################################################################################
###############################################################################################
# Recreate the plot from AGU with incidence angle
# Get wind speed from ERA5
# Get sigma0 and incidence angle from SWOT raster product
# Extract for each water body that I am studying
###############################################################################################
###############################################################################################

from rasterstats import zonal_stats
import gc
import glob
import pandas as pd
import rioxarray
import geopandas as gpd
import time
import os
from datetime import datetime

home = "C:/Users/kmcquil/Documents/SWOT_WIND/"
swot_files = glob.glob(
    os.path.join(home, "Data/SWOT_L2_HR_Raster/SWOT_L2_HR_Raster_2.0/*.nc")
)
boundary = gpd.read_file(os.path.join(home, "Data/Buoy/pld_with_buoys.shp"))


def extract_swot(infile):

    # Open the netcdf
    ds = rioxarray.open_rasterio(infile, chunks="auto")
    sig0 = ds.sig0
    inc = ds.inc

    # Filter
    sig0 = sig0.where(sig0 != sig0.rio.nodata)
    inc = inc.where(sig0 != sig0.rio.nodata)
    sig0 = sig0.where(ds.sig0_qual <= 2)
    inc = inc.where(ds.sig0_qual <= 2)

    # Add the CRS
    crs_wkt = sig0.crs.attrs["crs_wkt"]
    crs_wkt_split = crs_wkt.split(",")
    epsg = crs_wkt_split[len(crs_wkt_split) - 1].split('"')[1]
    sig0.rio.write_crs("epsg:" + epsg, inplace=True)
    inc.rio.write_crs("epsg:" + epsg, inplace=True)

    # Project boundary shapefile to match ncdf
    boundary_reproj = boundary.to_crs(sig0.rio.crs)
    sig0_np = sig0.to_numpy()[0, :, :]
    inc_np = inc.to_numpy()[0, :, :]
    aff = sig0.rio.transform()
    sig0_stats = pd.DataFrame(
        zonal_stats(
            boundary_reproj,
            sig0_np,
            affine=aff,
            stats="mean",
            nodata=sig0.rio.nodata,
            all_touched=True,
        )
    ).rename(columns={"mean": "sig0_mean"})
    inc_stats = pd.DataFrame(
        zonal_stats(
            boundary_reproj,
            inc_np,
            affine=aff,
            stats="mean",
            nodata=inc.rio.nodata,
            all_touched=True,
        )
    ).rename(columns={"mean": "inc_mean"})
    df = pd.concat([boundary_reproj["lake_id"].iloc[:], sig0_stats, inc_stats], axis=1)
    df["image_id"] = os.path.basename(infile)[:-3]

    ds.close()
    gc.collect()

    # Open the wind direction that matches the time
    date = (
        datetime.strptime(
            os.path.basename(infile).split("_")[13][0:11], "%Y%m%dT%H"
        ).strftime("%Y%m%d%H")
        + "0000"
    )
    wdir_file = glob.glob(
        os.path.join(home, "Data/ERA5/Processed/wdir/" + date + ".tif")
    )
    wspd_file = glob.glob(
        os.path.join(home, "Data/ERA5/Processed/wspd/" + date + ".tif")
    )
    if (len(wdir_file) == 0) | (len(wspd_file) == 0):
        df["wdir_mean"] = np.nan
        df["wspd_mean"] = np.nan
        return df

    ds = rioxarray.open_rasterio(wdir_file[0], chunks=True)
    ds_np = ds.to_numpy()[0, :, :]
    aff = ds.rio.transform()
    boundary_reproj = boundary_reproj.to_crs(ds.rio.crs)
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                zonal_stats(
                    boundary_reproj,
                    ds_np,
                    affine=aff,
                    stats="mean",
                    nodata=ds.rio.nodata,
                    all_touched=True,
                )
            ).rename(columns={"mean": "wdir_mean"}),
        ],
        axis=1,
    )
    ds.close()
    gc.collect()

    ds = rioxarray.open_rasterio(wspd_file[0], chunks=True)
    ds_np = ds.to_numpy()[0, :, :]
    aff = ds.rio.transform()
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                zonal_stats(
                    boundary_reproj,
                    ds_np,
                    affine=aff,
                    stats="mean",
                    nodata=ds.rio.nodata,
                    all_touched=True,
                )
            ).rename(columns={"mean": "wspd_mean"}),
        ],
        axis=1,
    )
    ds.close()
    gc.collect()

    return df


# There are 350 total
swot_df_list = []
k = 0
for file in swot_files:
    k = k + 1
    print(k)
    swot_df_list.append(extract_swot(file))

swot_df = pd.concat(swot_df_list)

swot_df_1 = pd.read_csv(os.path.join(home, "Data/Outputs/swot_era5_lake_1_median.csv"))
swot_df_2 = pd.read_csv(os.path.join(home, "Data/Outputs/swot_era5_lake_2_median.csv"))
swot_df_3 = pd.read_csv(os.path.join(home, "Data/Outputs/swot_era5_lake_3_median.csv"))
swot_df = pd.concat([swot_df_1, swot_df_2, swot_df_3], axis=0)
swot_df = swot_df.dropna(subset=["sig0_mean", "inc_mean", "wdir_mean", "wspd_mean"])
swot_df = swot_df.reset_index(drop=True)
swot_df["lake_id"] = swot_df["lake_id"].astype("Int64").astype(str)
swot_df = swot_df.merge(
    pld[["lake_id", "ref_area"]], left_on="lake_id", right_on="lake_id", how="left"
)

# Bin the wind speed and incience angle
swot_df["wspd_mean_bin"] = pd.cut(
    swot_df["wspd_mean"],
    [0, 2, 4, 6, 8, 20],
    labels=["0-2", "2-4", "4-6", "6-8", ">8"],
)
swot_df["inc_mean_bin"] = pd.cut(
    swot_df["inc_mean"],
    [0.5, 1.5, 2.5, 3.5, 4.51],
    labels=["0.5-1.5", "1.5-2.5", "2.5-3.5", "3.5-4.5"],
)
swot_df["ref_area_bin"] = pd.cut(
    swot_df["ref_area"],
    [0, 20, 100, 100000000],
    labels=["0-20", "20-100", ">100"],
)

swot_df["sig0_mean_db"] = swot_df["sig0_mean"]
swot_df["sig0_mean_db"] = np.where(
    swot_df["sig0_mean_db"] <= 0, 0.01, swot_df["sig0_mean_db"]
)
swot_df["sig0_mean_db"] = np.log10(swot_df["sig0_mean_db"]) * 10

# Overall
fig, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(data=swot_df, x="wspd_mean", y="sig0_mean_db", color="#3b528b")
ax.set_xlabel("ERA5 wind speed (m/s)", size=12)
ax.set_ylabel("SWOT Sigma0 (dB)", size=12)
plt.savefig(
    os.path.join(home, "Data/Figures/sig_wspd_era5_scatter.png"),
    dpi=1000,
    bbox_inches="tight",
)

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
    swot_df["wspd_mean"].tolist(),
    swot_df["sig0_mean"].tolist(),
)
slope
intercept
r_value
p_value


fig, ax = plt.subplots(figsize=(6, 6))
sns.boxplot(
    data=swot_df,
    x="wspd_mean_bin",
    y="sig0_mean_db",
    hue="inc_mean_bin",
    palette="viridis",
)
ax.set_xlabel("ERA5 Wind Speed (m/s)", size=12)
ax.set_ylabel("SWOT Sigma0 (dB)", size=12)
plt.legend(title="Incidence Angle")
plt.show()
plt.savefig(
    os.path.join(home, "Data/Figures/sig_wspd_inc_era5_boxplot.png"),
    dpi=1000,
    bbox_inches="tight",
)


g = sns.FacetGrid(swot_df, col="ref_area_bin")
g.map(
    sns.boxplot,
    data=swot_df,
    x="wspd_mean_bin",
    y="sig0_mean_db",
    hue="inc_mean_bin",
    palette="viridis",
)
plt.show()

np.mean(wind_df['wspd_buoy'] - wind_df['buoy_wspd_10m'])

