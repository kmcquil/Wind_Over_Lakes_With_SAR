###############################################################################################
# Katie McQuillan
# 04/30/2024
# Check for wind streaks
###############################################################################################

import os
import glob as glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rioxarray
from datetime import datetime, timedelta, timezone
import math
import itertools
from matplotlib import pyplot as plt

home = "C:/Users/kmcquil/Documents/SWOT_WIND/"
buoy_sf = gpd.read_file(os.path.join(home, "Data/Buoy/buoy_info.shp"))
boundary = gpd.read_file(os.path.join(home, "Data/Buoy/pld_with_buoys.shp"))

############################################################################################
# Check presence/absense of wind streaks
############################################################################################

# Create a 5 km buffer around buoy, clip the backscatter to the buffered area
# Plot the backscatter, with the buoy location on top
# Set the color scale to the tenth and 90th percentile to visualize streaks
# Save it according to ID number

# Open df of backscatter and wind matched with buoys
wind_df = pd.read_csv(os.path.join(home, "Data/Wind_Direction/wdir_df.csv"))
wind_df["time_diff"] = pd.to_timedelta(wind_df["time_diff"])

# Find rows where wind direction was estimated using local gradient and has a
# buoy wind direction estimate within one hour
wdir_subset = wind_df.dropna(subset=["wdir_corrected", "wdir_buoy"])
wdir_subset = wdir_subset[abs(wdir_subset["time_diff"]) < timedelta(hours=1)]
wdir_subset = wdir_subset[wdir_subset["wdir_id"] == "wdir_10pixels_"]

# Calculate the difference between the LG WDIR and Buoy WDIR
def calc_mae(obs, pred):
    d1 = (pred - obs) % 360
    d2 = 360 - d1
    return min(d1, d2)

wdir_subset["wdir_diff"] = wdir_subset.apply(
    lambda x: calc_mae(x.wdir_corrected, x.wdir_buoy), axis=1
)

# Create a df of just buoy id and image id
df = wdir_subset[["buoy_id", "satellite", "image_id", "wdir_diff"]]
# Add another column with the buoy-image-id
df["buoy_image_id"] = list(range(0, df.shape[0]))
# df.to_csv(os.path.join(home, "Data/Wind_Streaks/wind_streak_check.csv"))

def clip_sig0(buoy_image_id):
    fp_out = os.path.join(home, "Data/Wind_Streaks/Clips", str(buoy_image_id) + ".png")
    # if os.path.isfile(fp_out): return
    row = df[df["buoy_image_id"] == buoy_image_id]
    satellite = row["satellite"].iloc[0]
    buoy_id = row["buoy_id"].iloc[0]
    subset_buoy_sf = buoy_sf[buoy_sf["id"] == buoy_id]

    if satellite == "Sentinel1":
        fp = os.path.join(
            home,
            "Data",
            satellite,
            "Processed",
            os.path.basename(row["image_id"].iloc[0]) + ".nc",
        )
        src = rioxarray.open_rasterio(fp)
        sig0 = src.sig0

    if "SWOT" in satellite:
        fp = os.path.join(
            home,
            "Data",
            "SWOT_L2_HR_Raster",
            satellite,
            os.path.basename(row["image_id"].iloc[0]) + ".nc",
        )
        src = rioxarray.open_rasterio(fp)
        sig0 = src.sig0
        sig0 = sig0.where(sig0 != sig0.rio.nodata)
        # Filter sig0 observations. Only keep observations that corresond to 0 (good), 1(suspect) and 2 (degraded). 3 = bad
        sig0 = sig0.where(src.sig0_qual <= 2)
        # Get the epsg string and update the crs
        crs_wkt = sig0.crs.attrs["crs_wkt"]
        crs_wkt_split = crs_wkt.split(",")
        epsg = crs_wkt_split[len(crs_wkt_split) - 1].split('"')[1]
        sig0.rio.write_crs("epsg:" + epsg, inplace=True)

    # Project boundary shapefile to match ncdf and mask values outside the boundary to nan
    # This is a rough way of filtering out pixels that are not water
    # Convert to numpy array
    boundary_reproj = boundary.to_crs(sig0.rio.crs)
    sig0_clipped = sig0.rio.clip(boundary_reproj.geometry.values, boundary_reproj.crs)
    sig0_clipped = sig0_clipped.where(
        sig0_clipped != sig0_clipped.rio.nodata
    )  # added this post s1

    buoy_reproj = subset_buoy_sf.to_crs(sig0.rio.crs)
    buoy_buffered = buoy_reproj.buffer(20000, cap_style=3)  # I think this is meters
    sig0_clipped_clipped = sig0_clipped.rio.clip(buoy_buffered)
    sig0_clipped_clipped = sig0_clipped_clipped.where(
        sig0_clipped_clipped != sig0_clipped_clipped.rio.nodata
    )  # added this post s1
    sig0_clipped_clipped = sig0_clipped_clipped.squeeze()

    sig0_np = sig0_clipped_clipped.to_numpy()

    f, ax = plt.subplots(figsize=(11, 11))
    sig0_clipped_clipped.plot.imshow(
        cmap="Greys",
        ax=ax,
        vmin=np.nanquantile(sig0_np, 0.1),
        vmax=np.nanquantile(sig0_np, 0.9),
    )
    buoy_reproj.plot(color="red", edgecolor="red", ax=ax)
    ax.set(title="Raster Layer with Vector Overlay")
    ax.axis("off")
    # plt.show()
    plt.savefig(fp_out)
    plt.close()
    src.close()
    return

# Loop through the IDs
buoy_image_ids = df["buoy_image_id"].tolist()
for i in range(788, len(buoy_image_ids)):
    id = buoy_image_ids[i]
    clip_sig0(id)
