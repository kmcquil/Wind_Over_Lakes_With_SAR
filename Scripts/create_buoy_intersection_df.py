##########################################################################################
# Katie Mcquillan
# 04/18/2024
# Create a DF of which images intersect which buoys
##########################################################################################

# Import modules
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import glob
import rioxarray
from pyproj import Transformer

# Set home directory
home = "C:/Users/kmcquil/Documents/SWOT_WIND/"

# Load buoy shapefile
buoy_info_shp = gpd.read_file(os.path.join(home, "Data/Buoy/buoy_info.shp"))

##########################################################################################
# Find buoy intersections with each S1 image
##########################################################################################

# List S1 files
s1_files = glob.glob(os.path.join(home, "Data/Sentinel1/Processed/*.nc"))

# No shapefile of regular S1 orbits, so find buoys intersections with each S1 file
def check_int(buoy, rds, file):
    """
    This function checks for intersection between buoy and image extent.
    Inputs:
        buoy: row from buoy_info geopandas df corresponding to a buoy
        rds: image opened with rioxarray
    Outputs:
        If there is an intersection, return a pd df row with the buoy id and image name, and folder
        If not, return np.nan
    """
    # Extract buoy coordinates
    coords = buoy.get_coordinates()
    lon = coords["x"][0]
    lat = coords["y"][0]

    # Extract image value at buoy location
    value = rds.sel(x=lon, y=lat, method="nearest").values

    # If no value is extracted, there is no intersection. Return np.nan
    if np.isnan(value[0]) == True:
        return np.nan

    # If there is an intersection, return buoy, imagae, satellite info as pd df
    else:
        df = pd.DataFrame(
            {
                "buoy_id": [buoy["id"].iloc[0]],
                "image_id": [os.path.basename(file)],
                "satellite": ["Sentinel1"],
            }
        )
        return df

# Create empty list to store buoy-image intersection pd dfs
image_dt = []
# Loop through image to check for buoy intersections
for file in s1_files:
    # Open the image and select incidence angle bc incidence angle will never be nan
    rds = rioxarray.open_rasterio(file)
    rds = rds["inc_angle"]
    # Reproject buoy info to match image since image crs can change based on location in world
    buoy_reproj = buoy_info_shp.to_crs(rds.rio.crs)
    # Loop through each buoy and append intersections to list
    for i in range(0, buoy_info_shp.shape[0]):
        buoy = buoy_reproj.iloc[i : i + 1, :]
        buoy = buoy.reset_index()
        image_dt.append(check_int(buoy, rds, file))

# Drop elements of list that are np.nan and combine to one df
image_dt = [x for x in image_dt if isinstance(x, pd.DataFrame)]
image_dt = pd.concat(image_dt)
image_dt = image_dt.rename(columns={"s1_id": "image_id"})

##########################################################################################
# Find buoy intersections with each SWOT image
# SWOT released a shapefile of the orbit, so no need to check each individual image
# for buoy intersection. Instead, just find the scenes
##########################################################################################

# List all SWOT files
swot1_files = glob.glob(
    os.path.join(home, "Data/SWOT_L2_HR_Raster/SWOT_L2_HR_Raster_1.1/*.nc")
)
swot2_files = glob.glob(
    os.path.join(home, "Data/SWOT_L2_HR_Raster/SWOT_L2_HR_Raster_2.0/*.nc")
)

# Create empty list to append intersections
swot_dt = []

# Loop through each buoy
for i in range(0, buoy_info_shp.shape[0]):

    # Subset to the buoy
    buoy = buoy_info_shp.iloc[i : i + 1, :]

    # Some buoys don't have a calval scene that overlaps
    if buoy["scene_calv"].iloc[0] is None:
        science_scenes = buoy["scene_scie"].iloc[0].split(",")
        all_files = [
            file
            for file in swot2_files
            if any([scene in file for scene in science_scenes])
        ]
    else:
        # Get the scene id for the buoy
        calval_scenes = buoy["scene_calv"].iloc[0].split(",")
        science_scenes = buoy["scene_scie"].iloc[0].split(",")

        # Find the corresponding files
        calval_files = [
            file
            for file in swot1_files
            if any([scene in file for scene in calval_scenes])
        ]
        science_files = [
            file
            for file in swot2_files
            if any([scene in file for scene in science_scenes])
        ]
        all_files = calval_files + science_files

    # Convert to df
    all_folders = [os.path.dirname(file).split("/")[-1] for file in all_files]
    buoy_id = [str(buoy["id"].iloc[0])] * len(all_folders)
    row = pd.DataFrame(
        {"buoy_id": buoy_id, "image_id": all_files, "satellite": all_folders}
    )
    swot_dt.append(row)

# Create one final swot df
swot_dt = pd.concat(swot_dt)

# Combine the swot and s1 dataframes together for a full list of each buoy and its images that intersect
all_dt = pd.concat([image_dt, swot_dt])
all_dt.to_csv(os.path.join(home, "Data", "buoy_satellite_intersection.csv"))