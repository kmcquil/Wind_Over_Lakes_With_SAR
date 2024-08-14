##########################################################################################
##########################################################################################
# Katie Mcquillan
# 04/09/2024
# Download SWOT data from PODAAC at buoy locations from 2023 - 2024
##########################################################################################
##########################################################################################

# Import modules
import pandas as pd
import numpy as np
import glob as glob
import os
import geopandas as gpd
import fiona
from shapely.geometry import Point, Polygon
import itertools

# Set home directory
home = "C:/Users/kmcquil/Documents/SWOT_WIND/"

##########################################################################################
# Create geopandas df of science scenes
##########################################################################################

# Science tiles are in kmz format. Convert to geopandas
fiona.drvsupport.supported_drivers["KML"] = "rw"
fiona.drvsupport.supported_drivers["LIBKML"] = "rw"
science_coverage_file = os.path.join(
    home, "Data\SWOT_L2_HR_Raster\swot_science_coverage_20240319.kmz"
)
layers = fiona.listlayers(science_coverage_file)
layers = layers[3:]
gdf_list = []
for layer in layers:
    gdf = gpd.read_file(science_coverage_file, driver="LIBKML", layer=layer)
    gdf_list.append(gdf)
science_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))


# Extract the scene from the description column
def get_scene(desc):
    scene = desc.split("<b>")[3]
    scene = scene.split("</b>")[1]
    scene = scene.split("</p>")[0]
    scene = scene.strip()

    pass_ = desc.split("<b>")[1]
    pass_ = pass_.split("</b>")[1]
    pass_ = pass_.split("<br>")[0]
    pass_ = pass_.strip()

    return pass_ + "_" + scene


science_gdf["scene"] = science_gdf["description"].apply(get_scene)

#  Save as shapefile
science_gdf = science_gdf[["Name", "scene", "geometry"]]
science_gdf.to_file(os.path.join(home, "Data\SWOT_L2_HR_Raster\swot_science_tiles.shp"))

##########################################################################################
# Create geopandas df of cal/val scenes
##########################################################################################

# Convert the cal/val scenes to geopandas and save as a shapefile
calval_coverage_file = os.path.join(
    home, "Data\SWOT_L2_HR_Raster\swot_beta_preval_coverage_20231204.kmz"
)
layers = fiona.listlayers(calval_coverage_file)
calval_gdf = gpd.read_file(calval_coverage_file, driver="LIBKML", layer="scenes")
calval_gdf = calval_gdf[["Name", "geometry"]]
calval_gdf = calval_gdf.rename(columns={"Name": "scene"})
calval_gdf.to_file(os.path.join(home, "Data\SWOT_L2_HR_Raster\swot_beta_scenes.shp"))

##########################################################################################
# Identify the tiles that buoys are located inside
##########################################################################################

# Add a column to the buoy info shp for the calval coverage tiles
calval_gdf = gpd.read_file(
    os.path.join(home, "Data\SWOT_L2_HR_Raster\swot_beta_scenes.shp")
)
science_gdf = gpd.read_file(
    os.path.join(home, "Data\SWOT_L2_HR_Raster\swot_science_tiles.shp")
)
buoy_info_shp = gpd.read_file(os.path.join(home, "Data/Buoy/buoy_info.shp"))
buoy_info_shp["scene_calval"] = str(np.nan)

# Loop through points to find the cal/val scenes buoys are within
for i in range(0, buoy_info_shp.shape[0]):
    # Subset to the point
    pt = buoy_info_shp.iloc[i : i + 1, :]
    pt = pt.reset_index()
    # Create an empty list to store scenes that the point is located inside
    scenes = []
    # Loop through the calval polygons
    for j in range(0, calval_gdf.shape[0]):
        pg = calval_gdf.iloc[j : j + 1, :]
        pg = pg.reset_index()
        wi = list(pt.within(pg["geometry"]))[0]
        if wi == True:
            scenes.append(pg["scene"][0])

    # Once all of the scenes have been checked, add back to the dataframe
    if len(scenes) == 0:
        buoy_info_shp["scene_calval"].iloc[i] = np.nan
    else:
        scenes = ",".join(str(x) for x in scenes)
        buoy_info_shp["scene_calval"].iloc[i] = scenes


# Add a column to the buoy info shp for the science coverage tiles
buoy_info_shp["scene_science"] = str(np.nan)
# Loop through points to find the science scenes they are within
for i in range(0, buoy_info_shp.shape[0]):
    # Subset to the point
    pt = buoy_info_shp.iloc[i : i + 1, :]
    pt = pt.reset_index()
    # Create an empty list to store scenes that the point is located inside
    scenes = []
    # Loop through the science polygons
    for j in range(0, science_gdf.shape[0]):
        pg = science_gdf.iloc[j : j + 1, :]
        pg = pg.reset_index()
        wi = list(pt.within(pg["geometry"]))[0]
        if wi == True:
            scenes.append(pg["scene"][0])

    # Once all of the scenes have been checked, add back to the dataframe
    if len(scenes) == 0:
        buoy_info_shp["scene_science"].iloc[i] = np.nan
    else:
        scenes = ",".join(str(x) for x in scenes)
        buoy_info_shp["scene_science"].iloc[i] = scenes

# Save new info
buoy_info_shp.to_file(os.path.join(home, "Data/Buoy/buoy_info.shp"))

# Redo the buoy info csv with the updated columns
buoy_info = pd.DataFrame(buoy_info_shp.drop(columns="geometry"))
buoy_info.to_csv(os.path.join(home, "Data/Buoy/buoy_info.csv"))

##########################################################################################
# Download SWOT scenes that correspond to buoys
##########################################################################################

# Command to submit in anaconda that will save a txt file of all links available for download 2.0
# podaac-data-downloader -c SWOT_L2_HR_Raster_2.0 -d  C:/Users/kmcquil/Documents/SWOT_WIND/Data/SWOT_L2_HR_Raster/SWOT_L2_HR_Raster_2.0/ --start-date 2023-01-01T00:00:00Z --end-date 2025-01-01T23:59:59Z --dry-run > C:/Users/kmcquil/Documents/SWOT_WIND/Data/SWOT_L2_HR_Raster/SWOT_L2_HR_Raster_2.0/all_files.txt

# Command to submit in anaconda that will save a txt file of all links available for download 1.1
# podaac-data-downloader -c SWOT_L2_HR_Raster_1.1 -d  C:/Users/kmcquil/Documents/SWOT_WIND/Data/SWOT_L2_HR_Raster/SWOT_L2_HR_Raster_1.1/ --start-date 2023-01-01T00:00:00Z --end-date 2025-01-01T23:59:59Z --dry-run > C:/Users/kmcquil/Documents/SWOT_WIND/Data/SWOT_L2_HR_Raster/SWOT_L2_HR_Raster_1.1/all_files.txt

##########################################################################################
# Find the files that correspond to tiles with buoys and download
##########################################################################################


# Create index for files to download and grab links
def check_scene(row, scenes):
    test_string = row["fp"]
    scene = [ele for ele in scenes if (ele in test_string)]
    # istile = np.sum([ele in test_string for ele in tiles]) == 1
    is100 = "_100m_" in test_string
    if (len(scene) > 0) & (is100 == True):
        return scene[0]
    else:
        return np.nan


# Check if those links are already downloaded
def check_downloaded(link, folder):
    link = link.split("/")[5]
    check = (
        os.path.isfile(os.path.join(home, "Data/SWOT_L2_HR_Raster", folder, link))
        == False
    )  # so that we get true values for non-downloaded files
    return check


##########################################################################################
# Start with science orbit
# Create a dt of all potential links for download
##########################################################################################

file_dt = pd.read_csv(
    os.path.join(home, "Data/SWOT_L2_HR_Raster/SWOT_L2_HR_Raster_2.0/all_files.txt")
)
file_dt = file_dt.iloc[:, 1:2]
file_dt = file_dt.rename(columns={file_dt.columns[0]: "fp"})

# Create an array of tiles for the science orbit
buoy_info = pd.read_csv(os.path.join(home, "Data/Buoy/buoy_info.csv"))
science_scenes = list(buoy_info["scene_scie"])
science_scenes = [i.split(",") for i in science_scenes]
science_scenes = list(itertools.chain(*science_scenes))
science_scenes = list(set(science_scenes))

# Find files that correspond to tile of interest
file_dt["idx"] = file_dt.apply(check_scene, axis=1, scenes=science_scenes)
file_dt_dwnld = file_dt.dropna()

# Check if the file has already been downloaded
file_dt_dwnld["link"] = file_dt_dwnld["fp"].str.split("-", n=1, expand=True).iloc[:, 1]
download_mask = file_dt_dwnld["link"].apply(
    check_downloaded, folder="SWOT_L2_HR_Raster_2.0"
)
file_dt_dwnld = file_dt_dwnld[download_mask]
file_dt_dwnld["link"].to_csv(
    os.path.join(home, "Data/SWOT_L2_HR_Raster/SWOT_L2_HR_Raster_2.0/links.txt"),
    sep="\t",
    index=False,
    header=False,
)

## Use WGET to download all of the files
## cd C:\Users\kmcquil\Documents\SWOT_WIND\Data\SWOT_L2_HR_Raster\SWOT_L2_HR_Raster_2.0
## wget --load-cookies .\.urs_cookies --save-cookies .\.urs_cookies --auth-no-challenge=on --keep-session-cookies --user=katiemcquillan --password=Bulge1944! -i .\links.txt

##########################################################################################
# And also the calval orbit
##########################################################################################

# Create a dt of all potential links for download
file_dt = pd.read_csv(
    os.path.join(home, "Data/SWOT_L2_HR_Raster/SWOT_L2_HR_Raster_1.1/all_files.txt")
)
file_dt = file_dt.iloc[:, 1:2]
file_dt = file_dt.rename(columns={file_dt.columns[0]: "fp"})

# Create an array of tiles for the calval orbit
calval_scenes = list(buoy_info["scene_calv"])
calval_scenes = [str(i).split(",") for i in calval_scenes]
calval_scenes = list(itertools.chain(*calval_scenes))
calval_scenes = list(set(calval_scenes))
calval_scenes = [x for x in calval_scenes if str(x) != "nan"]

# Find files that correspond to tile of interest
file_dt["idx"] = file_dt.apply(check_scene, axis=1, scenes=calval_scenes)
file_dt_dwnld = file_dt.dropna()

# Check if the file has already been downloaded
file_dt_dwnld["link"] = file_dt_dwnld["fp"].str.split("-", n=1, expand=True).iloc[:, 1]
download_mask = file_dt_dwnld["link"].apply(
    check_downloaded, folder="SWOT_L2_HR_Raster_1.1"
)
file_dt_dwnld = file_dt_dwnld[download_mask]
file_dt_dwnld["link"].to_csv(
    os.path.join(home, "Data/SWOT_L2_HR_Raster/SWOT_L2_HR_Raster_1.1/links.txt"),
    sep="\t",
    index=False,
    header=False,
)

## Use WGET to download all of the files
## cd C:\Users\kmcquil\Documents\SWOT_WIND\Data\SWOT_L2_HR_Raster\SWOT_L2_HR_Raster_1.1
## wget --load-cookies .\.urs_cookies --save-cookies .\.urs_cookies --auth-no-challenge=on --keep-session-cookies --user=katiemcquillan --password=Bulge1944! -i .\links.txt
