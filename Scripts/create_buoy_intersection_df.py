##########################################################################################
# Katie Mcquillan
# 04/18/2024
# Create a DF of which images intersect which buoys 
##########################################################################################

# Import libraries 
import geopandas as gpd
import pandas as pd 
import numpy as np 
import os
import glob 
import rioxarray
from pyproj import Transformer

home = "C:/Users/kmcquil/Documents/SWOT_WIND/"

# Load buoys 
buoy_info_shp = gpd.read_file(os.path.join(home, "Data/Buoy/buoy_info.shp"))

# List all SWOT and Sentinel files 
s1_files = glob.glob(os.path.join(home, "Data/Sentinel1/Processed/*.nc"))
swot1_files = glob.glob(os.path.join(home, "Data/SWOT_L2_HR_Raster/SWOT_L2_HR_Raster_1.1/*.nc"))
swot2_files = glob.glob(os.path.join(home, "Data/SWOT_L2_HR_Raster/SWOT_L2_HR_Raster_2.0/*.nc"))


# I don't have a kmz of regular s1 orbits, so find which buoys intersect which files for s1 
# Check if buoy intersects with s1 images
def check_int(buoy, rds, file):
    # Checks for intersection. 
    # Buoy is a row from the buoy info shp
    # rds is the file opened with rioxarray 
    # Return
    # If there is an intersection, return a pd df row with the buoy id and image name, and folder 
    # If not, return na 

    # Get the long and lat from the buoy 
    coords = buoy.get_coordinates()
    lon = coords['x'][0]
    lat = coords['y'][0]
    # Extract value from raster
    value = rds.sel(x=lon, y=lat, method='nearest').values
    if np.isnan(value[0]) == True: 
        return np.nan
    else:
        s1_id = os.path.basename(file)
        df = pd.DataFrame({'buoy_id':[buoy['id'].iloc[0]], 'image_id':[os.path.basename(file)], 'satellite': ['Sentinel1']})
        return df


image_dt = []
K = 0
for file in s1_files:
    K = K + 1
    print(K)
    rds = rioxarray.open_rasterio(file)
    rds = rds['inc_angle']
    buoy_reproj = buoy_info_shp.to_crs(rds.rio.crs)
    
    for i in range(0, buoy_info_shp.shape[0]):
        buoy = buoy_reproj.iloc[i:i+1, :]
        buoy = buoy.reset_index()
        image_dt.append(check_int(buoy, rds, file))

image_dt =  [x for x in image_dt if isinstance(x, pd.DataFrame)]
image_dt = pd.concat(image_dt)

image_dt = image_dt.rename(columns={"s1_id":"image_id"})
#####################################################################################
# the SWOT orbit is regular. Use the scene ID for the calval and science orbits to 
# extract the buoy_id, swot_id, and satellite 

swot_dt = []
for i in range(0, buoy_info_shp.shape[0]):
    # Subset to the buoy 
    buoy = buoy_info_shp.iloc[i:i+1, :]
    
    # some buoys don't have a calval scene that overlaps
    if buoy['scene_calv'].iloc[0] is None:
        science_scenes = buoy['scene_scie'].iloc[0].split(",")
        all_files = [file for file in swot2_files if any([scene in file for scene in science_scenes])]
    else:
        # Get the scene id for the buoy
        calval_scenes = buoy['scene_calv'].iloc[0].split(",")
        science_scenes = buoy['scene_scie'].iloc[0].split(",")

        # Find the corresponding files 
        calval_files = [file for file in swot1_files if any([scene in file for scene in calval_scenes])]
        science_files = [file for file in swot2_files if any([scene in file for scene in science_scenes])]
        all_files = calval_files + science_files

    # Convert to df 
    all_folders = [os.path.dirname(file).split("/")[-1] for file in all_files]
    buoy_id = [str(buoy['id'].iloc[0])]*len(all_folders)
    row = pd.DataFrame({'buoy_id':buoy_id, 'image_id':all_files, 'satellite':all_folders})
    swot_dt.append(row)
swot_dt = pd.concat(swot_dt)

# Combine the swot and s1 dataframes together for a full list of each buoy and its iamges that intersect 
all_dt = pd.concat([image_dt, swot_dt])
all_dt.to_csv(os.path.join(home, "Data", "buoy_satellite_intersection.csv"))

# Summarize how many s1 and swot1 and swot2 images are at each buoy 
buoy_summary = all_dt.groupby(['buoy_id', 'satellite']).count().reset_index()
buoy_summary = buoy_summary.pivot(index='buoy_id', columns='satellite', values='image_id')


