##########################################################################################
# Katie Mcquillan
# 04/30/2023
# Performance stats of wind field at buoys 
##########################################################################################

import os
import glob as glob
import numpy as np
import pandas as pd 
import geopandas as gpd
from datetime import datetime, timedelta
import math
import itertools
import seaborn as sns
from matplotlib import pyplot as plt


home = "C:/Users/kmcquil/Documents/SWOT_WIND/"

###############################################################################################
# Calculate how many images are a space/time match with each buoy  
###############################################################################################
wind_df = pd.read_csv(os.path.join(home, "Data/Wind_Direction/wdir_df.csv"))
wind_df["time_diff"] = pd.to_timedelta(wind_df["time_diff"])

# How many buoys have images for eachs satellite
wdir_subset = wind_df.dropna(subset=['wdir_corrected', 'wdir_buoy'])
wdir_subset = wdir_subset[abs(wdir_subset["time_diff"]) < timedelta(hours=1)] 
wdir_subset.groupby("satellite")["buoy_id"].nunique()

# How many separate instances do I need to check for wind streaks 
check_for_ws = wdir_subset[["buoy_id", "wdir_id", "satellite", "image_id", "wdir_corrected", "wdir_buoy"]]
check_for_ws = check_for_ws[check_for_ws["wdir_id"] == "wdir_10pixels_"] 
# Leave 1,756 buoy-image combinations that I need to check for wind streaks 

# Wind direction: How many images are available for each buoy at each resolution within one hour of overpass 
wdir_subset_grouped = wdir_subset[["satellite", "wdir_id", "buoy_id", "wdir_corrected", "wdir_buoy"]].groupby(["satellite", "wdir_id", "buoy_id"]).count().reset_index()
wdir_subset_grouped.groupby(["satellite", "wdir_id"]).agg({"wdir_corrected":["mean", "min", "max"]})

# Wind speed: How many images are available for each buoy at each resolution within one hour of overpass 
wspd_subset = wind_df.dropna(subset=['wspd_sat_cmod5n', 'buoy_wspd_10m'])
wspd_subset = wspd_subset[abs(wspd_subset["time_diff"]) < timedelta(hours=1)]
wspd_subset_grouped = wspd_subset[["satellite", "wdir_id", "buoy_id", "wspd_sat_cmod5n", "buoy_wspd_10m"]].groupby(["satellite", "wdir_id", "buoy_id"]).count().reset_index()
wspd_subset_grouped.groupby(["satellite", "wdir_id"]).agg({"wspd_sat_cmod5n":["mean", "min", "max"]})


###############################################################################################
# Calculate performance stats for each buoy, satellite, resolution combination 
###############################################################################################

# Bias, MAE, RMSE, R2, N for LG wind wind direction, ERA5 wind direction, CMOD5-LG wind speed, CMOD5-ERA5 wind speed, ERA5 wind speed 
# Calculate wind for 360 degrees 
# Calculate for each buoy x satellite 
# Calculate for each satellite 

def calc_performance_stats(df):
    """
    Calculate bias, mae, rmse for wind direction and wind speed 
    df: pd df
    return df of metrics
    """ 

    def calc_bias(o, p, m):
        o_upper = (o + 180) % 360
        if o_upper < o: 
            if ((p>o) | (p<o_upper)): return m
            else: return m*-1
        if o_upper > o:
            if ((p>o) & (p<o_upper)): return m
            else: return m*-1

    # LG Wind Direction 
    sub = df.dropna(subset=['wdir_corrected', 'wdir_buoy'])
    if sub.shape[0] < 3: 
        row1 = pd.DataFrame([['WDIR-LG', sub.shape[0], np.nan, np.nan, np.nan]], columns=['ID', 'N', 'Bias', 'MAE', 'RMSE'])
    else:
        obs = sub["wdir_buoy"].tolist()
        pred = sub["wdir_corrected"].tolist() 
        # MAE. Get the smallest difference 
        d1 = [(p_i - o_i)%360 for p_i, o_i in zip(pred, obs)]
        d2 = [360 - i for i in d1]
        mae = [min(i, j) for i, j in zip(d1, d2)]
        # Bias. If the predicted is within + 180 from the observed, it is positive. If the predicted is within - 180 from the observed, it is negative                
        bias = [calc_bias(o, p, m) for o, p, m in zip(obs, pred, mae)]
        # Root mean square error 
        rmse = math.sqrt(np.square(bias).mean())
        bias = np.mean(bias)
        mae = np.mean(mae)
        row1 = pd.DataFrame([['WDIR-LG', sub.shape[0], bias, mae, rmse]], columns=['ID', 'N', 'Bias', 'MAE', 'RMSE'])

    # ERA5 Wind Direction. Use the same subset as for the LG wind direction so it's a direct comparison 
    if sub.shape[0] < 3: 
        row2 = pd.DataFrame([['WDIR-ERA5', sub.shape[0], np.nan, np.nan, np.nan]], columns=['ID', 'N', 'Bias', 'MAE', 'RMSE'])
    else:
        obs = sub["wdir_buoy"].tolist()
        pred = sub["wdir_era5"].tolist() 
        # MAE. Get the smallest difference 
        d1 = [(p_i - o_i)%360 for p_i, o_i in zip(pred, obs)]
        d2 = [360 - i for i in d1]
        mae = [min(i, j) for i, j in zip(d1, d2)]
        # Bias. If the predicted is within + 180 from the observed, it is positive. If the predicted is within - 180 from the observed, it is negative                
        bias = [calc_bias(o, p, m) for o, p, m in zip(obs, pred, mae)]
        # Root mean square error 
        rmse = math.sqrt(np.square(bias).mean())
        bias = np.mean(bias)
        mae = np.mean(mae)
        row2 = pd.DataFrame([['WDIR-ERA5', sub.shape[0], bias, mae, rmse]], columns=['ID', 'N', 'Bias', 'MAE', 'RMSE'])

    # CMOD5.n LG wind speed 
    sub = df.dropna(subset=['wspd_sat_cmod5n', 'wspd_buoy'])
    if sub.shape[0] < 3:
        row3 = pd.DataFrame([['WSPD-CMOD5-SAR', sub.shape[0], np.nan, np.nan, np.nan]], columns=['ID', 'N', 'Bias', 'MAE', 'RMSE'])
    else:
        obs = sub["wspd_buoy"].tolist()
        pred = sub["wspd_sat_cmod5n"].tolist()
        bias = np.mean(np.subtract(pred, obs))
        mae = np.mean(np.abs(np.subtract(pred, obs)))
        rmse = math.sqrt(np.square(np.subtract(pred,obs)).mean())
        row3 = pd.DataFrame([['WSPD-CMOD5-SAR', sub.shape[0], bias, mae, rmse]], columns=['ID', 'N', 'Bias', 'MAE', 'RMSE'])

    # CMOD5.n ERA5 wind speed 
    if sub.shape[0] < 3:
        row4 = pd.DataFrame([['WSPD-CMOD5-ERA5', sub.shape[0], np.nan, np.nan, np.nan]], columns=['ID', 'N', 'Bias', 'MAE', 'RMSE'])
    else:
        obs = sub["wspd_buoy"].tolist()
        pred = sub["wspd_era5_cmod5n"].tolist()
        bias = np.mean(np.subtract(pred, obs))
        mae = np.mean(np.abs(np.subtract(pred, obs)))
        rmse = math.sqrt(np.square(np.subtract(pred,obs)).mean())
        row4 = pd.DataFrame([['WSPD-CMOD5-ERA5', sub.shape[0], bias, mae, rmse]], columns=['ID', 'N', 'Bias', 'MAE', 'RMSE'])

    # ERA5 wind speed 
    if sub.shape[0] < 3:
        row5 = pd.DataFrame([['WSPD-ERA5', sub.shape[0], np.nan, np.nan, np.nan]], columns=['ID', 'N', 'Bias', 'MAE', 'RMSE'])
    else:
        obs = sub["wspd_buoy"].tolist()
        pred = sub["wspd_era5"].tolist()
        bias = np.mean(np.subtract(pred, obs))
        mae = np.mean(np.abs(np.subtract(pred, obs)))
        rmse = math.sqrt(np.square(np.subtract(pred,obs)).mean())
        row5 = pd.DataFrame([['WSPD-ERA5', sub.shape[0], bias, mae, rmse]], columns=['ID', 'N', 'Bias', 'MAE', 'RMSE'])

    df = pd.concat([row1, row2, row3, row4, row5], axis=0)
    return df


# subset to less than one hour difference and 1km resolution 
wdir_df_subset = wind_df[(abs(wind_df["time_diff"]) < timedelta(hours=1))]

# Loop through each buoy, satellite, resolution combo and create a df of results for all combos
buoy_unique = set(wdir_df_subset['buoy_id'])
satellite_unique = set(wdir_df_subset['satellite'])
resolution_unique = set(wdir_df_subset['wdir_id'])
a = [buoy_unique, satellite_unique, resolution_unique]
combos = list(itertools.product(*a))

stats_list = []
for combo in combos:
    df_sub = wdir_df_subset[((wdir_df_subset["buoy_id"] == combo[0]) & (wdir_df_subset["satellite"] == combo[1]) & (wdir_df_subset["wdir_id"] == combo[2]))]
    stats = calc_performance_stats(df_sub)
    stats = stats.reset_index()
    stats = stats.drop(['index'], axis=1)

    label = pd.DataFrame([[combo[0], combo[1], combo[2]]], columns=["Buoy_ID", "Satellite", "Resolution"])
    label = label.loc[label.index.repeat(stats.shape[0])].reset_index()
    label = label.drop(['index'], axis=1)

    stats_list.append(pd.concat([label, stats], axis=1))
stats_df = pd.concat(stats_list)

# Rename the resolution for figures 
pretty_resolution = stats_df["Resolution"].tolist()
pretty_resolution[:] = [x if x != "wdir_10pixels_" else "1 km" for x in pretty_resolution]
pretty_resolution[:] = [x if x != "wdir_20pixels_" else "2 km" for x in pretty_resolution]
pretty_resolution[:] = [x if x != "wdir_40pixels_" else "4 km" for x in pretty_resolution]
stats_df["Resolution"] = pretty_resolution

# max, min, mean for mae, rmse, bias, n for satellite and resolution 
stats_df_subset = stats_df.dropna()
stats_df_subset.groupby(['Satellite', 'Resolution', 'ID']).agg({'N':['mean', 'min', 'max'], 'Bias':['mean', 'min', 'max'], 'MAE':['mean', 'min', 'max'], 'RMSE':['mean', 'min', 'max']})

# Max a boxplot for LG WDIR comparing bias, mae, and rmse by resolution 
stats_df_lgwdir = stats_df_subset[stats_df_subset["ID"] == "WDIR-LG"]
stats_df_lgwdir.groupby(['Satellite', 'Resolution', 'ID']).agg({'N':['mean', ], 'Bias':['mean', ], 'MAE':['mean',], 'RMSE':['mean',]}).round(2)

fig, ax = plt.subplots(figsize = ( 8 , 5 )) 
sns.boxplot(x="Satellite", y="Bias", hue="Resolution", data=stats_df_lgwdir)
ax.set_xlabel( "Satellite" , size = 12 ) 
ax.set_ylabel( "Bias (degrees)" , size = 12 ) 
ax.set_title( "LG WDIR Performance summarized acrosss buoys" , size = 12) 
plt.show()

fig, ax = plt.subplots(figsize = ( 8 , 5 )) 
sns.boxplot(x="Satellite", y="MAE", hue="Resolution", data=stats_df_lgwdir)
ax.set_xlabel( "Satellite" , size = 12 ) 
ax.set_ylabel( "MAE (degrees)" , size = 12 ) 
#ax.set_title( "LG WDIR Performance summarized acrosss buoys" , size = 12) 
plt.show()

# Make boxplots comparing LG WDIR with ERA5 wdir performance at 1km resolution 
stats_df_wdir = stats_df_subset[(((stats_df_subset["ID"] == "WDIR-LG") | (stats_df_subset["ID"] == "WDIR-ERA5")) & (stats_df_subset["Resolution"] == "1 km"))]
stats_df_wdir.groupby(['Satellite', 'ID']).agg({'N':['mean', ], 'Bias':['mean', ], 'MAE':['mean',], 'RMSE':['mean',]}).round(2)

fig, ax = plt.subplots(figsize = ( 8 , 5 )) 
sns.boxplot(x="Satellite", y="Bias", hue="ID", data=stats_df_wdir)
ax.set_xlabel( "Satellite" , size = 12 ) 
ax.set_ylabel( "Bias (degrees)" , size = 12 ) 
ax.set_title( "Comparison of LG and ERA5 wind direction performance across buoys" , size = 12) 
plt.show()

fig, ax = plt.subplots(figsize = ( 8 , 5 )) 
sns.boxplot(x="Satellite", y="MAE", hue="ID", data=stats_df_wdir)
ax.set_xlabel( "Satellite" , size = 12 ) 
ax.set_ylabel( "MAE (degrees)" , size = 12 ) 
ax.set_title( "Comparison of LG and ERA5 wind direction performance across buoys" , size = 12) 
plt.show()

# Make a boxplot comparing the CMOD5 - LG wind speeds by resolution and satellite 
stats_df_lgwspd = stats_df_subset[stats_df_subset["ID"] == "WSPD-CMOD5-SAR"]
stats_df_lgwspd.groupby(['Satellite', 'Resolution', 'ID']).agg({'N':['mean', "median"], 'Bias':['mean', "median"], 'MAE':['mean',"median"], 'RMSE':['mean',"median"]}).round(2)

fig, ax = plt.subplots(figsize = ( 8 , 5 )) 
sns.boxplot(x="Resolution", y="Bias", data=stats_df_lgwspd)
ax.set_xlabel( "Resolution" , size = 12 ) 
ax.set_ylabel( "Bias (m/s)" , size = 12 ) 
ax.set_title( "CMOD5 - LG Wind speed performance by resolution" , size = 12) 
plt.show()

fig, ax = plt.subplots(figsize = ( 8 , 5 )) 
sns.boxplot(x="Resolution", y="MAE", data=stats_df_lgwspd)
ax.set_xlabel( "Resolution" , size = 12 ) 
ax.set_ylabel( "MAE (m/s)" , size = 12 ) 
ax.set_title( "CMOD5 - LG Wind speed performance by resolution" , size = 12) 
plt.show()



# Make a boxplot comparing the CMOD5 - LG wind speeds by resolution and satellite 
stats_df_lgwspd = stats_df_subset[(((stats_df_subset["ID"] == "WSPD-CMOD5-SAR") | (stats_df_subset["ID"] == "WSPD-CMOD5-ERA5") | (stats_df_subset["ID"] == "WSPD-ERA5")) & (stats_df_subset["Resolution"] == "1 km"))]
stats_df_lgwspd.groupby(['Satellite', 'ID']).agg({'N':['mean', "median"], 'Bias':['mean', "median"], 'MAE':['mean',"median"], 'RMSE':['mean',"median"]}).round(2)

fig, ax = plt.subplots(figsize = ( 8 , 5 )) 
sns.boxplot(x="ID", y="Bias", data=stats_df_lgwspd)
ax.set_xlabel( "Resolution" , size = 12 ) 
ax.set_ylabel( "Bias (m/s)" , size = 12 ) 
ax.set_title( "Comparing wind speed algorithms" , size = 12) 
plt.show()

fig, ax = plt.subplots(figsize = ( 8 , 5 )) 
sns.boxplot(x="ID", y="MAE", data=stats_df_lgwspd)
ax.set_xlabel( "Resolution" , size = 12 ) 
ax.set_ylabel( "MAE (m/s)" , size = 12 ) 
ax.set_title( "Comparing wind speed algorithms" , size = 12) 
plt.show()



# scatter plot observed v predicted wind direction at 1km resolution 

df_scatter = wdir_df_subset[((wdir_df_subset["wdir_id"] == "wdir_10pixels_"))]

fig, ax = plt.subplots(figsize = ( 8 , 8 )) 
sns.scatterplot(ax=ax, x="wdir_buoy", y="wdir_corrected", hue="satellite", data=df_scatter)
plt.plot([0,0], [365,365])
ax.set_xlabel( "Buoy wind direction (degrees)" , size = 12 ) 
ax.set_ylabel( "Local gradient wind direction (degrees)" , size = 12 ) 
#ax.set_title( "" , size = 12) 
plt.show()

fig, ax = plt.subplots(figsize = ( 8 , 8 )) 
sns.scatterplot(ax=ax, x="buoy_wspd_10m", y="wspd_sat_cmod5n", data=df_scatter)
ax.set(xlim=(0,32), ylim=(0,32))
ax.plot((0,32), (0,32), color='black')
ax.set_xlabel( "Buoy wind speed at 10m (m/s)" , size = 12 ) 
ax.set_ylabel( "Wind speed from CMOD5n with LG wdir" , size = 12 ) 
#ax.set_title( "" , size = 12) 
plt.show()



######################################################################################
# Maps
# Local gradient wind direction N, bias and MAE for 1km resolution 
# CMOD5n local gradient wind speed N, bias, and mae for 1km resolution 

buoy_sf = gpd.read_file(os.path.join(home, "Data/Buoy/buoy_info.shp"))
pld = gpd.read_file(os.path.join(home, "Data/Buoy/pld_with_buoys.shp"))

stats_df_subset_wdir_sentinel_1km = stats_df_subset[((stats_df_subset["Resolution"] == "1 km") & (stats_df_subset["ID"] == "WDIR-LG") & (stats_df_subset["Satellite"] == "Sentinel1"))]
stats_df_subset_wdir_sentinel_1km = stats_df_subset_wdir_sentinel_1km.merge(buoy_sf[["id", "type", "pld_id","geometry"]], left_on="Buoy_ID", right_on="id",how="left")
stats_df_subset_wdir_sentinel_1km = gpd.GeoDataFrame(stats_df_subset_wdir_sentinel_1km)
stats_df_subset_wdir_sentinel_1km.to_file(os.path.join(home, "Data/Wind_Direction/stats_wdir_s1_1km.shp"))


stats_df_subset_wspd_sentinel_1km = stats_df_subset[((stats_df_subset["Resolution"] == "1 km") & (stats_df_subset["ID"] == "WSPD-CMOD5-SAR") & (stats_df_subset["Satellite"] == "Sentinel1"))]
stats_df_subset_wspd_sentinel_1km = stats_df_subset_wspd_sentinel_1km.merge(buoy_sf[["id", "type","pld_id", "geometry"]], left_on="Buoy_ID", right_on="id",how="left")
stats_df_subset_wspd_sentinel_1km = gpd.GeoDataFrame(stats_df_subset_wspd_sentinel_1km)
stats_df_subset_wspd_sentinel_1km.to_file(os.path.join(home, "Data/Wind_Direction/stats_wspd_s1_1km.shp"))

# These maps suck, I just looked in arcpro


##########################################################################################
# Plot how the MAE and Bias vary according to sampling time and lake area 

fig, ax = plt.subplots(figsize = ( 8 , 5 )) 
sns.boxplot(x="type", y="Bias", data=stats_df_subset_wdir_sentinel_1km)
ax.set_xlabel( "Sampling time" , size = 12 ) 
ax.set_ylabel( " Bias (degrees)" , size = 12 ) 
ax.set_title( "Sentinel-1 1km " , size = 12) 
plt.show()

fig, ax = plt.subplots(figsize = ( 8 , 5 )) 
sns.boxplot(x="type", y="MAE", data=stats_df_subset_wdir_sentinel_1km)
ax.set_xlabel( "Sampling time" , size = 12 ) 
ax.set_ylabel( " MAE (degrees)" , size = 12 ) 
ax.set_title( "Sentinel-1 1km " , size = 12) 
plt.show()


fig, ax = plt.subplots(figsize = ( 8 , 5 )) 
sns.boxplot(x="type", y="Bias", data=stats_df_subset_wspd_sentinel_1km)
ax.set_xlabel( "Sampling time" , size = 12 ) 
ax.set_ylabel( " Bias (m/s)" , size = 12 ) 
ax.set_title( "Sentinel-1 1km " , size = 12) 
plt.show()

fig, ax = plt.subplots(figsize = ( 8 , 5 )) 
sns.boxplot(x="type", y="MAE", data=stats_df_subset_wspd_sentinel_1km)
ax.set_xlabel( "Sampling time" , size = 12 ) 
ax.set_ylabel( " MAE (m/s)" , size = 12 ) 
ax.set_title( "Sentinel-1 1km " , size = 12) 
plt.show()


# merge the pld reference area with the 
pld["lake_id"] = pld["lake_id"].astype(np.int64).astype(str)
stats_df_subset_wdir_sentinel_1km = stats_df_subset_wdir_sentinel_1km.merge(pld[["lake_id", "ref_area"]], left_on="pld_id", right_on="lake_id", how="left")
stats_df_subset_wspd_sentinel_1km = stats_df_subset_wspd_sentinel_1km.merge(pld[["lake_id", "ref_area"]], left_on="pld_id", right_on="lake_id", how="left")

fig, ax = plt.subplots(figsize = ( 8 , 8 )) 
sns.scatterplot(x="ref_area", y="Bias", data=stats_df_subset_wdir_sentinel_1km)
ax.set_xlabel( "Lake Area (m2)" , size = 12 ) 
ax.set_ylabel( " Bias (degrees)" , size = 12 ) 
ax.set_title( "Sentinel-1 1km " , size = 12) 
plt.show()

fig, ax = plt.subplots(figsize = ( 8 , 8 )) 
sns.scatterplot(x="ref_area", y="MAE", data=stats_df_subset_wdir_sentinel_1km)
ax.set_xlabel( "Lake Area (m2)" , size = 12 ) 
ax.set_ylabel( " MAE (degrees)" , size = 12 ) 
ax.set_title( "Sentinel-1 1km " , size = 12) 
plt.show()


fig, ax = plt.subplots(figsize = ( 8 , 8 )) 
sns.scatterplot(x="ref_area", y="Bias", data=stats_df_subset_wspd_sentinel_1km)
ax.set_xlabel( "Lake Area (m2)" , size = 12 ) 
ax.set_ylabel( " Bias (m/s)" , size = 12 ) 
ax.set_title( "Sentinel-1 1km " , size = 12) 
plt.show()

fig, ax = plt.subplots(figsize = ( 8 , 8 )) 
sns.scatterplot(x="ref_area", y="MAE", data=stats_df_subset_wspd_sentinel_1km)
ax.set_xlabel( "Lake Area (m2)" , size = 12 ) 
ax.set_ylabel( " MAE (m/s)" , size = 12 ) 
ax.set_title( "Sentinel-1 1km " , size = 12) 
plt.show()








#################################################################################
# Performance after I search for wind streaks 
#################################################################################








