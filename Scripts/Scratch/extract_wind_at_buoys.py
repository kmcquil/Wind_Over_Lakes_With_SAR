##########################################################################################
# Katie Mcquillan
# 04/23/2024
# Extract backscatter, incidence angle, azimuth, wind direction
# and calc wind speed at each buoy. Save the DF.
##########################################################################################

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

home = "C:/Users/kmcquil/Documents/SWOT_WIND/"
buoy_sat_int = pd.read_csv(os.path.join(home, "Data", "buoy_satellite_intersection.csv"))
buoy_sf = gpd.read_file(os.path.join(home, "Data/Buoy/buoy_info.shp"))

##########################################################################################
# Functions  
##########################################################################################

def extract_buoy_wdir(buoy_id, wdir_id):
    """ 
    Extract wdir from the satellite, era5, and buoy 
    buoy_id: string of the buoy id
    wdir_id: string of the prefix to the wdir files - "widr_10pixels" or "wdir_20pixels" ect. 
    return df of wdir 
    """
    # Open the buoy df 
    buoy_df = pd.read_csv(os.path.join(home, "Data/Buoy/Processed", buoy_id + ".csv"))
    buoy_df['datetime']= pd.to_datetime(buoy_df['datetime'], utc=True, format='ISO8601')

    # Subset to the rows that correspond to the buoy 
    subset_buoy_sat_int = buoy_sat_int[buoy_sat_int["buoy_id"] == buoy_id]
    subset_buoy_sf = buoy_sf[buoy_sf['id'] == buoy_id]

    # Loop through each row and extract the wind direction at buoy location and add to list
    row = []
    for i in range(0, subset_buoy_sat_int.shape[0]):
        satellite = subset_buoy_sat_int["satellite"].iloc[i]
        image_id = os.path.basename(subset_buoy_sat_int["image_id"].iloc[i])[:-3]
        if satellite == "Sentinel1": overpass_datetime = datetime.strptime(image_id.split("_")[5], "%Y%m%dT%H%M%S")
        else: overpass_datetime = datetime.strptime(image_id.split("_")[13], "%Y%m%dT%H%M%S")
        overpass_datetime = overpass_datetime.replace(tzinfo=timezone.utc)
        
        # If the wind direction raster of this file doesn't exist, it's because there wasn't a space/time matchup. skip to the next
        wdir_fp = os.path.join(home, "Data/Wind_Direction", satellite, wdir_id + image_id + ".tif")
        if os.path.isfile(wdir_fp) == False: continue

        # Extract from satellite without ERA5 correction 
        with rasterio.open(wdir_fp) as src:
            buoy_reproj = subset_buoy_sf.to_crs(src.crs)
            coords = buoy_reproj.get_coordinates().values
            wdir = [x[0] for x in src.sample(coords)][0]

        # Extract from satellite with ERA5 correction 
        wdir_fp = os.path.join(home, "Data/Wind_Direction", satellite, "ERA5_Corrected", "corrected_" + wdir_id + image_id + ".tif")
        with rasterio.open(wdir_fp) as src:
            buoy_reproj = subset_buoy_sf.to_crs(src.crs)
            coords = buoy_reproj.get_coordinates().values
            wdir_corrected = [x[0] for x in src.sample(coords)][0]

        # Extract from ERA5 
        era5_datetime = datetime.strftime(overpass_datetime, "%Y%m%d%H") + "0000"
        wdir_fp = os.path.join(home, "Data/ERA5/Processed/wdir", era5_datetime + ".tif")
        with rasterio.open(wdir_fp) as src:
            buoy_reproj = subset_buoy_sf.to_crs(src.crs)
            coords = buoy_reproj.get_coordinates().values
            wdir_era5 = [x[0] for x in src.sample(coords)][0]

        # Extract from ERA5 
        wspd_fp = os.path.join(home, "Data/ERA5/Processed/wspd", era5_datetime + ".tif")
        with rasterio.open(wspd_fp) as src:
            buoy_reproj = subset_buoy_sf.to_crs(src.crs)
            coords = buoy_reproj.get_coordinates().values
            wspd_era5 = [x[0] for x in src.sample(coords)][0]

        # Extract the buoy wdir 
        # Find the buoy observation with the smallest difference that is before overpass time 
        diffs = (buoy_df['datetime'] - overpass_datetime).tolist()
        # Just want differences less than 0 so that the buoy observation is before overpass time 
        diffs_less0 = [t for t in diffs if t < pd.Timedelta(0)]
        # To find the closest to 0, get the max
        if len(diffs_less0) == 0: 
           wdir_buoy = np.nan
           wspd_buoy = np.nan
           buoy_datetime = np.nan
           diff = np.nan 
        else:
            min_diff = max(diffs_less0)
            index_min_diff = diffs.index(min_diff)
            # Subset to the buoy row corresponding to the minimum time difference
            subset_buoy_df = buoy_df.iloc[index_min_diff]
            buoy_datetime = subset_buoy_df['datetime'].to_pydatetime()
            diff = buoy_datetime - overpass_datetime
            wdir_buoy = subset_buoy_df['wdir']
            wspd_buoy = subset_buoy_df['wspd']
        
        # Put into pd df 
        row.append(pd.DataFrame({"buoy_id":[buoy_id], 
                                 "wdir_id":[wdir_id],
                                 "satellite":[satellite],
                                 "image_id":[image_id],
                                 "overpass_datetime":[overpass_datetime], 
                                 "wdir":[wdir], 
                                 "wdir_corrected":[wdir_corrected], 
                                 "wdir_era5":[wdir_era5], 
                                 "wspd_era5":[wspd_era5],
                                 "wdir_buoy":[wdir_buoy], 
                                 "wspd_buoy":[wspd_buoy],
                                 "buoy_datetime":[buoy_datetime],
                                 "time_diff":[diff]}))
    wdir_df = pd.concat(row)
    return wdir_df 


def extract_buoy_sigma0(buoy_id):
    """ 
    Extract sigma0, incidence angle, and platform heading from all SAR images 
    buoy_id: string of the buoy id
    return df of wdir 
    """
    # Open the buoy df 
    buoy_df = pd.read_csv(os.path.join(home, "Data/Buoy/Processed", buoy_id + ".csv"))
    buoy_df['datetime']= pd.to_datetime(buoy_df['datetime'], utc=True, format='ISO8601')

    # Subset to the rows that correspond to the buoy 
    subset_buoy_sat_int = buoy_sat_int[buoy_sat_int["buoy_id"] == buoy_id]
    subset_buoy_sf = buoy_sf[buoy_sf['id'] == buoy_id]

    # Loop through each row and extract the wind direction at buoy location and add to list
    row = []
    for i in range(0, subset_buoy_sat_int.shape[0]):
        satellite = subset_buoy_sat_int["satellite"].iloc[i]
        image_id = os.path.basename(subset_buoy_sat_int["image_id"].iloc[i])[:-3]
        if satellite == "Sentinel1": overpass_datetime = datetime.strptime(image_id.split("_")[5], "%Y%m%dT%H%M%S")
        else: overpass_datetime = datetime.strptime(image_id.split("_")[13], "%Y%m%dT%H%M%S")
        overpass_datetime = overpass_datetime.replace(tzinfo=timezone.utc)
        
        # Extract sigma0, incidence angle, platform heading  
        if satellite == "Sentinel1":
            fp = os.path.join(home, "Data", "Sentinel1", "Processed", image_id + ".nc")
            with rioxarray.open_rasterio(fp) as src:
                buoy_reproj = subset_buoy_sf.to_crs(src.rio.crs)
                coords = buoy_reproj.get_coordinates().values
                sig0 = src.sig0.sel(x=coords[0][0], y=coords[0][1], method='nearest').to_numpy()[0]
                inc_angle = src.inc_angle.sel(x=coords[0][0], y=coords[0][1], method='nearest').to_numpy()[0]
                platform_heading = src.platform_heading

        if "SWOT" in satellite:
            fp = os.path.join(home, "Data", "SWOT_L2_HR_Raster", satellite, image_id + ".nc")
            with rioxarray.open_rasterio(fp) as src:
                sig0 = src.sig0     
                sig0 = sig0.where(sig0 != sig0.rio.nodata)
                crs_wkt = sig0.crs.attrs['crs_wkt']
                crs_wkt_split = crs_wkt.split(',')
                epsg = crs_wkt_split[len(crs_wkt_split)-1].split('"')[1]
                sig0.rio.write_crs("epsg:"+epsg, inplace=True)
                buoy_reproj = subset_buoy_sf.to_crs(sig0.rio.crs)
                coords = buoy_reproj.get_coordinates().values
                sig0 = sig0.sel(x=coords[0][0], y=coords[0][1], method='nearest').to_numpy()[0]
                inc_angle = src.inc.sel(x=coords[0][0], y=coords[0][1], method='nearest').to_numpy()[0]
                platform_heading = np.nan # don't need this because I'm not calculating wind speeed from swot 

        # Put into pd df 
        row.append(pd.DataFrame({"buoy_id":[buoy_id], 
                                 "satellite":[satellite],
                                 "image_id":[image_id],
                                 "overpass_datetime":[overpass_datetime], 
                                 "sig0":[sig0],
                                 "inc_angle":[inc_angle],
                                 "platform_heading":[platform_heading]
                                 }))
    sig0_df = pd.concat(row)
    return sig0_df 
    


def cmod5n_inverse(sigma0_obs, phi, incidence, iterations=10):
    '''!     ---------
    !     cmod5n_inverse(sigma0_obs, phi, incidence, iterations)
    !         inputs:
    !              sigma0_obs     Normalized Radar Cross Section [linear units]
    !              phi   in [deg] angle between azimuth and wind direction
    !                    (= D - AZM)
    !              incidence in [deg] incidence angle
    !              iterations: number of iterations to run
    !         output:
    !              Wind speed, 10 m, neutral stratification 
    !
    !        All inputs must be Numpy arrays of equal sizes
    !
    !    This function iterates the forward CMOD5N function
    !    until agreement with input (observed) sigma0 values   
    !---------------------------------------------------------------------
       '''
    from numpy import ones, array
    
    # First guess wind speed
    V = array([10.])*ones(sigma0_obs.shape)
    step=10.
    
    # Iterating until error is smaller than threshold
    for iterno in range(1, iterations):
        print(iterno)
        sigma0_calc = cmod5n_forward(V, phi, incidence)
        ind = sigma0_calc-sigma0_obs>0
        V = V + step
        V[ind] = V[ind] - 2*step 
        step = step/2

    #mdict={'s0obs':sigma0_obs,'s0calc':sigma0_calc}
    #from scipy.io import savemat
    #savemat('s0test',mdict)
        
    return V



def cmod5n_forward(v,phi,theta):
    '''!     ---------
    !     cmod5n_forward(v, phi, theta)
    !         inputs:
    !              v     in [m/s] wind velocity (always >= 0)
    !              phi   in [deg] angle between azimuth and wind direction
    !                    (= D - AZM)
    !              theta in [deg] incidence angle
    !         output:
    !              CMOD5_N NORMALIZED BACKSCATTER (LINEAR)
    !
    !        All inputs must be Numpy arrays of equal sizes
    !
    !     A. STOFFELEN              MAY  1991 ECMWF  CMOD4
    !     A. STOFFELEN, S. DE HAAN  DEC  2001 KNMI   CMOD5 PROTOTYPE
    !     H. HERSBACH               JUNE 2002 ECMWF  COMPLETE REVISION
    !     J. de Kloe                JULI 2003 KNMI,  rewritten in fortan90
    !     A. Verhoef                JAN  2008 KNMI,  CMOD5 for neutral winds
    !     K.F.Dagestad              OCT 2011 NERSC,  Vectorized Python version
    !---------------------------------------------------------------------
       '''
        
    from numpy import cos, exp, tanh, array
    
    DTOR   = 57.29577951
    THETM  = 40.
    THETHR = 25.
    ZPOW   = 1.6
    
    # NB: 0 added as first element below, to avoid switching from 1-indexing to 0-indexing
    C = [0, -0.6878, -0.7957,  0.3380, -0.1728, 0.0000,  0.0040, 0.1103, 0.0159, 
          6.7329,  2.7713, -2.2885, 0.4971, -0.7250, 0.0450, 
          0.0066,  0.3222,  0.0120, 22.7000, 2.0813,  3.0000, 8.3659,
          -3.3428,  1.3236,  6.2437,  2.3893, 0.3249,  4.1590, 1.6930]
    Y0 = C[19]
    PN = C[20]
    A  = C[19]-(C[19]-1)/C[20]

    B  = 1./(C[20]*(C[19]-1.)**(3-1))

#  !  ANGLES
    FI=phi/DTOR
    CSFI = cos(FI)
    CS2FI= 2.00 * CSFI * CSFI - 1.00

    X  = (theta - THETM) / THETHR
    XX = X*X

    #  ! B0: FUNCTION OF WIND SPEED AND INCIDENCE ANGLE
    A0 =C[ 1]+C[ 2]*X+C[ 3]*XX+C[ 4]*X*XX
    A1 =C[ 5]+C[ 6]*X
    A2 =C[ 7]+C[ 8]*X

    GAM=C[ 9]+C[10]*X+C[11]*XX
    S0 =C[12]+C[13]*X
    
    # V is missing! Using V=v as substitute, this is apparently correct
    V=v
    S = A2*V
    S_vec = S.copy() 
    #SlS0 = [S_vec<S0]
    SlS0 = S_vec<S0
    S_vec[SlS0]=S0[SlS0]
    A3=1./(1.+exp(-S_vec))
    SlS0 = (S<S0)
    A3[SlS0]=A3[SlS0]*(S[SlS0]/S0[SlS0])**( S0[SlS0]*(1.- A3[SlS0]))
    #A3=A3*(S/S0)**( S0*(1.- A3))
    B0=(A3**GAM)*10.**(A0+A1*V)
        
    #  !  B1: FUNCTION OF WIND SPEED AND INCIDENCE ANGLE
    B1 = C[15]*V*(0.5+X-tanh(4.*(X+C[16]+C[17]*V)))
    B1 = C[14]*(1.+X)- B1
    B1 = B1/(exp( 0.34*(V-C[18]) )+1.)

    #  !  B2: FUNCTION OF WIND SPEED AND INCIDENCE ANGLE
    V0 = C[21] + C[22]*X + C[23]*XX
    D1 = C[24] + C[25]*X + C[26]*XX
    D2 = C[27] + C[28]*X

    V2 = (V/V0+1.)
    V2ltY0 = V2<Y0
    V2[V2ltY0] = A+B*(V2[V2ltY0]-1.)**PN
    B2 = (-D1+D2*V2)*exp(-V2)

    #  !  CMOD5_N: COMBINE THE THREE FOURIER TERMS
    CMOD5_N = B0*(1.0+B1*CSFI+B2*CS2FI)**ZPOW
    return CMOD5_N


##########################################################################################
# Extract sigma0 and wind direction at buoys 
##########################################################################################
# List unique buoys and extractsigma0 and wind direction 
buoy_ids = buoy_sf['id'].tolist()
sig0_list = list(map(lambda p: extract_buoy_sigma0(p), buoy_ids))
sig0_df = pd.concat(sig0_list)
sig0_df.to_csv(os.path.join(home, "Data/Wind_Direction/sigma0_df.csv"))


# Wind direction at 1km 
wdir_list = list(map(lambda p: extract_buoy_wdir(p, "wdir_10pixels_"), buoy_ids))
wdir_10pixels_df = pd.concat(wdir_list)
wdir_10pixels_df.to_csv(os.path.join(home, "Data/Wind_Direction/wdir_10pixels_df.csv"))


# Wind direction at 2km 
wdir_list = list(map(lambda p: extract_buoy_wdir(p, "wdir_20pixels_"), buoy_ids))
wdir_20pixels_df = pd.concat(wdir_list)
wdir_20pixels_df.to_csv(os.path.join(home, "Data/Wind_Direction/wdir_20pixels_df.csv"))


# Wind direction at 4km 
wdir_list = list(map(lambda p: extract_buoy_wdir(p, "wdir_40pixels_"), buoy_ids))
wdir_40pixels_df = pd.concat(wdir_list)
wdir_40pixels_df.to_csv(os.path.join(home, "Data/Wind_Direction/wdir_40pixels_df.csv"))


##############################################################################################
# Combine all DFs so that there is one with wind and sigma0 for all resolutions
##############################################################################################
# Combine wind and sigma0 dfs 
wdir_df = pd.concat([wdir_10pixels_df, wdir_20pixels_df, wdir_40pixels_df])
wdir_df = pd.merge(wdir_df, sig0_df[["buoy_id", "satellite", "image_id", "sig0", "inc_angle", "platform_heading"]], 
                   on=["buoy_id", "satellite", "image_id"], how="left")

# Add in the buoy height 
wdir_df = pd.merge(wdir_df, buoy_sf[['id', 'height']], left_on='buoy_id', right_on='id', how='left')
wdir_df = wdir_df.drop('id', axis=1)

# Calculate wind direction relative to azimuth - this is just for S1 since that's the only one I will calc wind speed for 
# S1 is always right looking 
look_direction = np.mod(wdir_df["platform_heading"] + 90, 360)
wdir_df["wdir_wrt_azimuth_sat"] = np.mod(wdir_df["wdir_corrected"] - look_direction, 360)
wdir_df["wdir_wrt_azimuth_era5"] = np.mod(wdir_df["wdir_era5"] - look_direction, 360)

# Convert buoy wind speed to 10 m height 
# use method from this paper: https://www.mdpi.com/2076-3263/13/12/361
# U10 = Ua * ln(Za / Z0) / ln(H10 / Z0) 
# U10 is wind speed at 10m, Ua is wind speed a buoy height
# Z0 is constant at 1.52x10-4
# Za is buoy height, H10 = 10
def wspd10(buoy_wspd, buoy_height):
    Z0 = 1.52*(10**-4)
    return buoy_wspd * math.log(buoy_height/Z0) / math.log(10/Z0)
wdir_df['buoy_wspd_10m'] = wdir_df.apply(lambda x: wspd10(x.wspd_buoy, x.height), axis=1)
wdir_df = wdir_df.reset_index()

###############################################################################################
# Calculate wind speed using CMOD5.n  
###############################################################################################

def cmod5n(sigma0_obs, phi, incidence):
    if ((np.isnan(sigma0_obs)==True) | (np.isnan(phi)==True) | (np.isnan(incidence)==True)) :
        return np.nan
    else:
        return cmod5n_inverse( np.array([sigma0_obs]), np.array([phi]), np.array([incidence]), iterations=10)[0]

wdir_df["wspd_sat_cmod5n"] = wdir_df.apply(lambda x: cmod5n(x.sig0, x.wdir_wrt_azimuth_sat, x.inc_angle), axis=1)
wdir_df["wspd_era5_cmod5n"] = wdir_df.apply(lambda x: cmod5n(x.sig0, x.wdir_wrt_azimuth_era5, x.inc_angle), axis=1)

# Save
wdir_df.to_csv(os.path.join(home, "Data/Wind_Direction/wdir_df.csv"))



###############################################################################################
# Calculate wind speed for two images to use as example figures 
###############################################################################################
wind_df = pd.read_csv(os.path.join(home, "Data/Wind_Direction/wdir_df.csv"))


# West lake ontario 
###############################################################################################
# Load in the sig0, inc angle, and wind direction datasets 
from rasterio.enums import Resampling
import xarray

infile = os.path.join(home, "Data/Sentinel1/Processed/S1A_IW_GRDH_1SDV_20230520T231641_20230520T231706_048624_05D926_1C5B.nc")
ds = rioxarray.open_rasterio(infile)
wd_fp = glob.glob(os.path.join(home, "Data/Wind_Direction/Sentinel1/ERA5_Corrected", "corrected_wdir_10pixels_" + os.path.basename(infile)[:-3] + ".tif"))[0]
wd = rioxarray.open_rasterio(wd_fp, band_as_variable = True)
wd = wd.rename({'band_1': 'wd'})

# Resample the sig0 and incidence angle to match the wind direction since it is coarser 
ds_upsampled = ds.rio.reproject(dst_crs=wd.wd.rio.crs, 
                                #resolution=wd.wd.rio.resolution(), 
                                shape=wd.wd.rio.shape, 
                                transform=wd.wd.rio.transform(), 
                                resampling=Resampling.bilinear)
ds_upsampled = xarray.combine_by_coords([ds_upsampled, wd])

# Calculate phi: in [deg] angle between azimuth and wind direction (= D - AZM)
platform_heading = ds_upsampled.attrs['platform_heading']
platform_heading = np.mod(0 + platform_heading, 360)
look_direction = np.mod(platform_heading + 90, 360)
ds_upsampled['phi'] = np.mod(ds_upsampled.wd - look_direction, 360)

def apply_cmod5n(sigma0_obs, phi, incidence):
    if ((np.isnan(sigma0_obs)==True) | (np.isnan(phi)==True) | (np.isnan(incidence)==True)) :
        return np.nan
    else:
        return cmod5n_inverse( np.array([sigma0_obs]), np.array([phi]), np.array([incidence]), iterations=10)[0]
cmod5n_func = np.vectorize(apply_cmod5n)  
wind_speed = cmod5n_func(ds_upsampled.sig0.to_numpy()[0,:,:], ds_upsampled.phi.to_numpy(), ds_upsampled.inc_angle.to_numpy()[0,:,:])
ds_upsampled['wind_speed'] = (['y', 'x'], wind_speed)
ds_upsampled.wind_speed.rio.write_nodata(np.nan, encoded=True, inplace=True)
ds_upsampled.wind_speed.rio.to_raster(os.path.join(home, "Data/Outputs", "wspd_" + os.path.basename(infile)[:-3] + ".tif"))

ds_upsampled.to_netcdf(os.path.join(home, "Data/Outputs", "wspd_" + os.path.basename(infile)[:-3] + ".nc"))



# Lake Washington
###############################################################################################
# Load in the sig0, inc angle, and wind direction datasets 
from rasterio.enums import Resampling
import xarray

infile = os.path.join(home, "Data/Sentinel1/Processed/S1A_IW_GRDH_1SDV_20230321T015425_20230321T015450_047736_05BBFD_A2DE.nc")
ds = rioxarray.open_rasterio(infile)
wd_fp = glob.glob(os.path.join(home, "Data/Wind_Direction/Sentinel1/ERA5_Corrected", "corrected_wdir_10pixels_" + os.path.basename(infile)[:-3] + ".tif"))[0]
wd = rioxarray.open_rasterio(wd_fp, band_as_variable = True)
wd = wd.rename({'band_1': 'wd'})

# Resample the sig0 and incidence angle to match the wind direction since it is coarser 
ds_upsampled = ds.rio.reproject(dst_crs=wd.wd.rio.crs, 
                                #resolution=wd.wd.rio.resolution(), 
                                shape=wd.wd.rio.shape, 
                                transform=wd.wd.rio.transform(), 
                                resampling=Resampling.bilinear)
ds_upsampled = xarray.combine_by_coords([ds_upsampled, wd])

# Calculate phi: in [deg] angle between azimuth and wind direction (= D - AZM)
platform_heading = ds_upsampled.attrs['platform_heading']
platform_heading = np.mod(0 + platform_heading, 360)
look_direction = np.mod(platform_heading + 90, 360)
ds_upsampled['phi'] = np.mod(ds_upsampled.wd - look_direction, 360)

def apply_cmod5n(sigma0_obs, phi, incidence):
    if ((np.isnan(sigma0_obs)==True) | (np.isnan(phi)==True) | (np.isnan(incidence)==True)) :
        return np.nan
    else:
        return cmod5n_inverse( np.array([sigma0_obs]), np.array([phi]), np.array([incidence]), iterations=10)[0]
cmod5n_func = np.vectorize(apply_cmod5n)  
wind_speed = cmod5n_func(ds_upsampled.sig0.to_numpy()[0,:,:], ds_upsampled.phi.to_numpy(), ds_upsampled.inc_angle.to_numpy()[0,:,:])
ds_upsampled['wind_speed'] = (['y', 'x'], wind_speed)
ds_upsampled.wind_speed.rio.write_nodata(np.nan, encoded=True, inplace=True)
ds_upsampled.wind_speed.rio.to_raster(os.path.join(home, "Data/Outputs", "wspd_" + os.path.basename(infile)[:-3] + ".tif"))

