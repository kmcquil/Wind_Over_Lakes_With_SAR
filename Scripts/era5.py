##########################################################################################
# Katie Mcquillan
# 07/17/2024
# Process the raw ERA5 U and V to wind speed and direction gtiffs
##########################################################################################

# Import modules
import rioxarray
import xarray as xr
import numpy as np
import os
import glob

# Set home directory
home = "C:/Users/kmcquil/Documents/SWOT_WIND"

# List the ERA5 nc files
files = glob.glob(os.path.join(home, "Data/ERA5/Raw/*.nc"))

# Loop through each files
for file in files:
    # Chunks=true must be set otherwise this gets too big locally
    ds = rioxarray.open_rasterio(file, chunks=True)
    # Fix the coords to be -180 to 180 and then resort
    ds = ds.assign_coords(x=(((ds.x + 180) % 360) - 180))
    ds = ds.sortby("x")
    # Opened as integer so scale by 1000
    ds = ds / 1000
    # Set the crs
    ds.rio.write_crs("epsg:4326", inplace=True)

    # Get list of datetimes included in the file
    times = ds.time.to_numpy()
    # Loop through datetimes
    for time in times:
        # Make the datetime nice and check if its already been processed
        time_out = (
            str(time.year)
            + str(time.month).zfill(2)
            + str(time.day).zfill(2)
            + str(time.hour).zfill(2)
            + "0000"
        )
        if (
            os.path.exists(
                os.path.join(home, "Data/ERA5/Processed/wspd/", time_out + ".tif")
            )
            == True
        ) & (
            os.path.exists(
                os.path.join(home, "Data/ERA5/Processed/wdir/", time_out + ".tif")
            )
            == True
        ):
            continue

        # Calculate wind speed and direction from u and v components
        wspd = np.sqrt(np.square(ds["u10"].loc[time]) + np.square(ds["v10"].loc[time]))
        wdir = np.degrees(np.arctan2(ds["u10"].loc[time], ds["v10"].loc[time])) % 360
        # Save to tiffs
        wspd.rio.to_raster(
            os.path.join(home, "Data/ERA5/Processed/wspd/", time_out + ".tif")
        )
        wdir.rio.to_raster(
            os.path.join(home, "Data/ERA5/Processed/wdir/", time_out + ".tif")
        )
    # Make sure to close ds so memory doesn't keep growing
    ds.close()
