##########################################################################################
# Katie Mcquillan
# 05/17/2024
# Calculate wind direction using the LG-Mod method from Rana et al. 2015
##########################################################################################

import os
import itertools
import glob as glob
from datetime import datetime, timedelta, timezone
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.ndimage import convolve
from skimage.measure import label
import rioxarray
import xarray
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
from scipy import interpolate
import scipy
from shapely.geometry import box

home = "C:/Users/kmcquil/Documents/SWOT_WIND/"

#########################################################################################
# Functions to calculate wind direction from LG-Mod (Rana et al. 2015), to calculate
# wind speed using CMOD5.N, and extract all data at buoy locations
#########################################################################################


def correct_180_ambiguity(wdir, sat_wdir):
    """
    Correct 180 degree ambiguity using another source of wind direction
    wdir: "true" wind direction. probably from a reanalysis dataset
    swot_wdir: 180 degree ambiguous swot direction
    return: swot wind direction corrected for 180 degree ambiguity
    """
    if (np.isnan(sat_wdir)) | (np.isnan(wdir)):
        return np.nan

    upper = (wdir + 90) % 360
    lower = (wdir - 90) % 360

    if lower > upper:
        env = list(range(int(lower), 360)) + list(range(0, int(upper) + 1))
        if int(sat_wdir) in env:
            fixed_wdir = sat_wdir
        else:
            fixed_wdir = (sat_wdir + 180) % 360
    if lower < upper:
        env = list(range(int(lower), int(upper) + 1))
        if int(sat_wdir) in env:
            fixed_wdir = sat_wdir
        else:
            fixed_wdir = (sat_wdir + 180) % 360
    return fixed_wdir


def correction_with_era5(satellite, wdir_file, outfile):
    """
    Correct the satellite wind direction estimates for 180 degree ambiguity using ERA5
    wdir_file: raster file of the satellite wind direction estimate
    outfile: filepath to corrected wind direction
    """
    if satellite == "Sentinel1":
        date = datetime.strptime(
            os.path.basename(wdir_file).split("_")[6], "%Y%m%dT%H%M%S"
        )
        date = date.replace(second=0, microsecond=0, minute=0)
    if satellite == "SWOT":
        date = datetime.strptime(
            os.path.basename(wdir_file).split("_")[15][0:11], "%Y%m%dT%H"
        )

    era5_files = glob.glob(os.path.join(home, "Data/ERA5/Processed/wdir/*.tif"))
    era5_dates = [
        datetime.strptime(os.path.basename(i)[0:10], "%Y%m%d%H") for i in era5_files
    ]
    era5_file = era5_files[era5_dates.index(date)]
    reproj_match(era5_file, wdir_file, outfile)
    with rasterio.open(wdir_file) as swot:
        swot_wd = swot.read()
        swot_wd = swot_wd[0, :, :]

        with rasterio.open(outfile) as era5:
            era5_match = era5.read()
            era5_match = era5_match[0, :, :]

        fv = np.vectorize(correct_180_ambiguity)
        fixed_swot_wd = fv(era5_match, swot_wd)
        kwargs = swot.meta
        with rasterio.open(outfile, "w", **kwargs) as dst:
            dst.write_band(1, fixed_swot_wd)


def reproj_match(infile, matchfile, outfile):
    """
    Reproject a file to match the shape and projection of existing raster.
    infile : (string) path to input file to reproject
    match : (string) path to raster with desired shape and projection
    outfile : (string) path to output file tif
    """
    # open input
    with rasterio.open(infile) as src:

        # open input to match
        with rasterio.open(matchfile) as match:
            dst_crs = match.crs
            dst_transform = match.transform
            dst_width = match.width
            dst_height = match.height
            dst_nodata = match.nodata

            # Clip the input file to the match file
            geom = gpd.GeoDataFrame(
                {"geometry": box(*match.bounds)}, index=[0], crs=match.crs
            ).to_crs(src.crs)
            geom = getFeatures(geom)
            src_clip_img, src_clip_transform = mask(
                src, shapes=geom, crop=True, all_touched=True
            )
            src_clip_img[src_clip_img == src.nodata] = np.nan

        # set properties for output
        dst_kwargs = src.meta.copy()
        dst_kwargs.update(
            {
                "crs": dst_crs,
                "transform": dst_transform,
                "width": dst_width,
                "height": dst_height,
                "nodata": dst_nodata,
            }
        )

        # open output
        with rasterio.open(outfile, "w", **dst_kwargs) as dst:
            reproject(
                source=src_clip_img,
                destination=rasterio.band(dst, 1),
                src_transform=src_clip_transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
            )


def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json

    return [json.loads(gdf.to_json())["features"][0]["geometry"]]


def calc_rana_wind_direction(sig0, wf):
    """
    calculate the wind direction using rana method
    sig0: (np array) of sig0
    wf: (np array) of water fractions
    """
    # Sobel kernels
    Dx = (1 / 32) * np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]])
    Dy = Dx.transpose()

    # Convert the Dy kernel to complex matrix
    make_complex = lambda t: complex(0, t)
    vfunc = np.vectorize(make_complex)
    Dy_complex = vfunc(Dy)

    # Convolve sig0 with horizontal and vertical gradients
    G1 = convolve(
        sig0, (Dx + Dy_complex), mode="reflect"
    )  # reflect mode means outer values are reflected at the edge of the input to fill in missing values

    # Discard unusable points
    # 1. The first and last two rows of the image
    # 2. Non-water pixels

    # Create mask for pixels that half a water fraction < 0.9
    discard_mask = G1
    discard_mask = wf >= 0.9

    # Mask the first two columns and rows
    discard_mask[0:2, :] = False
    discard_mask[-2:, :] = False
    discard_mask[:, 0:2] = False
    discard_mask[:, -2:] = False

    # Apply mask to G2
    G1_masked = np.where(discard_mask, G1, np.nan)
    angles_rad = np.angle(G1_masked, deg=False)

    # Ignore 180 degree ambiguity by adding pi radians to all negative angles
    angles_rad[angles_rad < 0] = angles_rad[angles_rad < 0] + math.pi

    # Calculate the mean angle (Rana et al. 2016 equation 1)
    angles_rad = angles_rad[~np.isnan(angles_rad)]
    if len(angles_rad) == 0:
        return [np.nan, np.nan, np.nan]
    angle_2_sin = [math.sin(2 * i) for i in angles_rad]
    angle_2_cos = [math.cos(2 * i) for i in angles_rad]
    mean_angle = 0.5 * np.arctan2(np.mean(angle_2_sin), np.mean(angle_2_cos))
    # Convert the angle to degrees
    mean_angle = mean_angle * (180 / math.pi)
    # Gives degrees from -180 to 180. Transform to 360
    if mean_angle < 0:
        mean_angle = mean_angle + 360
    # X is positive to the right.  Y is positive down.  Therefore, a 45 degree angle is down and to the right.
    # Angles are measured starting from the horizontal, and are positive as you go clockwise (down).
    # So to get the right direction, add 90 degrees!!!!
    mean_angle = (mean_angle + 90) % 360

    # Add another 90 because wind direction is orthogonal to the max gradient
    wind_direction = (mean_angle + 90) % 360

    # Calculate a non dimensional parameter (R roi) which represents a measure of the alignment of the directions inside the ROI (rana et al. 2016 eq 2)
    dir_align = math.sqrt((np.mean(angle_2_cos) ** 2) + (np.mean(angle_2_sin) ** 2))

    # Calculate a confidence interval that can be used to discharge noisy rois (eq 3)
    alpha = 0.05
    ualpha = scipy.stats.norm.ppf(1 - (alpha / 2), 0, 1)
    alpha2roi = np.mean(
        np.cos(np.multiply(np.subtract(angles_rad, np.mean(angles_rad)), 4))
    )
    v = ualpha * math.sqrt((1 - alpha2roi) / (2 * len(angles_rad) * (dir_align**2)))
    if v < 1:
        ME = 0.5 * np.arcsin(v)
    if v >= 1:
        ME = 0.5 * np.arcsin(1)
    ME = ME * (180 / math.pi)
    return [wind_direction, dir_align, ME]


def wind_direction_by_image(
    satellite,
    infile,
    outfile_wd,
    outfile_da,
    outfile_me,
    boundary,
    qc_mask,
    N,
    max_na_frac,
    min_wf_frac,
):
    """
    Apply the local gradient method from Koch 2004 to a whole water body
    infile: filepath for hte .nc file of 100m SWOT raster
    outfile: filepath for the .tif file to write out wind dir
    boundary: goepandas df of water body
    qc_mask: a tif matching the infile with 1 for keep and 0 for don't keep
    N: size of subsets in number of pixels. This will be the resolution of wind direction (i.e. 10 pixels at 100m resolution = 1km x 1km wind direction estimates)
    max_na_frac: the max fraction of NAs in the subset to still estimate wind direction
    min_wf_frac: the min water_fraction in non-NA pixels to still estimate wind direction
    """

    # Open the netcdf
    ds = rioxarray.open_rasterio(infile)
    sig0 = ds.sig0

    # SWOT .nc file requires a couple extra steps of processing
    if satellite == "SWOT":
        # Covnert no data to na
        sig0 = sig0.where(sig0 != sig0.rio.nodata)
        # Filter sig0 observations. Only keep observations that corresond to 0 (good), 1(suspect) and 2 (degraded). 3 = bad
        sig0 = sig0.where(ds.sig0_qual <= 2)
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
    sig0_np = sig0_clipped.to_numpy()[0, :, :]

    # Use the lake boundary as the water mask. 0 = land, 1 = water
    wf_np = sig0_np.copy()
    wf_np = np.where(np.isnan(wf_np), wf_np, 1)
    wf_np = np.where(~np.isnan(wf_np), wf_np, 0)

    # Load the non-wind qc mask and add to the wf_np mask
    if qc_mask != "NA":
        with rasterio.open(qc_mask) as src:
            qc_np = src.read()[0, :, :]
        qc_np = np.where(qc_np == 0, np.nan, qc_np)
        # Update the water fraction mask. Bad quality set to 0
        # sig0_np = np.where(np.isnan(qc_np)==True, np.nan, sig0_np)
        wf_np = np.where(np.isnan(qc_np) == True, 0, wf_np)

    # define the subsets based on the shape of the array
    # row_start = [*range(0, sig0_np.shape[0], N)]
    # row_start = row_start[0:len(row_start)-1]
    # row_end = [x+N for x in row_start]

    # col_start = [*range(0, sig0_np.shape[1], N)]
    # col_start = col_start[0:len(col_start)-1]
    # col_end = [x+N for x in col_start]

    # Define one subset based on size of these new small subsets of the s1 images
    # rows: 200ish Center row is around 100 -- 50-150
    # columns: 140ish center column is around 70 -- 20 - 120
    # 50 - 150
    row_start = [50]
    row_end = [150]
    col_start = [20]
    col_end = [120]

    # Copy the array shape and fill in with the wind direction
    wind_direction_np = np.empty((sig0_np.shape[0], sig0_np.shape[1]))
    wind_direction_np[:] = np.nan
    dir_align_np = wind_direction_np.copy()
    me_np = wind_direction_np.copy()

    # For loop to traverse the sig0 raster and extract wind direction at subsets
    for i in range(0, len(row_start)):
        for j in range(0, len(col_start)):
            # Use the row/col start/end to select the subset
            sig0_sub = sig0_np[row_start[i] : row_end[i], col_start[j] : col_end[j]]
            wf_sub = wf_np[row_start[i] : row_end[i], col_start[j] : col_end[j]]

            # Fill the wf with 0.
            wf_sub_filled = np.where(np.isnan(wf_sub) == True, 0, wf_sub)

            # Calculate the fraction of subset that is NA and the fraction of the subset that is covered by water
            # Only keep subsets with 0% NA and 100% water
            na_frac = np.sum(np.isnan(sig0_sub)) / (
                sig0_sub.shape[0] * sig0_sub.shape[1]
            )
            wf_frac = np.nanmean(wf_sub_filled)
            if (na_frac > max_na_frac) | (wf_frac < min_wf_frac):
                continue

            # If there are Na values, fill using cubic interpolation. Necessary bc convolution won't work w NAs
            x = np.arange(0, sig0_sub.shape[1])
            y = np.arange(0, sig0_sub.shape[0])
            # mask invalid values
            sig0_sub = np.ma.masked_invalid(sig0_sub)
            xx, yy = np.meshgrid(x, y)
            # get only the valid values
            x1 = xx[~sig0_sub.mask]
            y1 = yy[~sig0_sub.mask]
            newarr = sig0_sub[~sig0_sub.mask]
            sig0_sub_filled = interpolate.griddata(
                (x1, y1),
                newarr.ravel(),
                (xx, yy),
                method="cubic",
                fill_value=np.mean(newarr),
            )

            # Calculate the wind direction
            wind_direction = calc_rana_wind_direction(sig0_sub_filled, wf_sub_filled)
            # print("i: " + str(i) + " j: " + str(j))
            # print(wind_direction[2])
            # Put that wind direction in all cells belonging to this subset on wind direction np array
            wind_direction_np[row_start[i] : row_end[i], col_start[j] : col_end[j]] = (
                wind_direction[0]
            )
            dir_align_np[row_start[i] : row_end[i], col_start[j] : col_end[j]] = (
                wind_direction[1]
            )
            me_np[row_start[i] : row_end[i], col_start[j] : col_end[j]] = (
                wind_direction[2]
            )

    # Combine the sigma0 and wind direction 2d array into one xarray dataset that retains the coords/spatial referencing
    sig0_clipped["wd"] = (["y", "x"], wind_direction_np)
    wd = sig0_clipped.reset_coords("wd").wd
    sig0_clipped = sig0_clipped.drop_vars("wd")
    final = xarray.combine_by_coords([sig0_clipped, wd])
    final.wd.rio.write_nodata(np.nan, encoded=True, inplace=True)
    final.wd.rio.to_raster(outfile_wd)

    # same for directional alignment
    sig0_clipped["da"] = (["y", "x"], dir_align_np)
    da = sig0_clipped.reset_coords("da").da
    sig0_clipped = sig0_clipped.drop_vars("da")
    final = xarray.combine_by_coords([sig0_clipped, da])
    final.da.rio.write_nodata(np.nan, encoded=True, inplace=True)
    final.da.rio.to_raster(outfile_da)

    # same for marginal error
    sig0_clipped["me"] = (["y", "x"], me_np)
    me = sig0_clipped.reset_coords("me").me
    sig0_clipped = sig0_clipped.drop_vars("me")
    final = xarray.combine_by_coords([sig0_clipped, me])
    final.me.rio.write_nodata(np.nan, encoded=True, inplace=True)
    final.me.rio.to_raster(outfile_me)

    # Correct the 180 degree ambiguity
    # Add the ERA5 corrected folder if it doesn't already exist
    corrected_path = os.path.join(os.path.dirname(outfile_wd), "ERA5_Corrected")
    isExist = os.path.exists(corrected_path)
    if not isExist:
        os.makedirs(corrected_path)
    correction_with_era5(
        satellite,
        outfile_wd,
        os.path.join(corrected_path, "corrected_" + os.path.basename(outfile_wd)),
    )

    return


def extract_buoy_wdir(buoy_id, wdir_id):
    """
    Extract wdir from the satellite, era5, and buoy
    buoy_id: string of the buoy id
    wdir_id: string of the prefix to the wdir files - "widr_10pixels" or "wdir_20pixels" ect.
    return df of wdir
    """
    # Open the buoy df
    buoy_df = pd.read_csv(os.path.join(home, "Data/Buoy/Processed", buoy_id + ".csv"))
    buoy_df["datetime"] = pd.to_datetime(
        buoy_df["datetime"], utc=True, format="ISO8601"
    )

    # Subset to the rows that correspond to the buoy
    subset_buoy_sat_int = buoy_sat_int[buoy_sat_int["buoy_id"] == buoy_id]
    subset_buoy_sf = buoy_sf[buoy_sf["id"] == buoy_id]

    # Loop through each row and extract the wind direction at buoy location and add to list
    row = []
    for i in range(0, subset_buoy_sat_int.shape[0]):
        satellite = subset_buoy_sat_int["satellite"].iloc[i]
        image_id = os.path.basename(subset_buoy_sat_int["image_id"].iloc[i])[:-3]
        if satellite == "Sentinel1":
            overpass_datetime = datetime.strptime(
                image_id.split("_")[5], "%Y%m%dT%H%M%S"
            )
        else:
            overpass_datetime = datetime.strptime(
                image_id.split("_")[13], "%Y%m%dT%H%M%S"
            )
        overpass_datetime = overpass_datetime.replace(tzinfo=timezone.utc)

        # If the wind direction raster of this file doesn't exist, it's because there wasn't a space/time matchup. skip to the next
        wdir_fp = os.path.join(
            home,
            "Data/Wind_Direction/Rana",
            satellite + "_10m",
            buoy_id,
            "WindDirection",
            wdir_id + image_id + ".tif",
        )
        if os.path.isfile(wdir_fp) == False:
            continue

        # Extract from satellite without ERA5 correction
        with rasterio.open(wdir_fp) as src:
            buoy_reproj = subset_buoy_sf.to_crs(src.crs)
            coords = buoy_reproj.get_coordinates().values
            wdir = [x[0] for x in src.sample(coords)][0]

        # Extract from satellite with ERA5 correction
        wdir_fp = os.path.join(
            home,
            "Data/Wind_Direction/Rana",
            satellite + "_10m",
            buoy_id,
            "WindDirection/ERA5_Corrected",
            "corrected_" + wdir_id + image_id + ".tif",
        )
        with rasterio.open(wdir_fp) as src:
            buoy_reproj = subset_buoy_sf.to_crs(src.crs)
            coords = buoy_reproj.get_coordinates().values
            wdir_corrected = [x[0] for x in src.sample(coords)][0]

        # Extract the directional aligment unitless measure
        da_fp = os.path.join(
            home,
            "Data/Wind_Direction/Rana",
            satellite + "_10m",
            buoy_id,
            "DirectionAlignment",
            "da_100pixels_" + image_id + ".tif",
        )
        with rasterio.open(da_fp) as src:
            buoy_reproj = subset_buoy_sf.to_crs(src.crs)
            coords = buoy_reproj.get_coordinates().values
            da = [x[0] for x in src.sample(coords)][0]

        # Extract the ME (marginal error)
        me_fp = os.path.join(
            home,
            "Data/Wind_Direction/Rana",
            satellite + "_10m",
            buoy_id,
            "MarginalError",
            "me_100pixels_" + image_id + ".tif",
        )
        with rasterio.open(me_fp) as src:
            buoy_reproj = subset_buoy_sf.to_crs(src.crs)
            coords = buoy_reproj.get_coordinates().values
            me = [x[0] for x in src.sample(coords)][0]

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
        diffs = (buoy_df["datetime"] - overpass_datetime).tolist()
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
            buoy_datetime = subset_buoy_df["datetime"].to_pydatetime()
            diff = buoy_datetime - overpass_datetime
            wdir_buoy = subset_buoy_df["wdir"]
            wspd_buoy = subset_buoy_df["wspd"]

        # Put into pd df
        row.append(
            pd.DataFrame(
                {
                    "buoy_id": [buoy_id],
                    "wdir_id": [wdir_id],
                    "satellite": [satellite],
                    "image_id": [image_id],
                    "overpass_datetime": [overpass_datetime],
                    "wdir": [wdir],
                    "wdir_corrected": [wdir_corrected],
                    "da": [da],
                    "me": [me],
                    "wdir_era5": [wdir_era5],
                    "wspd_era5": [wspd_era5],
                    "wdir_buoy": [wdir_buoy],
                    "wspd_buoy": [wspd_buoy],
                    "buoy_datetime": [buoy_datetime],
                    "time_diff": [diff],
                }
            )
        )

    if len(row) == 0:
        wdir_df = pd.DataFrame(
            {
                "buoy_id": [buoy_id],
                "wdir_id": [np.nan],
                "satellite": [satellite],
                "image_id": [np.nan],
                "overpass_datetime": [np.nan],
                "wdir": [np.nan],
                "wdir_corrected": [np.nan],
                "da": [np.nan],
                "me": [np.nan],
                "wdir_era5": [np.nan],
                "wspd_era5": [np.nan],
                "wdir_buoy": [np.nan],
                "wspd_buoy": [np.nan],
                "buoy_datetime": [np.nan],
                "time_diff": [np.nan],
            }
        )
        return wdir_df
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
    buoy_df["datetime"] = pd.to_datetime(
        buoy_df["datetime"], utc=True, format="ISO8601"
    )

    # Subset to the rows that correspond to the buoy
    subset_buoy_sat_int = buoy_sat_int[buoy_sat_int["buoy_id"] == buoy_id]
    subset_buoy_sf = buoy_sf[buoy_sf["id"] == buoy_id]

    # Loop through each row and extract the wind direction at buoy location and add to list
    row = []
    for i in range(0, subset_buoy_sat_int.shape[0]):
        satellite = subset_buoy_sat_int["satellite"].iloc[i]
        image_id = os.path.basename(subset_buoy_sat_int["image_id"].iloc[i])[:-3]
        if satellite == "Sentinel1":
            overpass_datetime = datetime.strptime(
                image_id.split("_")[5], "%Y%m%dT%H%M%S"
            )
        else:
            overpass_datetime = datetime.strptime(
                image_id.split("_")[13], "%Y%m%dT%H%M%S"
            )
        overpass_datetime = overpass_datetime.replace(tzinfo=timezone.utc)

        # Extract sigma0, incidence angle, platform heading
        if satellite == "Sentinel1":
            fp = os.path.join(home, "Data", "Sentinel1", "Processed", image_id + ".nc")
            with rioxarray.open_rasterio(fp) as src:
                buoy_reproj = subset_buoy_sf.to_crs(src.rio.crs)
                coords = buoy_reproj.get_coordinates().values
                sig0 = src.sig0.sel(
                    x=coords[0][0], y=coords[0][1], method="nearest"
                ).to_numpy()[0]
                inc_angle = src.inc_angle.sel(
                    x=coords[0][0], y=coords[0][1], method="nearest"
                ).to_numpy()[0]
                platform_heading = src.platform_heading

        if "SWOT" in satellite:
            fp = os.path.join(
                home, "Data", "SWOT_L2_HR_Raster", satellite, image_id + ".nc"
            )
            with rioxarray.open_rasterio(fp) as src:
                sig0 = src.sig0
                sig0 = sig0.where(sig0 != sig0.rio.nodata)
                crs_wkt = sig0.crs.attrs["crs_wkt"]
                crs_wkt_split = crs_wkt.split(",")
                epsg = crs_wkt_split[len(crs_wkt_split) - 1].split('"')[1]
                sig0.rio.write_crs("epsg:" + epsg, inplace=True)
                buoy_reproj = subset_buoy_sf.to_crs(sig0.rio.crs)
                coords = buoy_reproj.get_coordinates().values
                sig0 = sig0.sel(
                    x=coords[0][0], y=coords[0][1], method="nearest"
                ).to_numpy()[0]
                inc_angle = src.inc.sel(
                    x=coords[0][0], y=coords[0][1], method="nearest"
                ).to_numpy()[0]
                platform_heading = (
                    np.nan
                )  # don't need this because I'm not calculating wind speeed from swot

        # Put into pd df
        row.append(
            pd.DataFrame(
                {
                    "buoy_id": [buoy_id],
                    "satellite": [satellite],
                    "image_id": [image_id],
                    "overpass_datetime": [overpass_datetime],
                    "sig0": [sig0],
                    "inc_angle": [inc_angle],
                    "platform_heading": [platform_heading],
                }
            )
        )
    sig0_df = pd.concat(row)
    return sig0_df


def cmod5n_inverse(sigma0_obs, phi, incidence, iterations=10):
    """
    https://github.com/nansencenter/openwind/blob/master/openwind/cmod5n.py
    This function iterates the forward CMOD5N function
    until agreement with input (observed) sigma0 values
    Inputs (All inputs must be Numpy arrays of equal sizes):
    sigma0_obs     Normalized Radar Cross Section [linear units]
    phi   in [deg] angle between azimuth and wind direction (= D - AZM)
    incidence in [deg] incidence angle
    iterations: number of iterations to run
    output:Wind speed, 10 m, neutral stratification
    """
    from numpy import ones, array

    # First guess wind speed
    V = array([10.0]) * ones(sigma0_obs.shape)
    step = 10.0

    # Iterating until error is smaller than threshold
    for iterno in range(1, iterations):
        print(iterno)
        sigma0_calc = cmod5n_forward(V, phi, incidence)
        ind = sigma0_calc - sigma0_obs > 0
        V = V + step
        V[ind] = V[ind] - 2 * step
        step = step / 2

    # mdict={'s0obs':sigma0_obs,'s0calc':sigma0_calc}
    # from scipy.io import savemat
    # savemat('s0test',mdict)

    return V


def cmod5n_forward(v, phi, theta):
    """
    https://github.com/nansencenter/openwind/blob/master/openwind/cmod5n.py
    inputs (All inputs must be Numpy arrays of equal sizes):
    v     in [m/s] wind velocity (always >= 0)
    phi   in [deg] angle between azimuth and wind direction (= D - AZM)
    theta in [deg] incidence angle
    output:
    CMOD5_N NORMALIZED BACKSCATTER (LINEAR)
    """

    from numpy import cos, exp, tanh, array

    DTOR = 57.29577951
    THETM = 40.0
    THETHR = 25.0
    ZPOW = 1.6

    # NB: 0 added as first element below, to avoid switching from 1-indexing to 0-indexing
    C = [
        0,
        -0.6878,
        -0.7957,
        0.3380,
        -0.1728,
        0.0000,
        0.0040,
        0.1103,
        0.0159,
        6.7329,
        2.7713,
        -2.2885,
        0.4971,
        -0.7250,
        0.0450,
        0.0066,
        0.3222,
        0.0120,
        22.7000,
        2.0813,
        3.0000,
        8.3659,
        -3.3428,
        1.3236,
        6.2437,
        2.3893,
        0.3249,
        4.1590,
        1.6930,
    ]
    Y0 = C[19]
    PN = C[20]
    A = C[19] - (C[19] - 1) / C[20]

    B = 1.0 / (C[20] * (C[19] - 1.0) ** (3 - 1))

    #  !  ANGLES
    FI = phi / DTOR
    CSFI = cos(FI)
    CS2FI = 2.00 * CSFI * CSFI - 1.00

    X = (theta - THETM) / THETHR
    XX = X * X

    #  ! B0: FUNCTION OF WIND SPEED AND INCIDENCE ANGLE
    A0 = C[1] + C[2] * X + C[3] * XX + C[4] * X * XX
    A1 = C[5] + C[6] * X
    A2 = C[7] + C[8] * X

    GAM = C[9] + C[10] * X + C[11] * XX
    S0 = C[12] + C[13] * X

    # V is missing! Using V=v as substitute, this is apparently correct
    V = v
    S = A2 * V
    S_vec = S.copy()
    # SlS0 = [S_vec<S0]
    SlS0 = S_vec < S0
    S_vec[SlS0] = S0[SlS0]
    A3 = 1.0 / (1.0 + exp(-S_vec))
    SlS0 = S < S0
    A3[SlS0] = A3[SlS0] * (S[SlS0] / S0[SlS0]) ** (S0[SlS0] * (1.0 - A3[SlS0]))
    # A3=A3*(S/S0)**( S0*(1.- A3))
    B0 = (A3**GAM) * 10.0 ** (A0 + A1 * V)

    #  !  B1: FUNCTION OF WIND SPEED AND INCIDENCE ANGLE
    B1 = C[15] * V * (0.5 + X - tanh(4.0 * (X + C[16] + C[17] * V)))
    B1 = C[14] * (1.0 + X) - B1
    B1 = B1 / (exp(0.34 * (V - C[18])) + 1.0)

    #  !  B2: FUNCTION OF WIND SPEED AND INCIDENCE ANGLE
    V0 = C[21] + C[22] * X + C[23] * XX
    D1 = C[24] + C[25] * X + C[26] * XX
    D2 = C[27] + C[28] * X

    V2 = V / V0 + 1.0
    V2ltY0 = V2 < Y0
    V2[V2ltY0] = A + B * (V2[V2ltY0] - 1.0) ** PN
    B2 = (-D1 + D2 * V2) * exp(-V2)

    #  !  CMOD5_N: COMBINE THE THREE FOURIER TERMS
    CMOD5_N = B0 * (1.0 + B1 * CSFI + B2 * CS2FI) ** ZPOW
    return CMOD5_N


#########################################################################################
# Calculate wind direction at 1km resolution for S1 using 10-m imagery
#########################################################################################

# Create list of buoys
buoy_sf = gpd.read_file(os.path.join(home, "Data/Buoy/buoy_info.shp"))
buoy_ids = buoy_sf["id"].tolist()

# Open geopandas shapefile of PLD boundaries
boundary = gpd.read_file(os.path.join(home, "Data/Buoy/pld_with_buoys.shp"))

# Number of pixels to include in x and y direction in each ROI. 100 pixels x 10 m = 1000 m = 1km resolution
N = 100

# Loop through each buoy and calculate the wind direction
for buoy in buoy_ids:
    in_dir = os.path.join(
        "C:/Users/kmcquil/Documents/SWOT_WIND/Data/Sentinel1/Processed_10m", buoy
    )
    out_dir = os.path.join(
        "C:/Users/kmcquil/Documents/SWOT_WIND/Data/Wind_Direction/Rana/Sentinel1_10m/",
        buoy,
    )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        os.makedirs(os.path.join(out_dir, "WindDirection"))
        os.makedirs(os.path.join(out_dir, "MarginalError"))
        os.makedirs(os.path.join(out_dir, "DirectionAlignment"))

    files = glob.glob(os.path.join(in_dir, "*.nc"))
    for infile in files:
        print(infile)
        outfile_wd = os.path.join(
            out_dir,
            "WindDirection",
            "wdir_" + str(N) + "pixels_" + os.path.basename(infile[:-3]) + ".tif",
        )
        outfile_da = os.path.join(
            out_dir,
            "DirectionAlignment",
            "da_" + str(N) + "pixels_" + os.path.basename(infile[:-3]) + ".tif",
        )
        outfile_me = os.path.join(
            out_dir,
            "MarginalError",
            "me_" + str(N) + "pixels_" + os.path.basename(infile[:-3]) + ".tif",
        )
        if os.path.exists(outfile_wd):
            continue
        wind_direction_by_image(
            "Sentinel1",
            infile,
            outfile_wd,
            outfile_da,
            outfile_me,
            boundary,
            "NA",
            N,
            0.6,
            0.4,
        )


#########################################################################################
# Extract the wind and sigma0 to one df
#########################################################################################
buoy_sat_int = pd.read_csv(
    os.path.join(home, "Data", "buoy_satellite_intersection.csv")
)

# Open geopandas shapefile of buoy information
buoy_sf = gpd.read_file(os.path.join(home, "Data/Buoy/buoy_info.shp"))

# Extract sigma0 and wind direction at buoys
# List unique buoys and extractsigma0 and wind direction
buoy_ids = buoy_sf["id"].tolist()
# sig0_list = list(map(lambda p: extract_buoy_sigma0(p), buoy_ids))
# sig0_df = pd.concat(sig0_list)
# sig0_df.to_csv(os.path.join(home, "Data/Wind_Direction/sigma0_df.csv")) # already done
sig0_df = pd.read_csv(os.path.join(home, "Data/Wind_Direction/sigma0_df.csv"))

# Wind direction at 1km
wdir_list = list(map(lambda p: extract_buoy_wdir(p, "wdir_100pixels_"), buoy_ids))
wind_df = pd.concat(wdir_list)
wind_df.to_csv(os.path.join(home, "Data/Wind_Direction/wdir_rana_100pixels_df.csv"))
wind_df = pd.merge(
    wind_df,
    sig0_df[
        ["buoy_id", "satellite", "image_id", "sig0", "inc_angle", "platform_heading"]
    ],
    on=["buoy_id", "satellite", "image_id"],
    how="left",
)

# Add in the buoy height
wind_df = pd.merge(
    wind_df, buoy_sf[["id", "height"]], left_on="buoy_id", right_on="id", how="left"
)
wind_df = wind_df.drop("id", axis=1)

# Calculate wind direction relative to azimuth - this is just for S1 since that's the only one I will calc wind speed for
# S1 is always right looking
look_direction = np.mod(wind_df["platform_heading"] + 90, 360)
wind_df["wdir_wrt_azimuth_sat"] = np.mod(
    wind_df["wdir_corrected"] - look_direction, 360
)
wind_df["wdir_wrt_azimuth_era5"] = np.mod(wind_df["wdir_era5"] - look_direction, 360)
wind_df["wdir_wrt_azimuth_buoy"] = np.mod(wind_df["wdir_buoy"] - look_direction, 360)


# Convert buoy wind speed to 10 m height
# use method from this paper: https://www.mdpi.com/2076-3263/13/12/361
# U10 = Ua * ln(Za / Z0) / ln(H10 / Z0)
# U10 is wind speed at 10m, Ua is wind speed a buoy height
# Z0 is constant at 1.52x10-4
# Za is buoy height, H10 = 10
def wspd10(buoy_wspd, buoy_height):
    Z0 = 1.52 * (10**-4)
    return buoy_wspd * math.log(buoy_height / Z0) / math.log(10 / Z0)


wind_df["buoy_wspd_10m"] = wind_df.apply(
    lambda x: wspd10(x.wspd_buoy, x.height), axis=1
)
wind_df = wind_df.reset_index()


def cmod5n(sigma0_obs, phi, incidence):
    if (
        (np.isnan(sigma0_obs) == True)
        | (np.isnan(phi) == True)
        | (np.isnan(incidence) == True)
    ):
        return np.nan
    else:
        return cmod5n_inverse(
            np.array([sigma0_obs]),
            np.array([phi]),
            np.array([incidence]),
            iterations=10,
        )[0]


# Calculate wind speed
wind_df["wspd_sat_cmod5n"] = wind_df.apply(
    lambda x: cmod5n(x.sig0, x.wdir_wrt_azimuth_sat, x.inc_angle), axis=1
)
wind_df["wspd_era5_cmod5n"] = wind_df.apply(
    lambda x: cmod5n(x.sig0, x.wdir_wrt_azimuth_era5, x.inc_angle), axis=1
)
wind_df["wspd_buoy_cmod5n"] = wind_df.apply(
    lambda x: cmod5n(x.sig0, x.wdir_wrt_azimuth_buoy, x.inc_angle), axis=1
)
wind_df.to_csv(os.path.join(home, "Data/Wind_Direction/wdir_rana_100pixels_df.csv"))


#########################################################################################
# Extract the wind and sigma0 to one df
#########################################################################################


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
        # Bias. If the predicted is within + 180 from the observed, it is positive. If the predicted is within - 180 from the observed, it is negative
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
        # Bias. If the predicted is within + 180 from the observed, it is positive. If the predicted is within - 180 from the observed, it is negative
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
    sub = df.dropna(subset=["wspd_sat_cmod5n", "wspd_buoy"])
    if sub.shape[0] < 3:
        row3 = pd.DataFrame(
            [["WSPD-CMOD5-SAR", sub.shape[0], np.nan, np.nan, np.nan]],
            columns=["ID", "N", "Bias", "MAE", "RMSE"],
        )
    else:
        obs = sub["wspd_buoy"].tolist()
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
        obs = sub["wspd_buoy"].tolist()
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
        obs = sub["wspd_buoy"].tolist()
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


wind_df = pd.read_csv(
    os.path.join(home, "Data/Wind_Direction/wdir_rana_100pixels_df.csv")
)
wind_df["time_diff"] = pd.to_timedelta(wind_df["time_diff"])
wind_df = wind_df.dropna(subset=["wdir_corrected", "wdir_buoy"])
wind_df = wind_df[abs(wind_df["time_diff"]) < timedelta(hours=1)]
wind_df = wind_df[wind_df["wdir_id"] == "wdir_100pixels_"]
wind_df = wind_df[wind_df["satellite"] != "SWOT_L2_HR_Raster_1.1"]
wind_df = wind_df[wind_df["buoy_id"] != "buchillonfieldstation"]

wind_streak_check = pd.read_csv(
    os.path.join(home, "Data/Wind_Streaks/wind_streak_check.csv")
)
wind_df = wind_df.merge(
    wind_streak_check[["buoy_id", "image_id", "buoy_image_id", "wind_streak"]],
    on=["buoy_id", "image_id"],
    how="left",
)

# Create df with just entries with wind streaks
wind_df_ws = wind_df[wind_df["wind_streak"] == 1]

# Create df with no wind streaks
wind_df_nows = wind_df[wind_df["wind_streak"] == 0]

perf_overall = []
satellites = ["Sentinel1"]
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
overall_table = perf_overall[
    ((perf_overall["ID"] == "WDIR-LG-180") | (perf_overall["ID"] == "WDIR-ERA5-180"))
]
overall_table = overall_table[overall_table["Wind_Streak"] != "No Wind Streaks"]
overall_table = overall_table.drop(["Resolution", "Bias", "RMSE"], axis=1)
overall_table = (
    overall_table.pivot(
        index=["Satellite", "Wind_Streak", "ME Limit", "N"], columns="ID", values="MAE"
    )
    .round(2)
    .reset_index()
)
overall_table = overall_table.sort_values(
    ["Satellite", "Wind_Streak", "ME Limit"], ascending=[True, True, False]
)
overall_table.to_csv(os.path.join(home, "Data/Outputs/s1_100pixels_wdir_perf.csv"))


perf_buoys = []
limits = [360, 40, 30, 20]
buoys = set(wind_df["buoy_id"])
satellites = ["Sentinel1"]
combos = list(itertools.product(*[buoys, satellites, limits]))
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

perf_buoys = pd.concat(perf_buoys)

perf_buoys[perf_buoys["ID"] == "WDIR-LG-180"].groupby(
    ["Wind_Streak", "Satellite", "ME_Limit"]
)["MAE"].median().round(3)
