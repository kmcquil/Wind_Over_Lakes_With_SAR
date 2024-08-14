##########################################################################################
# Katie Mcquillan
# 04/19/2024
# Calculate wind direction 
##########################################################################################

import os
import glob as glob
import numpy as np
import pandas as pd 
import geopandas as gpd
import math
import cmath
import cv2
from scipy.ndimage import convolve
from sklearn.linear_model import LinearRegression
from skimage.measure import label
import rioxarray
import xarray 
import rasterio
from rasterio.enums import Resampling
from shapely.geometry import box
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling, calculate_default_transform
from datetime import datetime, timedelta
from scipy import interpolate


home = "C:/Users/kmcquil/Documents/SWOT_WIND/"

#########################################################################################
# Functions for calculating wind direction and QC
#########################################################################################

def image_reduction(M):
    """
    Smooth, resize to half dimension, and smooth again according to Koch 2004
    M: numpy array 
    return: smoothed and resized array 
    """
    # Smoothing kernel
    B2 = (1/16)*np.array([[1,2,1], [2,4,2], [1,2,1]])

    # Smooth using the smoothing kernel 
    M_B4 = convolve(M, B2, mode='reflect')
    M_B4 = convolve(M_B4, B2, mode='reflect')

    # Reduce size by half using bicubic interpolation
    # Complex numbers can't be interpolated without breaking into real and imag parts 
    if np.sum(np.iscomplex(M_B4))>0:
        M_B4_real = np.real(M_B4)
        M_B4_imag = np.imag(M_B4)

        real_bicubic = cv2.resize(M_B4_real, (int(M_B4_real.shape[1]/2), int(M_B4_real.shape[0]/2)), interpolation = cv2.INTER_CUBIC)
        imag_bicubic = cv2.resize(M_B4_imag, (int(M_B4_imag.shape[1]/2), int(M_B4_imag.shape[0]/2)), interpolation = cv2.INTER_CUBIC)
        M_half = real_bicubic + 1j*imag_bicubic
    else:
        #M_half = cv2.resize(M_B4, (int(M_B4.shape[1]/2), int(M_B4.shape[0]/2)), interpolation = cv2.INTER_CUBIC)
        M_half = cv2.resize(M_B4, (int(M_B4.shape[1]/2), int(M_B4.shape[0]/2)), interpolation = cv2.INTER_LINEAR)

    # Smooth again
    M_B2 = convolve(M_half, B2, mode='reflect')
    return M_B2



def quality_parameters(sig0):
    """
    Calculate quality parameters on sigma0 from Koch 2004
    sig0: numpy array of sigma0 
    return all quality parameters as numpy arrays 
    """
    B2 = (1/16)*np.array([[1,2,1], [2,4,2], [1,2,1]])
    B22 = (1/16) * np.array([[1,0,2,0,1], [0,0,0,0,0], [2,0,4,0,2], [0,0,0,0,0], [1,0,2,0,1]])

    # Sobel kernels
    Dx = (1/32) * np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]])
    Dy = Dx.transpose()

    # Convert the Dy kernel to complex matrix
    # Dy_complex = Dy.astype(complex)
    make_complex = lambda t: complex(0, t)
    vfunc = np.vectorize(make_complex)
    Dy_complex = vfunc(Dy)

    # Convolve sub image with horizontal and vertical gradients 
    G1 = convolve(sig0, (Dx+Dy_complex), mode='reflect') # reflect mode means outer values are reflected at the edge of the input to fill in missing values 

    # Square and smooth/resize/smooth the gradient image = B2 S|2 B4
    G2 = image_reduction(G1**2)

    # Calculate QC scalar c 
    G3 = image_reduction(np.abs(G1**2))
    c = np.abs(G2)/G3

    # First parameter 
    A1 = image_reduction(sig0)
    J = convolve(A1, B2, mode='reflect')
    J = convolve(J, B2, mode='reflect')
    J = convolve(J, B22, mode='reflect')
    J = convolve(J, B22, mode='reflect')
    J1 = convolve(A1**2, B2, mode='reflect')
    J1 = convolve(J1, B2, mode='reflect')
    J1 = convolve(J1, B22, mode='reflect')
    J1 = convolve(J1, B22, mode='reflect')
    J2 = np.sqrt(J1-J**2)
    P1 = J2/J
    def pw_p1(i):
        if np.isnan(i): return float(np.nan)
        if i > 0.055: return float(0)
        elif i < 0.035: return float(1)
        else:
            reg = LinearRegression().fit(np.array([0.055, 0.035]).reshape(-1,1), np.array([0, 1]).reshape(-1,1))
            return reg.predict(np.array([i]).reshape(1, -1)).tolist()[0][0]
    F1_func = np.vectorize(pw_p1)
    F1 = F1_func(P1)

    # Second parameter 
    A1_half = image_reduction(A1)
    hpf_img = cv2.resize(A1_half, (int(A1.shape[1]), int(A1.shape[0])), interpolation = cv2.INTER_CUBIC)
    IA1 = np.identity(hpf_img.shape[0]) * A1
    K = IA1 - hpf_img
    P2 = (K**2)/(J**2)
    def pw_p2(i):
        if np.isnan(i): return float(np.nan)
        if i > 0.0006: return float(0)
        elif i < 0.0004: return float(1)
        else:
            reg = LinearRegression().fit(np.array([0.0006, 0.0004]).reshape(-1,1), np.array([0, 1]).reshape(-1,1))
            return reg.predict(np.array([i]).reshape(1, -1)).tolist()[0][0]
    F2_func = np.vectorize(pw_p2)
    F2 = F2_func(P2)

    # Third parameter 
    G4 = convolve(G3, B2, mode='reflect')
    G4 = convolve(G4, B2, mode='reflect')
    G4 = convolve(G4, B22, mode='reflect')
    G4 = convolve(G4, B22, mode='reflect')
    P3 = G3/G4
    def pw_p3(i):
        if np.isnan(i): return float(np.nan)
        if i > 1.6: return float(0)
        elif i < 0.53: return float(1)
        else:
            reg = LinearRegression().fit(np.array([1.6, 0.53]).reshape(-1,1), np.array([0, 1]).reshape(-1,1))
            return reg.predict(np.array([i]).reshape(1, -1)).tolist()[0][0]
    F3_func = np.vectorize(pw_p3)
    F3 = F3_func(P3)

    # Fourth parameter 
    P4 = np.sqrt(c)
    def pw_p4(i):
        if np.isnan(i): return float(np.nan)
        if i > 0.63: return float(0)
        elif i < 0.53: return float(1)
        else:
            reg = LinearRegression().fit(np.array([0.63, 0.53]).reshape(-1,1), np.array([0, 1]).reshape(-1,1))
            return reg.predict(np.array([i]).reshape(1, -1)).tolist()[0][0]
    F4_func = np.vectorize(pw_p4)
    F4 = F4_func(P4)

    F = np.sqrt((F1**2 + F2**2 + F3**2 + F4**2)*(1/4))
    F = cv2.resize(F, (int(sig0.shape[1]), int(sig0.shape[1])), interpolation=cv2.INTER_NEAREST)

    return F


def con_points(f_usable, f, min_pix):
    """ 
    Find connected pixels and switch if less than 1km2 from koch 2004
    f_usable: numpy matrix based on F of usable points. Usable points = 1. 
    f: numpy matrix of F that has not been thresholded 
    min_pix: minimum number of pixels for a patch to remain as is
    return mask of usable points
    """
    # Connect usable points (=1). If connected area is less than 1km2, it is considered unusable and is switched 
    labels, num = label(label_image=f_usable, background=0, return_num=True)
    for i in range(1, num + 1):
        num_pix = len(labels[labels == i])
        if num_pix < min_pix:
            # Create a mask of pixels 
            lab_mask = np.where(labels == i, True, False)
            f_usable[lab_mask] = 0

    # Connect unusable points (=0). If connected area is less than 1km2, it is considered usable and is switched
    labels, num = label(label_image=f_usable, background=1, return_num=True)
    for i in range(1, num + 1):
        num_pix = len(labels[labels == i])
        if num_pix < min_pix:
            # Create a mask of pixels 
            lab_mask = np.where(labels == i, True, False)
            f_usable[lab_mask] = 1 
        else:
        # For connected unusable points, replace with the average 
            lab_mask = np.where(labels == i, True, False)
            avg_connected = np.nanmean(f[lab_mask])
            f[lab_mask] = avg_connected
        
    return f_usable



def connecting_quality_parameters(subset100, threshold, min_area):
    """
    Calculate F, the qc statistic from Koch 2004, and use it to create a mask of usable points 
    subset100: a numpy array of sigma0 at 100m resolution 
    threshold: minimum for F to be considered usable 
    min_area: minimum area for a patch to remain as is 
    return mask of usable points
    """
    # Calculate F for 100 - 800 m resolution 
    # This must be calculated in a square matrix because it uses an identity matrix 
    F100 = quality_parameters(subset100)
    pix_100m = min_area/(100*100)
    F100_usable = np.where(F100 >= threshold, 1, 0)
    F100_connected = con_points(F100_usable, F100, min_pix=pix_100m)
    return F100_connected



def calc_wind_direction(sig0, wf):
    """
    Calculate wind direction using local gradient method in Koch 2004
    sig0: numpy array of sigma0 
    wf: numpy array of water fraction 
    returns: wind direction 
    """
    
    # Sobel kernels
    Dx = (1/32) * np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]])
    Dy = Dx.transpose()

    # Convert the Dy kernel to complex matrix
    make_complex = lambda t: complex(0, t)
    vfunc = np.vectorize(make_complex)
    Dy_complex = vfunc(Dy)

    # Convolve sig0 with horizontal and vertical gradients 
    G1 = convolve(sig0, (Dx+Dy_complex), mode='reflect') # reflect mode means outer values are reflected at the edge of the input to fill in missing values 

    # Square and smooth/resize/smooth the gradient image = B2 S|2 B4
    G2 = image_reduction(G1**2)

    # Calculate QC scalar c 
    G3 = image_reduction(np.abs(G1**2))
    c = np.abs(G2)/G3

    # Discard unusable points 
    # 1. The first and last two rows of the image
    # 2. Non-water pixels 

    # Resize the water fraction to match sigma0
    wf_halfsize = cv2.resize(wf, (int(wf.shape[1]/2), int(wf.shape[0]/2)), interpolation = cv2.INTER_CUBIC)

    # Create mask for pixels that half a water fraction < 0.9
    discard_mask = G2
    discard_mask = wf_halfsize >= 0.9

    # Mask the first two columns and rows 
    discard_mask[0:2,:] = False
    discard_mask[-2:,:] = False
    discard_mask[:,0:2] = False
    discard_mask[:,-2:] = False

    # Apply mask to G2 
    G2_masked = np.where(discard_mask, G2, np.nan)

    # Gf is obtained from normalized complex numbers that are weighted by c and r 
    # Calculate qc scalar r - it has to be after discarding points bc you take the mean of the full matrix 
    r = np.abs(G2_masked)/(np.abs(G2_masked) + np.nanmean(np.abs(G2_masked)))
    Gf = G2_masked * c * r

    # Gf is sorted into 72 bins that each represent 5 degrees
    angle = np.angle(Gf, deg = True)
    df = pd.DataFrame({"gf": Gf.ravel(),
                       "angle": angle.ravel()})
    df.loc[df["angle"]<0, "angle"] = df.loc[df["angle"]<0, "angle"] + 360
    df["bins"] = pd.cut(x=df["angle"], bins=range(0, 361, 5), labels=range(0, 360, 5))

    # All of the complex numbers in the same interval are summed to one complex number. Convert that complex number to magnitude
    weighted_bins = df.groupby(["bins"], observed=False)['gf'].sum().reset_index()
    weighted_bins['mag'] = np.abs(weighted_bins['gf'])

    # The histogram is smoothed using multiple operators and then bin of max smoothed histogram is determined 
    # treat hist like nx1 mat and just smooth it with kernels using convolution
    B28x = (1/4) * np.array([1,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,1])
    B24x = (1/4) * np.array([1,0,0,0,2,0,0,0,1])
    B22x = (1/4) * np.array([1,0,2,0,1])
    B2x = (1/4) * np.array([1,2,1])

    binned_mag = weighted_bins['mag'].to_numpy()
    binned_mag_s1 = convolve(binned_mag, B28x, mode='reflect')
    binned_mag_s2 = convolve(binned_mag_s1, B24x, mode='reflect')
    binned_mag_s3 = convolve(binned_mag_s2, B22x, mode='reflect')
    binned_mag_s4 = convolve(binned_mag_s3, B2x, mode='reflect')
    weighted_bins['smoothed_mag'] = binned_mag_s4

    max_mag = weighted_bins[weighted_bins["smoothed_mag"] == max(weighted_bins["smoothed_mag"])]

    # The sqrt of complex number in max bin gives the direction of max local gradient. The orthogonal of that gives wind direction w 180 degree ambiguity 
    wind_direction = np.angle(cmath.sqrt(max_mag['gf'].iloc[0]), deg=True) 
    # Gives degrees from -180 to 180. Transform to 360
    if wind_direction < 0: wind_direction = wind_direction + 360

    # X is positive to the right.  Y is positive down.  Therefore, a 45 degree angle is down and to the right.  
    # Angles are measured starting from the horizontal, and are positive as you go clockwise (down).
    # So to get the right direction, add 90 degrees!!!! 
    wind_direction = (wind_direction + 90)%360

    # Add another 90 because wind direction is orthogonal to the max gradient 
    wind_direction = (wind_direction + 90)%360

    return wind_direction



def quality_parameters_by_image(satellite, infile, boundary, outdir, outprefix, threshold, min_area):
    """
    Calculate the QC mask from Koch 2004 that attempts to find non-wind features 
    infile: filepath to the .nc file for swot or sentinel 
    boundary: geopandas shapefile of all lakes included in the study 
    outdir: directory to output the qc tif 
    outprefix: prefix in the filename written out. comes before the image id
    threshold: 0-1 threshold for the F statistic. above threshold = usable. Koch 2004 used 0.6
    min_area: minimum area in m2 of usable or unusable qc patch until it is switched - -from koch 2004
    return write qc to tif 
    """
    # Open the netcdf 
    ds = rioxarray.open_rasterio(infile)
    sig0 = ds.sig0

    # SWOT .nc file requires a couple extra steps of processing 
    if satellite == "SWOT":
        # Covnert no data to na 
        sig0 = sig0.where(sig0 != sig0.rio.nodata)
        #Filter sig0 observations. Only keep observations that corresond to 0 (good), 1(suspect) and 2 (degraded). 3 = bad 
        sig0 = sig0.where(ds.sig0_qual<=2)
        # Get the epsg string and update the crs 
        crs_wkt = sig0.crs.attrs['crs_wkt']
        crs_wkt_split = crs_wkt.split(',')
        epsg = crs_wkt_split[len(crs_wkt_split)-1].split('"')[1]
        sig0.rio.write_crs("epsg:"+epsg, inplace=True)

    # Project boundary shapefile to match ncdf and mask values outside the boundary to nan 
    # This is a rough way of filtering out pixels that are not water 
    # Convert to numpy array     
    boundary_reproj = boundary.to_crs(sig0.crs.attrs['crs_wkt'])
    sig0_clipped = sig0.rio.clip(boundary_reproj.geometry.values, boundary_reproj.crs)
    sig0_np = sig0_clipped.to_numpy()[0, :, :]
    boundary_mask = sig0_np.copy()

    # Convert the NAs to random fill very close to 0. Necessary bc convolution doesn't work with NAs 
    # Must be random distribution because otherwise all 0s would break the gradient calculations
    # This doesn't change the result because the NA areas are not water and will be masked out again at the end.
    # It's just necessary to get the QC results for the water pixels 
    def rand_fill(x):
        if ((np.isnan(x)) | (x <= 0)):
            mu, sigma = 0.1, 0.005
            return np.random.normal(mu, sigma, 1)[0]
        else: return x
    rf = np.vectorize(rand_fill)
    sig0_np = rf(sig0_np)

    # Calculate F, the QC parameter. Since the raster will never be square we must chunk it
    # Figure out the square chunks
    num_sqr = math.ceil(max(sig0_np.shape)/min(sig0_np.shape))
    min_idx = sig0_np.shape.index(min(sig0_np.shape))
    if min_idx == 1:
        # all of the columns are used and break up the rows 
        col_start = [0]*num_sqr
        col_end = [sig0_np.shape[1]]*num_sqr
        row_start = list(range(0, sig0_np.shape[1]*num_sqr+1, sig0_np.shape[1]))[:-1]
        row_end = row_start[1:] + [sig0_np.shape[0]]
        row_start[num_sqr-1] = row_end[num_sqr-1] - sig0_np.shape[1]
    if min_idx == 0: 
        # all of the rows are used and break up the columns 
        row_start = [0]*num_sqr
        row_end = [sig0_np.shape[0]]*num_sqr
        col_start = list(range(0, sig0_np.shape[0]*num_sqr+1, sig0_np.shape[0]))[:-1]
        col_end = col_start[1:] + [sig0_np.shape[1]]
        col_start[num_sqr-1] = col_end[num_sqr-1]-sig0_np.shape[0]

    f_list = []
    for i in range(0, len(col_start)):
        n = np.empty((sig0_np.shape[0], sig0_np.shape[1]))
        n[:] = np.nan
        n[row_start[i]:row_end[i], col_start[i]:col_end[i]] = connecting_quality_parameters(sig0_np[row_start[i]:row_end[i], col_start[i]:col_end[i]], threshold, min_area)
        f_list.append(n)  
    f = np.nansum(f_list, axis=0)
    f = np.where(f >= 1, 1, 0)

    # Redo the water boundary mask  
    f = np.where(~np.isnan(boundary_mask), f, np.nan)

    # Save as a tif by adding the numpy array to rioxarray dataset
    sig0_clipped['F'] = (['y', 'x'], f)
    F = sig0_clipped.reset_coords('F').F
    sig0_clipped = sig0_clipped.drop_vars('F')
    F = xarray.combine_by_coords([sig0_clipped, F])
    F.F.rio.write_nodata(np.nan, encoded=True, inplace=True)
    F.F.rio.to_raster(os.path.join(outdir, outprefix + "_" + os.path.basename(infile)[:-3] + ".tif"))      



def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]



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
            geom = gpd.GeoDataFrame({'geometry':box(*match.bounds)}, index=[0], crs=match.crs).to_crs(src.crs)
            geom = getFeatures(geom)
            src_clip_img, src_clip_transform = mask(src, shapes=geom, crop=True, all_touched =True)
            src_clip_img[src_clip_img == src.nodata] = np.nan

        # set properties for output
        dst_kwargs = src.meta.copy()
        dst_kwargs.update({"crs": dst_crs,
                           "transform": dst_transform,
                           "width": dst_width,
                           "height": dst_height,
                           "nodata": dst_nodata})
        
        # open output
        with rasterio.open(outfile, "w", **dst_kwargs) as dst:
            reproject(
                source=src_clip_img,
                destination=rasterio.band(dst, 1),
                src_transform=src_clip_transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)



def correct_180_ambiguity(wdir, sat_wdir):
    """
    Correct 180 degree ambiguity using another source of wind direction 
    wdir: "true" wind direction. probably from a reanalysis dataset
    swot_wdir: 180 degree ambiguous swot direction
    return: swot wind direction corrected for 180 degree ambiguity
    """
    if ((np.isnan(sat_wdir)) | (np.isnan(wdir))): return np.nan

    upper = (wdir + 90)%360
    lower = (wdir - 90)%360

    if lower>upper: 
        env = list(range(int(lower), 360)) + list(range(0, int(upper)+1))
        if int(sat_wdir) in env: 
            fixed_wdir = sat_wdir
        else: fixed_wdir = (sat_wdir + 180)%360
    if lower<upper:
        env = list(range(int(lower), int(upper)+1))
        if int(sat_wdir) in env:
            fixed_wdir = sat_wdir 
        else: fixed_wdir = (sat_wdir + 180)%360
    return fixed_wdir



def correction_with_era5(satellite, wdir_file, outfile):
    """
    Correct the satellite wind direction estimates for 180 degree ambiguity using ERA5
    wdir_file: raster file of the satellite wind direction estimate
    outfile: filepath to corrected wind direction 
    """
    if satellite == "Sentinel1":
        date = datetime.strptime(os.path.basename(wdir_file).split("_")[6], "%Y%m%dT%H%M%S") 
        date = date.replace(second=0, microsecond=0, minute=0)
    if satellite == "SWOT":
        date = datetime.strptime(os.path.basename(wdir_file).split("_")[15][0:11], "%Y%m%dT%H")
    
    era5_files = glob.glob(os.path.join(home, "Data/ERA5/Processed/wdir/*.tif"))
    era5_dates = [datetime.strptime(os.path.basename(i)[0:10], "%Y%m%d%H") for i in era5_files]
    era5_file = era5_files[era5_dates.index(date)]
    reproj_match(era5_file, wdir_file, outfile )
    with rasterio.open(wdir_file) as swot:
        swot_wd = swot.read()
        swot_wd = swot_wd[0, :, :]

        with rasterio.open(outfile) as era5:
            era5_match = era5.read()
            era5_match = era5_match[0, :, :]

        fv = np.vectorize(correct_180_ambiguity)
        fixed_swot_wd = fv(era5_match, swot_wd)
        kwargs=swot.meta
        with rasterio.open(outfile, "w", **kwargs) as dst:
            dst.write_band(1, fixed_swot_wd)



def wind_direction_by_image(satellite, infile, outfile, boundary, qc_mask, N, max_na_frac, min_wf_frac):
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
        #Filter sig0 observations. Only keep observations that corresond to 0 (good), 1(suspect) and 2 (degraded). 3 = bad 
        sig0 = sig0.where(ds.sig0_qual<=2)
        # Get the epsg string and update the crs 
        crs_wkt = sig0.crs.attrs['crs_wkt']
        crs_wkt_split = crs_wkt.split(',')
        epsg = crs_wkt_split[len(crs_wkt_split)-1].split('"')[1]
        sig0.rio.write_crs("epsg:"+epsg, inplace=True)

    # Project boundary shapefile to match ncdf and mask values outside the boundary to nan 
    # This is a rough way of filtering out pixels that are not water 
    # Convert to numpy array     
    boundary_reproj = boundary.to_crs(sig0.rio.crs)
    sig0_clipped = sig0.rio.clip(boundary_reproj.geometry.values, boundary_reproj.crs)
    sig0_clipped = sig0_clipped.where(sig0_clipped != sig0_clipped.rio.nodata) # added this post s1 
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
        wf_np = np.where(np.isnan(qc_np)==True, 0, wf_np)

    # define the subsets based on the shape of the array 
    row_start = [*range(0, sig0_np.shape[0], N)]
    row_start = row_start[0:len(row_start)-1]
    row_end = [x+N for x in row_start]

    col_start = [*range(0, sig0_np.shape[1], N)]
    col_start = col_start[0:len(col_start)-1]
    col_end = [x+N for x in col_start]

    # Copy the array shape and fill in with the wind direction 
    wind_direction_np = np.empty((sig0_np.shape[0], sig0_np.shape[1]))
    wind_direction_np[:] = np.nan

    # For loop to traverse the sig0 raster and extract wind direction at subsets 
    for i in range(0, len(row_start)):
        for j in range(0, len(col_start)):
            # Use the row/col start/end to select the subset
            sig0_sub = sig0_np[row_start[i]:row_end[i], col_start[j]:col_end[j]]
            wf_sub = wf_np[row_start[i]:row_end[i], col_start[j]:col_end[j]]

            # Fill the wf with 0.
            wf_sub_filled = np.where(np.isnan(wf_sub) == True, 0, wf_sub)

            # Calculate the fraction of subset that is NA and the fraction of the subset that is covered by water 
            # Only keep subsets with 0% NA and 100% water 
            na_frac = np.sum(np.isnan(sig0_sub))/(sig0_sub.shape[0]*sig0_sub.shape[1])
            wf_frac = np.nanmean(wf_sub_filled)
            if ((na_frac > max_na_frac) | (wf_frac < min_wf_frac)): continue

            # If there are Na values, fill using cubic interpolation. Necessary bc convolution won't work w NAs
            x = np.arange(0, sig0_sub.shape[1])
            y = np.arange(0, sig0_sub.shape[0])
            #mask invalid values
            sig0_sub = np.ma.masked_invalid(sig0_sub)
            xx, yy = np.meshgrid(x, y)
            #get only the valid values
            x1 = xx[~sig0_sub.mask]
            y1 = yy[~sig0_sub.mask]
            newarr = sig0_sub[~sig0_sub.mask]
            sig0_sub_filled = interpolate.griddata((x1, y1), newarr.ravel(),
                                                   (xx, yy),
                                                   method='cubic',
                                                   fill_value=np.mean(newarr))

            # Calculate the wind direction 
            wind_direction = calc_wind_direction(sig0_sub_filled, wf_sub_filled)
            # Put that wind direction in all cells belonging to this subset on wind direction np array 
            wind_direction_np[row_start[i]:row_end[i], col_start[j]:col_end[j]] = wind_direction

    # Combine the sigma0 and wind direction 2d array into one xarray dataset that retains the coords/spatial referencing
    sig0_clipped['wd'] = (['y', 'x'], wind_direction_np)
    wd = sig0_clipped.reset_coords("wd").wd
    sig0_clipped = sig0_clipped.drop_vars('wd')
    final = xarray.combine_by_coords([sig0_clipped, wd])
    final.wd.rio.write_nodata(np.nan, encoded=True, inplace=True)
    final.wd.rio.to_raster(outfile)

    #Resample using N as the scale and write to raster 
    #downscale_factor = 1/N
    #new_width = final.rio.width * downscale_factor
    #new_height = final.rio.height * downscale_factor
    #down_sampled = final.rio.reproject(final.rio.crs, shape=(int(new_height), int(new_width)), resampling=Resampling.nearest)
    #down_sampled.wd.rio.to_raster(outfile)
   
    # Correct the 180 degree ambiguity 
    # Add the ERA5 corrected folder if it doesn't already exist 
    corrected_path = os.path.join(os.path.dirname(outfile), "ERA5_Corrected")
    isExist = os.path.exists(corrected_path)
    if not isExist:
       os.makedirs(corrected_path)
    correction_with_era5(satellite, outfile, os.path.join(corrected_path, "corrected_" + os.path.basename(outfile)))

    return



#########################################################################################
# Calculate wind direction 
#########################################################################################
# Calculate wind direction for these scenarios: 
# Wind direction with no QC at 1 km, 2 km, 4km, 8 km 
# Wind direction with QC at 1 km, 2 km, 4 km, 8 km

s1_files = glob.glob(os.path.join(home, "Data/Sentinel1/Processed/*.nc"))
swot1_files = glob.glob(os.path.join(home, "Data/SWOT_L2_HR_Raster/SWOT_L2_HR_Raster_1.1/*.nc"))
swot2_files = glob.glob(os.path.join(home, "Data/SWOT_L2_HR_Raster/SWOT_L2_HR_Raster_2.0/*.nc"))
boundary = gpd.read_file(os.path.join(home, "Data/Buoy/pld_with_buoys.shp"))

#########################################################################################
# Subset to files that are within 3 hours of a buoy overpass 
# This should cut down on processing time
from datetime import timezone 
# Open df. each  buoy has a row for each image id that intersects it 
buoy_sat_int = pd.read_csv(os.path.join(home, "Data", "buoy_satellite_intersection.csv"))
# Find the index of images with a buoy observation within 3 hours 
index = []
for i in range(0, buoy_sat_int.shape[0]):
    print(i)
    sub = buoy_sat_int.iloc[i]
    buoy_id = sub.buoy_id
    buoy_df = pd.read_csv(os.path.join(home, "Data/Buoy/Processed", buoy_id + ".csv"))
    buoy_df['datetime']= pd.to_datetime(buoy_df['datetime'], utc=True, format='ISO8601')

    satellite = sub.satellite
    image_id = os.path.basename(sub.image_id)[:-3]
    if satellite == "Sentinel1": overpass_datetime = datetime.strptime(image_id.split("_")[5], "%Y%m%dT%H%M%S")
    else: overpass_datetime = datetime.strptime(image_id.split("_")[13], "%Y%m%dT%H%M%S")
    overpass_datetime = overpass_datetime.replace(tzinfo=timezone.utc)

    diffs = (buoy_df['datetime'] - overpass_datetime).tolist()
    # Just want differences less than 0 so that the buoy observation is before overpass time 
    diffs_less0 = [t for t in diffs if t < pd.Timedelta(0)]
    # To find the closest to 0, get the max
    if len(diffs_less0) == 0: 
        continue
    else:
        min_diff = abs(max(diffs_less0))
        if min_diff < timedelta(hours=3): 
            index.append(i)
        else: continue

buoy_sat_int_sub = buoy_sat_int.iloc[index]
all_images = buoy_sat_int_sub['image_id'].tolist()
all_images = [os.path.basename(i) for i in all_images]

# Subset S1, SWOT 1.1 and SWOT 2.0 files to only include the ones in the new df 
s1_files = [file for file in s1_files if any([scene in file for scene in all_images])]
swot1_files = [file for file in swot1_files if any([scene in file for scene in all_images])]
swot2_files = [file for file in swot2_files if any([scene in file for scene in all_images])]

#########################################################################################
# Wind direction no QC at 1km resolution 
# Sentinel 
N = 10
for i in range(0, len(s1_files)):
    infile = s1_files[i]
    outfile = os.path.join(home, "Data/Wind_Direction/Sentinel1", "wdir_" + str(N) + "pixels_" + os.path.basename(infile[:-3]) + ".tif")
    wind_direction_by_image("Sentinel1", infile, outfile, boundary, "NA", N, 0.6, 0.4)

# SWOT 1.1 
for i in range(0, len(swot1_files)):
    infile = swot1_files[i]
    outfile = os.path.join(home, "Data/Wind_Direction/SWOT_L2_HR_Raster_1.1", "wdir_" + str(N) + "pixels_" + os.path.basename(infile[:-3]) + ".tif")
    if os.path.isfile(outfile): continue
    wind_direction_by_image("SWOT", infile, outfile, boundary, "NA", N,  0.6, 0.4)
    
# SWOT 2.0 
for i in range(0, len(swot2_files)):
    infile = swot2_files[i]
    outfile = os.path.join(home, "Data/Wind_Direction/SWOT_L2_HR_Raster_2.0", "wdir_" + str(N) + "pixels_" + os.path.basename(infile[:-3]) + ".tif")
    if os.path.isfile(outfile): continue
    wind_direction_by_image("SWOT", infile, outfile, boundary, "NA", N, 0.6, 0.4)


################################################################################################
# 3 extra files that didn't have matchups, don't include in actual analysis 
################################################################################################
# Just do 3 files 
extra_files = ["C:/Users/kmcquil/Documents/SWOT_WIND/Data/SWOT_L2_HR_Raster/SWOT_L2_HR_Raster_2.0/SWOT_L2_HR_Raster_100m_UTM18T_N_x_x_x_010_048_039F_20240126T165014_20240126T165035_PIC0_01.nc", 
               "C:/Users/kmcquil/Documents/SWOT_WIND/Data/SWOT_L2_HR_Raster/SWOT_L2_HR_Raster_2.0/SWOT_L2_HR_Raster_100m_UTM18T_N_x_x_x_011_397_116F_20240229T011603_20240229T011624_PIC0_01.nc", 
               "C:/Users/kmcquil/Documents/SWOT_WIND/Data/SWOT_L2_HR_Raster/SWOT_L2_HR_Raster_2.0/SWOT_L2_HR_Raster_100m_UTM18T_N_x_x_x_012_397_116F_20240320T220106_20240320T220127_PIC0_01.nc"]

for i in range(0, len(extra_files)):
    infile = extra_files[i]
    outfile = os.path.join(home, "Data/Wind_Direction/SWOT_L2_HR_Raster_2.0", "wdir_" + str(N) + "pixels_" + os.path.basename(infile[:-3]) + ".tif")
    if os.path.isfile(outfile): continue
    wind_direction_by_image("SWOT", infile, outfile, boundary, "NA", N, 0.6, 0.4)

import shutil
for i in range(0, len(extra_files)):
    infile = extra_files[i]
    outfile = os.path.join(home, "Data/Wind_Direction/SWOT_L2_HR_Raster_2.0", "ERA5_Corrected","corrected_wdir_" + str(N) + "pixels_" + os.path.basename(infile[:-3]) + ".tif")
    wd_out = os.path.join("C:/Users/kmcquil/Documents/SWOT_WIND/Data/Maps/SWOT/", os.path.basename(outfile))
    shutil.copy(outfile, wd_out)
################################################################################################
################################################################################################


#########################################################################################
# Wind direction no QC at 2km resolution 
N = 20
for i in range(0, len(s1_files)):
    infile = s1_files[i]
    outfile = os.path.join(home, "Data/Wind_Direction/Sentinel1", "wdir_" + str(N) + "pixels_" + os.path.basename(infile[:-3]) + ".tif")
    wind_direction_by_image("Sentinel1", infile, outfile, boundary, "NA", N, 0.6, 0.4)

# SWOT 1.1 
for i in range(0, len(swot1_files)):
    infile = swot1_files[i]
    outfile = os.path.join(home, "Data/Wind_Direction/SWOT_L2_HR_Raster_1.1", "wdir_" + str(N) + "pixels_" + os.path.basename(infile[:-3]) + ".tif")
    wind_direction_by_image("SWOT", infile, outfile, boundary, "NA", N, 0.6, 0.4)


# SWOT 2.0 
for i in range(0, len(swot2_files)):
    infile = swot2_files[i]
    outfile = os.path.join(home, "Data/Wind_Direction/SWOT_L2_HR_Raster_2.0", "wdir_" + str(N) + "pixels_" + os.path.basename(infile[:-3]) + ".tif")
    wind_direction_by_image("SWOT", infile, outfile, boundary, "NA", N,0.6, 0.4)


#########################################################################################
# Wind direction no QC at 4km resolution 
N = 40
for i in range(0, len(s1_files)):
    infile = s1_files[i]
    outfile = os.path.join(home, "Data/Wind_Direction/Sentinel1", "wdir_" + str(N) + "pixels_" + os.path.basename(infile[:-3]) + ".tif")
    wind_direction_by_image("Sentinel1", infile, outfile, boundary, "NA", N, 0.6, 0.4)

# SWOT 1.1 
for i in range(0, len(swot1_files)):
    infile = swot1_files[i]
    outfile = os.path.join(home, "Data/Wind_Direction/SWOT_L2_HR_Raster_1.1", "wdir_" + str(N) + "pixels_" + os.path.basename(infile[:-3]) + ".tif")
    wind_direction_by_image("SWOT", infile, outfile, boundary, "NA", N, 0.6, 0.4)


# SWOT 2.0 
for i in range(0, len(swot2_files)):
    infile = swot2_files[i]
    outfile = os.path.join(home, "Data/Wind_Direction/SWOT_L2_HR_Raster_2.0", "wdir_" + str(N) + "pixels_" + os.path.basename(infile[:-3]) + ".tif")
    wind_direction_by_image("SWOT", infile, outfile, boundary, "NA", N, 0.6, 0.4)

