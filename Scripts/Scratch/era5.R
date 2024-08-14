#######################################################################################
# Katie Mcquillan
# 04/09/2024
# Process ERA5 data 
# Data downloaded from:
# https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form
#######################################################################################

library(httpgd)
library(languageserver)
library(ncdf4)
library(raster)
library(stringr)
home <- "C:/Users/kmcquil/Documents/SWOT_WIND/"

files <- list.files(paste0(home, "Data/ERA5/Raw"), full.names=TRUE)

for(file in files){
    era5 <- nc_open(file)
    era5_datetime <- as.character(as.POSIXct(ncvar_get(era5, "time")*3600, tz="UTC", origin="1900-01-01 00:00"))
    varsize <- era5$var$u10$varsize
    varsize[3] <- 1 
    xmin <- (min(ncvar_get(era5, "longitude")) - 0.125) - 180
    xmax <- (max(ncvar_get(era5, "longitude")) + 0.125) - 180
    ymin <- min(ncvar_get(era5, "latitude")) - 0.125
    ymax <- max(ncvar_get(era5, "latitude")) + 0.125

    for(i in 1:length(era5_datetime)){

        # check if the file already exists 
        date_time <- paste(c(unlist(str_split(substr(era5_datetime[i], 1, 10), "-")),  unlist(str_split(substr(era5_datetime[i], 12, 19), ":"))), collapse="")
        wspd_out <- paste0(home, "Data/ERA5/Processed/wspd/", date_time, ".tif")
        wdir_out <- paste0(home, "Data/ERA5/Processed/wdir/", date_time, ".tif")
        if(file.exists(wspd_out) == TRUE & file.exists(wdir_out) == TRUE){next}

        # If it didn't, extract wind speed and direction from the u and v components and save tifs
        u10 <- ncvar_get(era5, "u10", start=c(1,1,i), count=varsize)
        v10 <- ncvar_get(era5, "v10", start=c(1,1,i), count=varsize)

        wspd <- sqrt((u10^2) + (v10^2))
        wdir <- (270-atan2(u10,v10)*180/pi)%%360

        wspd <- raster(x = wspd, crs = "EPSG:4326", xmn=xmin, xmx=xmax, ymn=ymin, ymx=ymax)
        wdir <- raster(x = wdir, crs = "EPSG:4326", xmn=xmin, xmx=xmax, ymn=ymin, ymx=ymax)        
        writeRaster(wspd, wspd_out, overwrite = TRUE)
        writeRaster(wdir, wdir_out, overwrite = TRUE)
    }
   
}
