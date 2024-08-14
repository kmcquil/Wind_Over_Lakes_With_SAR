#######################################################################################
# Katie Mcquillan
# 03/20/2024
# Download and process buoy data 
#######################################################################################

library(ncdf4)
library(data.table)
library(sf)
library(stringr)
library(lubridate)
library(httpgd)
library(languageserver)

# Set the home directory
home <- "C:/Users/kmcquil/Documents/SWOT_WIND/"

#######################################################################################
#######################################################################################
# NOAA Buoys
#######################################################################################
#######################################################################################

#######################################################################################
# Find NOAA buoys located within prior lake database water bodies 
#######################################################################################
# Load csv of all current NOAA buoys
noah_stations <- fread(paste0(home, "Data/Buoy/NOAA/noah_national_buoy_center_active_station_list.csv"))
noah_stations <- st_as_sf(noah_stations, coords=c("lon", "lat"), crs=4326)
noah_stations$source <- "Noah National Data Buoy Center"

# List PLD shapefiles
pld_files <- list.files(paste0(home, "Data/SWOT_PLD"), full.names = TRUE, pattern = "*.shp$")
pld_files <- pld_files[grepl("pfaf", pld_files)]

# Check if buoys within PLD water body
pld_int <- function(x){
    "
    Test if buoy inside PLD
    x: sf df of pld sub dataset 
    return sf df of noaa stations that intersected with pld 
    "
    pld_sub <- st_read(pld_files[x])
    int_mat <- st_intersects(noah_stations, pld_sub, sparse = FALSE)
    noah_stations[rowSums(int_mat) >= 1, ]
}
sf_use_s2(FALSE)
stations_in_pld <- do.call(rbind, lapply(1:length(pld_files), pld_int))

# Save the NOAA stations within the PLD 
st_write(stations_in_pld, paste0(home, "Data/Buoy/NOAA/stations_in_pld.shp"), append = FALSE)

#######################################################################################
# Clean the buoy dataset to identify buoys with data during 2023-2024 
#######################################################################################
# I initially wrote code to scrape this from online but there were certain edge 
# cases that were missing data. For consistency, check each station manually for if it has
# height and wind direction and speed available in 2023-2024
# write out the station id names and track progress in csv
fwrite(as.data.table(as.data.frame(stations_in_pld[,c("id")])), paste0(home, "Data/Buoy/NOAA/stations_in_pld.csv"))

# Read in stations with the height, start, and end date manually added by going to each station web page 
station_info <- fread(paste0(home, "Data/Buoy/NOAA/stations_in_pld.csv"))
station_info[station_info == "NA"] <- NA

# Merge with the shapefile 
stations_in_pld <- merge(stations_in_pld, station_info, by="id", all.x=TRUE)

# Find stations that are classified as a buoy and have data in 2023-2024 
good_stations <- stations_in_pld[!is.na(stations_in_pld$end_date) & stations_in_pld$end_date >= 2023 & stations_in_pld$type == "buoy",]

# Now check each of those stations to make sure there is actually wind speed and direction available during 2023-2024
fwrite(as.data.table(as.data.frame(good_stations[,c("id", "Height", "start_date", "end_date")])), paste0(home, "Data/Buoy/NOAA/good_stations_in_pld.csv"))
stations <- fread(paste0(home, "Data/Buoy/NOAA/good_stations_in_pld.csv"))
stations_with_data <- stations[wspd == "yes" & wdir == "yes"]

# Merge those with the noaa dataset 
stations_in_pld <- st_read(paste0(home, "Data/Buoy/NOAA/stations_in_pld.shp"))
stations_in_pld <- merge(stations_in_pld, stations_with_data, by="id", inner=TRUE)

#######################################################################################
# One additional station needs to be added because it wasn't included in the initial
# dataset 
#######################################################################################

lake_murray <- data.table( id='LMFS1',
                            created=NA,
                            count=NA,
                            elev=NA,
                            name="Lake Murray",
                            owner="National Weather Service Eastern Region",
                            pgm=NA,
                            type="buoy",
                            met=NA,
                            currnts=NA,
                            wtrqlty=NA,
                            dart=NA,
                            seq=NA,
                            source= "Noah National Data Buoy Center",
                            Height=9,
                            start_date="2007",
                            end_date="2024",
                            wspd="yes",
                            wdir="yes", 
                            long=-81.271, lat=34.107)
lake_murray <- st_as_sf(lake_murray, coords = c("long","lat"), crs=st_crs(stations_in_pld))
stations_in_pld <- rbind(stations_in_pld, lake_murray)


#######################################################################################
# Fill in the rest of the heights and save the final noaa dataset
#######################################################################################
# Add height to the stations that are missing height
# The canada buoys can all get a standarized 10m height based on documentation here
# https://collaboration.cmc.ec.gc.ca/cmc/cmos/public_doc/msc-data/obs_station/SWOB-ML_Product_User_Guide_e.pdf
# The other buoys could get the average of be discarded 
stations_in_pld$height_est <- stations_in_pld$Height
stations_in_pld[stations_in_pld$owner == "Environment and Climate Change Canada",]$height_est <- 10
stations_in_pld[is.na(stations_in_pld$height_est),]$height_est <- mean(stations_in_pld$Height, na.rm=TRUE)

# Save shapefile as the final noaa stations to use in the study
st_write(stations_in_pld, paste0(home, "Data/Buoy/NOAA/noaa_stations.shp"), append=FALSE)
# Remove the intermidate files created

#######################################################################################
# Manually downloaded data. Process into a common format.
#######################################################################################

process_noaa <- function(row){

    # Extract files corresponding to the stations id
    all_files <- list.files(paste0(home, "Data/Buoy/NOAA/Raw"), full.names=TRUE)
    files <- all_files[grepl(row$id, all_files)]
    dts <- list()
    # Since there can be more than one file associated with the station id, loop through
    for(i in 1:length(files)){

        # Files must be processed separately if they are owned by Canada 
        if(row$owner == "Environment and Climate Change Canada"){
            table_content <- fread(files[i])
            table_content <- table_content[-1,]
            num_cols <- c("avg_wnd_spd_pst10mts", "avg_wnd_dir_pst10mts", "avg_wnd_spd_pst10mts_qa_summary", "avg_wnd_dir_pst10mts_qa_summary")
            table_content[, (num_cols) := lapply(.SD, as.numeric), .SDcols = num_cols]
            table_content[, wmo_synop_id := as.character(wmo_synop_id)]  
            table_content[, date_tm := as.POSIXct(date_tm, tz="UTC", format="%Y-%m-%dT%H:%M:%SZ")]
            table_content[, date_tm := as.character(date_tm)]
        
            # Convert wspd and wdir to NA if doesn't meet QC check 
            # https://collaboration.cmc.ec.gc.ca/cmc/cmos/public_doc/msc-data/obs_station/SWOB-ML_Product_User_Guide_e.pdf
            table_content[avg_wnd_spd_pst10mts_qa_summary < 100,]$avg_wnd_spd_pst10mts <- NA
            table_content[avg_wnd_dir_pst10mts_qa_summary < 100,]$avg_wnd_dir_pst10mts <- NA

            # Drop rows where wind speed and direction are both missing
            table_content <- table_content[!is.na(avg_wnd_spd_pst10mts) & !is.na(avg_wnd_dir_pst10mts)]

            # Convert from km/hr to m/s
            table_content[, avg_wnd_spd_pst10mts := avg_wnd_spd_pst10mts * (1000/3600)]

            # Convert to a standard format 
            dt <- data.table(id=rep(row$id, nrow(table_content)),
                        datetime=table_content$date_tm,
                        wspd=table_content$avg_wnd_spd_pst10mts, 
                        wdir=table_content$avg_wnd_dir_pst10mts)
            dts[[i]] <- dt


        }else{
            # Convert to standard format and clean missing observations 
            # they have different column order and sometimes different column names depending on year
            # go through each element of the data list and pull out the same columns in the same order 
            # fix the number of digits in years earlier than 1998
            format_dt <- function(x){
                x <- fread(x)
                x_colnames <- lapply(colnames(x), FUN=function(y){z <- strsplit(y, ' ')[[1]]; z[!z == ""]})
                index <- which(x_colnames %in% c("YY", "YYYY", "#YY", "MM", "DD", "hh", "mm", "WD", "WDIR", "WSPD"))
                x_indexed <- x[,..index]
                colnames(x_indexed) <- c("year", "month", "day", "hour", "minute", "wdir", "wspd")

                # Convert direction and speed to numeric then drop if both NA. 
                # This takes care of the years when the second row is sometimes units
                x_indexed$wdir <- as.numeric(x_indexed$wdir)
                x_indexed$wspd <- as.numeric(x_indexed$wspd)
                x_indexed <- x_indexed[!is.na(wdir) & !is.na(wspd)]

                # fix the year to be four digits. It is two digits 1998 and earlier 
                x_indexed$year <- as.numeric(x_indexed$year)
                x_indexed[year < 1000]$year <- x_indexed[year < 1000]$year + 1900
                return(x_indexed)
            }

            dt <- as.data.table(format_dt(files[i]))
            dt$datetime <- as.character(as.POSIXct(paste0(dt$year, dt$month, dt$day, dt$hour, dt$minute), tz="UTC", format="%Y%m%d%H%M"))
            dt$id <- row$id
            dt[wspd == 99]$wspd <- NA
            dt[wdir == 999]$wdir <- NA
            dt <- dt[!is.na(wdir)]
            #dt <- dt[!is.na(wspd) | !is.na(wdir)]

            # add the other variables 
            dt <- dt[,c("id","datetime", "wspd", "wdir")]
            dts[[i]] <- dt
        }  

    }

    # Combine dts and save
    final_dt <- rbindlist(dts)
    fwrite(final_dt, paste0(paste0(home, "Data/Buoy/Processed/"), row$id, ".csv"))
    

}

stations_in_pld <- st_read(paste0(home, "Data/Buoy/NOAA/noaa_stations.shp"))
for(i in 1:nrow(stations_in_pld)){
    print(i)
    process_noaa(stations_in_pld[i,])
}

#######################################################################################
#######################################################################################
# Swiss Buoys from Datalakes 
#######################################################################################
#######################################################################################
# We have 5 total stations across 4 lakes 
# 2 on lake geneva, 1 on lake murten, lake aegeri, and lake greifen 
# the data is all formatted differently, so go one by one :)

####################################################################################### 
# Buchillon - Lake Geneva
# Unzip and list the files 
file = paste0(home, "Data/Buoy/Swiss_Datalakes/Raw/buchillonfieldstation_datalakesdownload.zip")
exdir <- paste0(home, exdir="Data/Buoy/Swiss_Datalakes/Raw/", substr(basename(file), 1, nchar(basename(file))-4))
untar(file, exdir = exdir)
ncs <- list.files(exdir, full.names=TRUE, pattern="*.nc$")

# Extract data into a dt
dt_list <- list()
for(i in 1:length(ncs)){
    file_nc <- nc_open(ncs[i])
    dt <- data.table(datetime=ncvar_get(file_nc, "time"),
                    wspd=ncvar_get(file_nc, "wind_speed"), 
                    wspd_qual=ncvar_get(file_nc, "wind_speed_qual"),
                    wdir=ncvar_get(file_nc, "wind_dir"), 
                    wdir_qual=ncvar_get(file_nc, "wind_dir_qual"))
    nc_close(file_nc)
    dt$datetime <- as.POSIXct(dt$datetime, origin = "1970-01-01")
    dt_list[[i]] <- dt
}
dt_all <- rbindlist(dt_list)
    
# For qual, 0 = nothing to report and 1 = more investigation. Convert to NA where it equals 1
dt_all[wspd_qual == 1]$wspd <- NA
dt_all[wdir_qual == 1]$wdir <- NA

# Add station id, source and height 
dt_all$id <- strsplit(basename(file), "_")[[1]][1]

# Clean it up and add height and source 
dt_all <- dt_all[,c("id", "datetime", "wspd", "wdir")]
fwrite(dt_all, paste0(home, "Data/Buoy/Processed/", strsplit(basename(file), "_")[[1]][1], ".csv"))
  
####################################################################################### 
# Léxplore meteostation - Lake Geneva

file = paste0(home, "Data/Buoy/Swiss_Datalakes/Raw/léxploremeteostation_datalakesdownload.zip")
exdir <- paste0(home, exdir="Data/Buoy/Swiss_Datalakes/Raw/", substr(basename(file), 1, nchar(basename(file))-4))
untar(file, exdir = exdir)
ncs <- list.files(exdir, full.names=TRUE, pattern="*.nc$")

# Extract data into a dt
dt_list <- list()
for(i in 1:length(ncs)){
    file_nc <- nc_open(ncs[i])
    dt <- data.table(datetime=ncvar_get(file_nc, "time"),
                    wspd=ncvar_get(file_nc, "WS"), 
                    wspd_qual=ncvar_get(file_nc, "WS_qual"),
                    wdir=ncvar_get(file_nc, "WindDir"), 
                    wdir_qual=ncvar_get(file_nc, "WindDir_qual"))
    nc_close(file_nc)
    dt$datetime <- as.POSIXct(dt$datetime, origin = "1970-01-01")
    dt_list[[i]] <- dt
}
dt_all <- rbindlist(dt_list)
    
# For qual, 0 = nothing to report and 1 = more investigation. Convert to NA where it equals 1
dt_all[wspd_qual == 1]$wspd <- NA
dt_all[wdir_qual == 1]$wdir <- NA

# Add station id, source and height 
dt_all$id <- strsplit(basename(file), "_")[[1]][1]

# Clean it up and add height and source 
dt_all <- dt_all[,c("id", "datetime", "wspd", "wdir")]
fwrite(dt_all, paste0(home, "Data/Buoy/Processed/", strsplit(basename(file), "_")[[1]][1], ".csv"))

####################################################################################### 
# Lake Murten 

file = paste0(home, "Data/Buoy/Swiss_Datalakes/Raw/meteostationlakemurten_datalakesdownload.zip")
exdir <- paste0(home, exdir="Data/Buoy/Swiss_Datalakes/Raw/", substr(basename(file), 1, nchar(basename(file))-4))
untar(file, exdir = exdir)
ncs <- list.files(exdir, full.names=TRUE, pattern="*.nc$")

# Extract data into a dt
dt_list <- list()
for(i in 1:length(ncs)){
    file_nc <- nc_open(ncs[i])
    dt <- data.table(datetime=ncvar_get(file_nc, "time"),
                    wspd=ncvar_get(file_nc, "WS"), 
                    wspd_qual=ncvar_get(file_nc, "WS_qual"),
                    wdir=ncvar_get(file_nc, "WindDir"), 
                    wdir_qual=ncvar_get(file_nc, "WindDir_qual"))
    nc_close(file_nc)
    dt$datetime <- as.POSIXct(dt$datetime, origin = "1970-01-01")
    dt_list[[i]] <- dt
}
dt_all <- rbindlist(dt_list)
    
# For qual, 0 = nothing to report and 1 = more investigation. Convert to NA where it equals 1
dt_all[wspd_qual == 1]$wspd <- NA
dt_all[wdir_qual == 1]$wdir <- NA

# Add station id, source and height 
dt_all$id <- strsplit(basename(file), "_")[[1]][1]

# Clean it up and add height and source 
dt_all <- dt_all[,c("id", "datetime", "wspd", "wdir")]
fwrite(dt_all, paste0(home, "Data/Buoy/Processed/", strsplit(basename(file), "_")[[1]][1], ".csv"))


####################################################################################### 
# Lake Greifen 

file = paste0(home, "Data/Buoy/Swiss_Datalakes/Raw/lakegreifenmeteostation_datalakesdownload.zip")
exdir <- paste0(home, exdir="Data/Buoy/Swiss_Datalakes/Raw/", substr(basename(file), 1, nchar(basename(file))-4))
untar(file, exdir = exdir)
ncs <- list.files(exdir, full.names=TRUE, pattern="*.nc$")

# Extract data into a dt
dt_list <- list()
for(i in 1:length(ncs)){
    file_nc <- nc_open(ncs[i])
    dt <- data.table(datetime=ncvar_get(file_nc, "time"),
                    wspd=ncvar_get(file_nc, "wind speed"),
                    wdir=ncvar_get(file_nc, "wind direction"))
    nc_close(file_nc)
    dt$datetime <- as.POSIXct(dt$datetime, origin = "1970-01-01")
    dt_list[[i]] <- dt
}
dt_all <- rbindlist(dt_list)
    
# No quality checks available
dt_all$id <- strsplit(basename(file), "_")[[1]][1]

# Clean it up and add height and source 
dt_all <- dt_all[,c("id", "datetime", "wspd", "wdir")]
fwrite(dt_all, paste0(home, "Data/Buoy/Processed/", strsplit(basename(file), "_")[[1]][1], ".csv"))


####################################################################################### 
# Lake Aegeri 

file = paste0(home, "Data/Buoy/Swiss_Datalakes/Raw/meteostationlakeaegeri_datalakesdownload.zip")
exdir <- paste0(home, exdir="Data/Buoy/Swiss_Datalakes/Raw/", substr(basename(file), 1, nchar(basename(file))-4))
untar(file, exdir = exdir)
ncs <- list.files(exdir, full.names=TRUE, pattern="*.nc$")

# Extract data into a dt
dt_list <- list()
for(i in 1:length(ncs)){
    file_nc <- nc_open(ncs[i])
    dt <- data.table(datetime=ncvar_get(file_nc, "time"),
                    wspd=ncvar_get(file_nc, "WS"), 
                    wspd_qual=ncvar_get(file_nc, "WS_qual"),
                    wdir=ncvar_get(file_nc, "WindDir"), 
                    wdir_qual=ncvar_get(file_nc, "WindDir_qual"))
    nc_close(file_nc)
    dt$datetime <- as.POSIXct(dt$datetime, origin = "1970-01-01")
    dt_list[[i]] <- dt
}
dt_all <- rbindlist(dt_list)
    
# No quality checks available
dt_all$id <- strsplit(basename(file), "_")[[1]][1]

# Clean it up and add height and source 
dt_all <- dt_all[,c("id", "datetime", "wspd", "wdir")]
fwrite(dt_all, paste0(home, "Data/Buoy/Processed/", strsplit(basename(file), "_")[[1]][1], ".csv"))


# One datatable with id, date, wind speed, wind direction for each station 
# One datatable with id, source, height, lat, long that includes all stations 

# buchillon = 10
# lexplore = 5 
# murten = 2 
# greifen = unknown
# aegeri = unkown


#######################################################################################
#######################################################################################
# King County, Washington, USA Buoys  
#######################################################################################
#######################################################################################

##################################################################################################
## Process wind vector from buoy at Lake Washington
lake_washington_dt_2023 <- fread(paste0(home, "Data/Buoy/King_County/Raw/WashingtonMet_2023.txt"))
lake_washington_dt_2024 <- fread(paste0(home, "Data/Buoy/King_County/Raw/WashingtonMet_2024.txt"))
lake_washington_dt <- rbind(lake_washington_dt_2023, lake_washington_dt_2024)
lake_washington_dt$datetime <- as.POSIXct(lake_washington_dt$Date, format="%m/%d/%Y %I:%M:%S %p", origin = '1970-01-01' , tz ="America/Los_Angeles")
attributes(lake_washington_dt$datetime)$tzone <- "UTC"
lake_washington_dt$id <- "Washington"
lake_washington_dt <- lake_washington_dt[,c("id", "datetime", "Wind Speed (m/sec)", "Wind Direction (degrees)")]
colnames(lake_washington_dt) <- c("id", "datetime", "wspd", "wdir")
fwrite(lake_washington_dt, paste0(home, "Data/Buoy/Processed/Washington.csv"))

##################################################################################################
## Process wind vector from buoy at Lake Sammamish
lake_sammamish_dt_2024 <- fread(paste0(home, "Data/Buoy/King_County/Raw/SammamishMet_2024.txt"))
lake_sammamish_dt_2023 <- fread(paste0(home, "Data/Buoy/King_County/Raw/SammamishMet_2023.txt"))
lake_sammamish_dt <- rbind(lake_sammamish_dt_2024, lake_sammamish_dt_2023)
lake_sammamish_dt$datetime <- as.POSIXct(lake_sammamish_dt$Date, format="%m/%d/%Y %I:%M:%S %p", origin = '1970-01-01' , tz ="America/Los_Angeles")
lake_sammamish_dt$id <- "Sammamish"
lake_sammamish_dt <- lake_sammamish_dt[,c("id", "datetime", "Wind Speed (m/sec)", "Wind Direction (degrees)")]
colnames(lake_sammamish_dt) <- c("id", "datetime", "wspd", "wdir")
fwrite(lake_sammamish_dt, paste0(home, "Data/Buoy/Processed/Sammamish.csv"))


#######################################################################################
#######################################################################################
# Create dt with all buoys that incudes important attributes like 
# location and height 
#######################################################################################
#######################################################################################

# Columns to include buoy_id, buoy name, source, height, lat, long
# Start with noaa stations 
noah_stations <- st_read(paste0(home, "Data/Buoy/NOAA/noaa_stations.shp"))
all_stations <- st_drop_geometry(noah_stations[,c("id", "name", "source", "height_est")])
all_stations$longitude <- st_coordinates(noah_stations)[,1]
all_stations$latitude <- st_coordinates(noah_stations)[,2]
#fwrite(all_stations, paste0(home, "Data/Buoy/buoy_info.csv"))
# Add the Swiss Datalakes and King County, WA buoys manually in excel 

#######################################################################################
#######################################################################################
# Identify the PLD ID that each buoy is located within 
#######################################################################################
#######################################################################################

# List PLD shapefiles 
pld_files <- list.files(paste0(home, "Data/SWOT_PLD"), full.names = TRUE, pattern = "*.shp$")
pld_files <- pld_files[grepl("pfaf", pld_files)]
pld_sub <- do.call(rbind, lapply(pld_files, st_read))

# convert all stations to a shapefile 
buoy_info <- fread(paste0(home, "Data/Buoy/buoy_info.csv"))
buoy_info_sf <- st_as_sf(buoy_info, coords=c("longitude", "latitude"), crs=4326)
# Spatial intersect to find the pld_id
int_mat <- st_intersects(buoy_info_sf, pld_sub, sparse = FALSE)

get_idx <- function(x){
    idx = which(x==1)
    if(length(idx) == 0){
        return(NA)
    }else{
        pld_id <- as.character(pld_sub[idx,]$lake_id)
        return(pld_id)
    }
}
pld_id <- unlist(apply(int_mat, 1, get_idx))
buoy_info$pld_id <- pld_id
buoy_info[id == "buchillonfieldstation",]$pld_id <- "2160046863"
buoy_info[id == "LMFS1",]$pld_id <- "7320411603"
fwrite(buoy_info, paste0(home, "Data/Buoy/buoy_info.csv"))

#######################################################################################
#######################################################################################
# Create shapefile of all lakes included in the study  
#######################################################################################
#######################################################################################

pld_keep <- pld_sub[pld_sub$lake_id %in% buoy_info$pld_id,]
st_write(pld_keep, paste0(home, "Data/Buoy/pld_with_buoys.shp"))

# Range of lake area included in study 
min(pld_keep$ref_area)/1e6
max(pld_keep$ref_area)/1e6

hist(pld_keep$ref_area/1e6, main='Area (km2) of lakes with buoys', xlab='Area (km2)', ylab='Frequency', breaks=15)

#######################################################################################
#######################################################################################
# Exploration of buoy availability 
#######################################################################################
#######################################################################################

# Sumjmarize buoy sampling rate 
type_dt <- buoy_info[,.(type=.N), .(type)]
type_dt <- type_dt[c(7, 3, 1, 6, 5,2,4)]

# Check that the type of date is the same in all of the processed buoy data 
processed_files <- list.files(paste0(home, "Data/Buoy/Processed"), full.names=TRUE)
date_list <- c()
for(i in processed_files){
    dt <- fread(i)
    d <- as.character(dt$datetime[1])
    date_list <- c(date_list, d)
}
# It is :)

# Create a dt with the sampling rate and the number of wspd and wdir observations during 2023-2024 with the start and end date so far 
rows <- list()
for(i in 1:length(processed_files)){
    dt <- fread(processed_files[i])
    start_date <- as.character(dt$datetime[1])
    end_date <- as.character(dt$datetime[nrow(dt)])
    type <- buoy_info[id == dt$id[1]]$type
    N_wspd <- length(dt[!is.na(wspd)]$wspd)
    N_wdir <-  length(dt[!is.na(wdir)]$wspd)
    r <- data.table(id=dt$id[1], type=type, start=start_date, end=end_date,N_wpd=N_wspd,N_wdir=N_wdir)
    rows[[i]] <- r
}
rows <- rbindlist(rows)
View(rows)


#######################################################################################
#######################################################################################
# Get rid of buoy 45202 from the analysis because it only has 7 observations all on one day 
#######################################################################################
#######################################################################################

buoy_info <- fread(paste0(home, "Data/Buoy/buoy_info.csv"))
# Many buoys are in the PLD water body where this buoy is located
# So I don't need to change the pld_with_buoys.shp
buoy_info[pld_id == buoy_info[id == 45202]$pld_id,]

buoy_info <- buoy_info[!id == 45202]
fwrite(buoy_info, paste0(home, "Data/Buoy/buoy_info.csv"))

# Sumjmarize buoy sampling rate 
type_dt <- buoy_info[,.(type=.N), .(type)]
type_dt <- type_dt[c(7, 3, 1, 6, 5,2,4)]

buoy_info_sf <- st_as_sf(buoy_info, coords=c("longitude", "latitude"), crs=4326)
st_write(buoy_info_sf, paste0(home, "Data/Buoy/buoy_info.shp"), append=FALSE)



#######################################################################################
#######################################################################################
# Station 45135 got deleted somewhere along the way but it does exist 

# Download and process the data 
noah_stations <- fread(paste0(home, "Data/Buoy/NOAA/noah_national_buoy_center_active_station_list.csv"))
noah_stations <- st_as_sf(noah_stations, coords=c("lon", "lat"), crs=4326)
noah_stations$source <- "Noah National Data Buoy Center"
noah_stations <- noah_stations[noah_stations$id == "45135",]

# List PLD shapefiles 
pld_files <- list.files(paste0(home, "Data/SWOT_PLD"), full.names = TRUE, pattern = "*.shp$")
pld_files <- pld_files[grepl("pfaf_07", pld_files)]

# Check if buoys within PLD water body
pld_int <- function(x){
    "
    Test if buoy inside PLD
    x: sf df of pld sub dataset 
    return sf df of noaa stations that intersected with pld 
    "
    pld_sub <- st_read(pld_files[x])
    int_mat <- st_intersects(noah_stations, pld_sub, sparse = FALSE)
    noah_stations[rowSums(int_mat) >= 1, ]
}
sf_use_s2(FALSE)
station_45135 <- do.call(rbind, lapply(1:length(pld_files), pld_int))
process_noaa(station_45135[1,])


# Add it to the buoy_info.shp and buoy_info.csv 
buoy_info_45135 <- fread(paste0(home, "Data/Buoy/buoy_info.csv"))
buoy_info_45135 <- buoy_info_45135[id == "45135"]
buoy_info_45135$latitude <- 43.780
buoy_info_45135$longitude <- -76.870
buoy_info_45135 <- st_as_sf(buoy_info_45135, coords=c("longitude", "latitude"), crs=4326)
buoy_info_45135 <- buoy_info_45135[,2:ncol(buoy_info_45135)]
buoy_info_45135$pld_id <- as.character(buoy_info_45135$pld_id)

buoy_info_sf <- st_read(paste0(home, "Data/Buoy/buoy_info.shp"))
buoy_info_sf <- buoy_info_sf[!buoy_info_sf$id == "45135",]
colnames(buoy_info_45135) <- colnames(buoy_info_sf)
buoy_info_sf <- rbind(buoy_info_sf, buoy_info_45135)
st_write(buoy_info_sf, paste0(home, "Data/Buoy/buoy_info.shp"), append=FALSE)