##########################################################################################
# Katie Mcquillan
# 04/10/2024
# Download Sentinel-1 data from copernicus at buoy locations from 2023 - 2024
# https://medium.com/rotten-grapes/download-sentinel-data-using-python-from-copernicus-9ec0a789e470
##########################################################################################

# Import packages
import os
from datetime import date, timedelta
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from time import sleep

# Set home directory 
home = "C:/Users/kmcquil/Documents/SWOT_WIND/"

###########################################################################################
# Import buoy locations 
buoy_info_shp = gpd.read_file(os.path.join(home, "Data/Buoy/buoy_info.shp"))
# Buffer to get a square 
buoy_info_shp = buoy_info_shp.buffer(0.05, 1)

##########################################################################################
# Set up inputs

copernicus_user = 'kmcquil@vt.edu' # data hub username
copernicus_password = 'Babyraccoon1!' # data hub password
data_collection = "SENTINEL-1"

#start_string = '2023-04-08'
#end_string = '2023-04-26'

start_string = '2023-01-01'
end_string = '2024-04-10'


############################################################################################

def get_keycloak(username: str, password: str) -> str:
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    try:
        r = requests.post(
            "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
            data=data,
        )
        r.raise_for_status()
    except Exception as e:
        raise Exception(
            f"Keycloak token creation failed. Reponse from the server was: {r.json()}"
        )
    return r.json()["access_token"]

def get_images(data_collection, ft, start_string, end_string):
    
    try:
        json_ = requests.get(
            f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq '{data_collection}' and OData.CSC.Intersects(area=geography'SRID=4326;{ft}') and ContentDate/Start gt {start_string}T00:00:00.000Z and ContentDate/Start lt {end_string}T00:00:00.000Z&$count=True&$top=1000"
            #f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq '{data_collection}' and \
             #Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'transmitterReceiverPolarisation' and att/OData.CSC.StringAttribute/Value eq 'VV VH') and \
             #OData.CSC.Intersects(area=geography'SRID=4326;{ft}') and ContentDate/Start gt {start_string}T00:00:00.000Z and ContentDate/Start lt {end_string}T00:00:00.000Z&$count=True&$top=1000"
        ).json() 
        p = pd.DataFrame.from_dict(json_["value"]) # Fetch available dataset
        return p
    except:
        sleep(300)
        json_ = requests.get(
            f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq '{data_collection}' and OData.CSC.Intersects(area=geography'SRID=4326;{ft}') and ContentDate/Start gt {start_string}T00:00:00.000Z and ContentDate/Start lt {end_string}T00:00:00.000Z&$count=True&$top=1000"
        ).json() 
        p = pd.DataFrame.from_dict(json_["value"]) # Fetch available dataset
        return p

p_list = []
for ft in buoy_info_shp:
    p_list.append(get_images(data_collection, ft, start_string, end_string))
p = pd.concat(p_list)
p = p[p["Name"].str.contains('IW_GRDH')] # Only keep IW GRD High resolution 
p = p[~p["Name"].str.contains('COG')] # Don't include cog  


p["geometry"] = p["GeoFootprint"].apply(shape)
productDF = gpd.GeoDataFrame(p).set_geometry("geometry") # Convert PD to GPD
productDF["identifier"] = productDF["Name"].str.split(".").str[0]

for index,feat in enumerate(productDF.iterfeatures()):
    print(index)
    # if the file exists, skip it 
    if os.path.exists(f"{home}Data/Sentinel1/SAFE/{feat['properties']['identifier']}.zip") == True: 
        print('already exists')
        continue
    
    # download 
    try:
        session = requests.Session()
        keycloak_token = get_keycloak(copernicus_user,copernicus_password)
        session.headers.update({"Authorization": f"Bearer {keycloak_token}"})
        url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({feat['properties']['Id']})/$value"
        response = session.get(url, allow_redirects=False)
        while response.status_code in (301, 302, 303, 307):
            url = response.headers["Location"]
            response = session.get(url, allow_redirects=False)
        print(feat["properties"]["Id"])
        file = session.get(url, verify=False, allow_redirects=True)

        with open(
            f"{home}Data/Sentinel1/SAFE/{feat['properties']['identifier']}.zip", #location to save zip from copernicus 
            "wb",
        ) as p:
            print(feat["properties"]["Name"])
            p.write(file.content)
    except:
        print("problem with server")




















###########################################################################################
# I could not get the geometry search term to work correctly
# Instead, find images and then filter for buoy location myself 

# Make a query - specify sentinel-1, interfermetric wide mode, grd processed, VV polarization, 10m resolution 
collection = "Sentinel1"
search_terms = {
    "maxRecords": "2000",
    "startDate": "2023-01-01",
    "completionDate": "2023-02-01",
    "processingLevel": "LEVEL1",
    "sensorMode": "IW",
    "productType": "IW_GRDH_1S",
    "polarisation": "VV"
    #"geometry":buoys
}
features = list(query_features(collection, search_terms))
len(features)































###########################################################################################
# Buoy is a point but we need a rectangle.Make a buffer around each point and save 
# Import buoy locations 
buoy_info_shp = gpd.read_file(os.path.join(home, "Data/Buoy/buoy_info.shp"))
# Buffer and save
buoy_buffered = buoy_info_shp.iloc[54:55, :].buffer(3, 1)
buoy_buffered.to_file(os.path.join(home, "Data/Buoy/buoys_buffered.shp"), append=False)

###########################################################################################
# Start just trying at one buoy 
buoys = shape_to_wkt(os.path.join(home, "Data/Buoy/buoys_buffered.shp"))

# Make a query - specify sentinel-1, interfermetric wide mode, grd processed, VV polarization, 10m resolution 
collection = "Sentinel1"
search_terms = {
    "maxRecords": "2000",
    "startDate": "2023-01-01",
    "completionDate": "2024-04-10",
    "processingLevel": "LEVEL1",
    "sensorMode": "IW",
    "productType": "IW_GRDH_1S",
    "polarisation": "VV",
    "geometry":buoys
}
features = list(query_features(collection, search_terms))



# Extract the buoy long and lat 
buoy_coords = buoy_info_shp.iloc[0:1, :].get_coordinates()
long = buoy_coords['x'][0]
lat = buoy_coords['y'][0]

test = shape_to_wkt(os.path.join(home, "Data/Buoy/buoy_info.shp"), appe)

# Make a query - specify sentinel-1, interfermetric wide mode, grd processed, VV polarization, 10m resolution 
collection = "Sentinel1"
search_terms = {
    "maxRecords": "2000",
    "startDate": "2023-01-01",
    "completionDate": "2024-04-10",
    "processingLevel": "LEVEL1",
    "sensorMode": "IW",
    "productType": "IW_GRDH_1S",
    "polarisation": "VV",
    #"geometry":footprint
    "lon": -87,
    "lat": 48,
}
features = list(query_features(collection, search_terms))

# Buoy is a point but we need a rectangle. So make a small box. 
buoy_coords = buoy_info_shp.iloc[0:1, :].get_coordinates()
long = buoy_coords['x'][0]
lat = buoy_coords['y'][0]
long_left = long - 0.001
long_right = long + 0.001
lat_up = lat + 0.001
lat_down = lat - 0.001
footprint = {
    "coordinates":[[[long_left, lat_down], [long_left, lat_up], [long_right, lat_down], [long_right, lat_up]]],
    "type":"Polygon"
}
footprint = shape(footprint)
# Convert to wkt format
footprint = footprint.wkt

# Make a query - specify sentinel-1, interfermetric wide mode, grd processed, VV polarization, 10m resolution 
collection = "Sentinel1"
search_terms = {
    "maxRecords": "2000",
    "startDate": "2023-01-01",
    "completionDate": "2024-04-10",
    "processingLevel": "LEVEL1",
    "sensorMode": "IW",
    "productType": "IW_GRDH_1S",
    "polarisation": "VV",
    #"geometry":footprint
}
features = list(query_features(collection, search_terms))



# Use the sentinel api to download 
api = SentinelAPI('kmcquil@vt.edu', 'Babyraccoon1!', 'https://apihub.copernicus.eu/apihub')



# search by polygon, time, and SciHub query keywords

# Convert my buoy to small polygon and make that a wkt 
buoy_info_shp = gpd.read_file(os.path.join(home, "Data/Buoy/buoy_info.shp"))
buoy_coords = buoy_info_shp.iloc[0:1, :].get_coordinates()
long = buoy_coords['x'][0]
lat = buoy_coords['y'][0]
long_left = long - 0.001
long_right = long + 0.001
lat_up = lat + 0.001
lat_down = lat - 0.001
footprint = {
    "coordinates":[[[long_left, lat_down], [long_left, lat_up], [long_right, lat_down], [long_right, lat_up]]],
    "type":"Polygon"
}
footprint = shape(footprint)
footprint = footprint.wkt

products = api.query(footprint,
                     date=(date(2023, 1, 1), date(2024, 4, 10)),
                     platformname='Sentinel-1')

products = api.query(footprint,
                     date=(date(2023, 1, 1), date(2024, 4, 10)),
                     platformname='Sentinel-1', 
                     productType='GRD',
                     instrument='SAR',
                     polarisationMode='VV')

# convert to Pandas DataFrame
products_df = api.to_dataframe(products)

# instrument name=SAR
# product type = GRD
# polarization = VV 








# Write API queries to list all of the images that fit my criteria for each buoy
# Thre will definitely overlap since many buoys located close together
# Filter to the unique queries and 







# Do this for just one buoy to start 
# https://scihub.copernicus.eu/userguide/BatchScripting
buoy_info_shp = gpd.read_file(os.path.join(home, "Data/Buoy/buoy_info.shp"))

buoy_coords = buoy_info_shp.iloc[0:1, :].get_coordinates()
long = buoy_coords['x'][0]
lat = buoy_coords['y'][0]

buoy_info_shp.iloc[0:1,:].to_wkt()

d = '' # url of the data hub service to be polled
u = 'kmcquil@vt.edu' # data hub username
p = 'Babyraccoon1!' # data hub password
m = 'Sentinel-1' # sentinel mission name
i = 'SAR' # instrument name
s = '2023-01-01T00:00:00.000Z' # ingestation date from
e = '2024-04-10T00:00:00.000Z' # ingestion date to 
S = '2023-01-01T00:00:00.000Z' # sensing date from
E = '2024-04-10T00:00:00.000Z' # sensing date to 
c = [long, lat, long, lat]
T = 'GRD'

