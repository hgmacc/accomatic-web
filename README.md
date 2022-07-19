    if input() != 1: sys.exit(0)

Geotop takes a normal date
YY/MM/DD HH:MM
Origin is = 1970 (unix time)
ERDAPP format

How to get sites from ERDAPP
Only has Ekati data
Download some Ekati data from erddap and see if the .nc structures play well with GTPEM

Database has tables for "field" or "column" sort and filter super fast
Index the last recorded time stamp

50 cm
Take blue from .5
Time-slice ID: years, seasons, monthly, snow binary, entire dataset("all data"), "all_data_winter"
Pre-configure
    
add obs in netcdf

add a group called meta = what is the raw data this is based on?

Meta = point to model and obs

terrain is an attribute of terrain

Terrain = everyone has their own ideas

Additional table to describe the site

Terrain types are not defined by



# How to

```
source accomatic-web/bin/activate
cd accomatic-web/accomatic
python3 /home/hma000/accomatic-web/accomatic/accomatic_run.py
```

## Data Export

```
q <- "select name, ST_Y(coordinates) AS lat, ST_X(coordinates) AS lon, elevation_in_metres, comment from locations where name similar to 'LDG%[0-9]{2}'"
loc <- dbGetQuery(con, q)

# Generate siteslist.csv for downscaling / interpolating in GlobSim
write.csv(loc, </output_csv_path.csv/>)

# Get observations for all sites from DB
dbpf_export_nc_surface(con, loc$name, "obs.nc", freq="hourly")
```

## This is how we get location names from the LDG KML file. 

```
import geopandas as gpd
import fiona

# file_path: /fs/yedoma/usr-storage/hma000/LDG/

gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
df = gpd.read_file('/fs/yedoma/usr-storage/hma000/LDG/LacDeGras.kml', driver='KML')

df['lat'] <- NA
df['lon'] <- NA
df['elevation_in_metres'] <- NA
df['comment'] <- NA


for (row in 1:nrow(df)) {
    q <- paste0("select name, ST_Y(coordinates) AS lat, ST_X(coordinates) AS lon, elevation_in_metres, comment from locations where name = '", df[row, "Name"], "'")
    tmp <- dbGetQuery(con, q)
    
    df[row, "lat"] = tmp$lat
    df[row, "lon"] = tmp$lon    
    df[row, "elevation_in_metres"] = tmp$elevation_in_metres
    df[row, 'comment'] <- gsub(".*;","", tmp$comment)
}
```





