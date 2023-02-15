import numpy as np
import functions_basic_sak as funsak
import AIS_import
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
us_coastline = geopandas.read_file(None)

# world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
##
# url = "https://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_20m.json"
#
# states = geopandas.read_file(url)
##
timestamp = '2022-02-14 12:00'
dfs = {date: AIS_import.get_df(date=date)[0] for date in pd.date_range('2022-01-01', '2022-01-11')}
##
radius_m = 50E3
time_interval_s = 600
lon, lat = -80.1164+.2, 26.0854
lon, lat = -118.2877, 33.78
tmp = df[ np.abs(df['timestamp'] - pd.to_datetime(timestamp)).dt.total_seconds() < time_interval_s ]
dist_m = funsak.distance_haversine(lon1 = lon, lat1 = lat, lon2 = tmp.longitude, lat2 = tmp.latitude, convert2radians=True)
tmp = tmp[dist_m < radius_m ]

gdf = geopandas.GeoDataFrame(
    tmp.drop( columns=['latitude', 'longitude']), geometry=geopandas.points_from_xy(tmp.longitude, tmp.latitude))

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
us_coastline.plot(ax=ax, cmap='winter', legend = 'US coastline')

gdf.plot(markersize = 4, color = 'r', ax = ax)
ax.set_xlim([tmp.longitude.min(), tmp.longitude.max()])
ax.set_ylim([tmp.latitude.min(), tmp.latitude.max()])
plt.show()
##

