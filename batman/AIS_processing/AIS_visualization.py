import pandas as pd
# from tqdm import tqdm
# import os
# import inspect
# import AIS_parameters
# import numpy as np
# import AIS_harvest
# import datetime
# import AIS_description
# import matplotlib.pyplot as plt
# from skimage import data, util, measure
# from scipy.ndimage.measurements import label
import re
behavior_file = None
downsample_file = None
downsample_file = re.sub( '\(AIS_behaviors.*',  '(AIS_harvest H).feather', behavior_file )

import geopandas
df = pd.read_feather( None)
tmp = df[df['mmsi']==1072211352]
gdf = geopandas.GeoDataFrame(
    tmp.drop( columns=['latitude', 'longitude']), geometry=geopandas.points_from_xy(tmp.longitude, tmp.latitude))
##
import matplotlib.pyplot as plt
import geopandas
from shapely.geometry import Polygon

capitals = geopandas.read_file(geopandas.datasets.get_path("naturalearth_cities"))
world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))

# Create a subset of the world data that is just the South American continent
south_america = world[world["continent"].isin(["South America", "North America"])]

# Create a custom polygon
polygon = Polygon([(0, 0), (0, 90), (180, 90), (180, 0), (0, 0)])
poly_gdf = geopandas.GeoDataFrame([1], geometry=[polygon], crs=world.crs)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
world.boundary.plot(ax=ax1)
gdf.plot(ax = ax1)
# plt.show()

poly_gdf.boundary.plot(ax=ax1, color="red")
south_america.boundary.plot(ax=ax2, color="green")
capitals.plot(ax=ax2, color="purple")
ax1.set_title("All Unclipped World Data", fontsize=20)
ax2.set_title("All Unclipped Capital Data", fontsize=20)
ax1.set_axis_off()
ax2.set_axis_off()
plt.show()
##

polygon = Polygon([(0, 0), (0, 90), (180, 90), (180, 0), (0, 0)])
world_clipped = world.clip(polygon)

# Plot the clipped data
# The plot below shows the results of the clip function applied to the world
# sphinx_gallery_thumbnail_number = 2
fig, ax = plt.subplots(figsize=(12, 8))
world_clipped.plot(ax=ax, color="purple")
world.boundary.plot(ax=ax)
poly_gdf.boundary.plot(ax=ax, color="red")
ax.set_title("World Clipped", fontsize=20)
ax.set_axis_off()
plt.show()

capitals_clipped = capitals.clip(south_america)

# Plot the clipped data
# The plot below shows the results of the clip function applied to the capital cities
fig, ax = plt.subplots(figsize=(12, 8))
capitals_clipped.plot(ax=ax, color="purple")
south_america.boundary.plot(ax=ax, color="green")
ax.set_title("Capitals Clipped", fontsize=20)
ax.set_axis_off()
plt.show()

