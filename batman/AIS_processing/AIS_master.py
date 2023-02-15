import neighbors
import loitering
import old_ports
import elevation
import coast_distance
import shipping_routes
import pandas as pd
import functions_basic_sak as funsak
import trajectory
from tqdm import tqdm
from skimage import data, util, measure
from scipy.ndimage.measurements import label

pd.set_option('display.max_columns', 10, 'display.width', 1500)

file = None
freq = 'H'
df = funsak.AIS_rename(pd.read_feather(file))
##
we = elevation.world_elevation(ESRI_stride = 2)
df = pd.concat( [trajectory.add(df), we.elevation(df=df), coast_distance.distance_km(df=df), shipping_routes.intensity(df=df)], axis = 1 )
##
df_agg = funsak.df_time_aggregate(df, freq= freq, columns = ['delta sec', 'delta km'])
##
df_resample = funsak.df_time_resample( df, freq = freq).reset_index()
##

df_append = [df_resample]
df_append.append( neighbors.neighbors(df = df_resample, n_nearest=2) )
df_append.append( loitering.loitering(df = df_resample, radii_km=[5]) )
df_append.append( ports.port_neighborhood(df = df_resample, file_world_ports = None, n_nearest=1) )
##
df_ = pd.concat( df_append, axis = 1)
df_both = [df_agg.merge( df_resample, how = 'outer', on = ['timestamp', 'mmsi'])]
##

