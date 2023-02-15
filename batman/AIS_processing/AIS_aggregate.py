import AIS_neighbors
import AIS_geopandas_tagging
import pandas as pd
import import_AIS
import functions_basic_sak as funsak
import trajectory
from tqdm import tqdm
from skimage import data, util, measure
from scipy.ndimage.measurements import label
import AIS_parameters


pd.set_option('display.max_columns', 10, 'display.width', 1500)

##
date = '2020_02_02'
df = dict()
filename = dict()
df['AIS_geopandas_tagging'], filename['AIS_geopandas_tagging'] = AIS_geopandas_tagging.get_df(date, tagging = { 'Oil_Gas_Pipelines': { 'tags': 'PipelineName', 'geometry': 'geometry', 'buffer': 5E3 },
              'world_ports': {'tags': ['portname', 'geonameid'], 'geometry': 'lat_lon', 'buffer': 10E3 }}, overwrite = False, freq = None )
##
df['import_AIS'], filename['import_AIS'] = import_AIS.get_df(date, multicolumn=True)
##
import AIS_numpy_tagging
df['AIS_numpy_tagging'], filename['AIS_numpy_tagging'] = AIS_numpy_tagging.get_df(date=date,
              files = ['shipping_intensity', 'coast_distance_km', 'earth_elevation_m'], multicolumn=True )
##
df_concat = pd.concat( df.values(), axis = 1 )
df_concat.columns = pd.MultiIndex.from_tuples( df_concat.columns.to_list() )
##
