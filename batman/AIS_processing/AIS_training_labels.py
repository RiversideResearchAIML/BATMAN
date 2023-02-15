import numpy as np
import functions_basic_sak as funsak
import pandas as pd
from tqdm import tqdm
import os
from skimage import data, util, measure
from scipy.ndimage.measurements import label
import inspect
import AIS_downsample
import AIS_neighbors
import AIS_parameters
import AIS_trajectory
import AIS_geopandas_tagging
import AIS_numpy_tagging
import AIS_loitering
import re
import import_AIS
fname = os.path.split( inspect.getfile(inspect.currentframe()) )[1]

date_str = '2022_01_01'
df_master = pd.read_feather( os.path.join( AIS_parameters.dirs['root'], 'training', 'AIS_harvest %s.feather'%date_str) )
print(df_master)
