import pandas as pd
import numpy as np
import functions_basic_sak as funsak
import inspect
import os
import pathlib
import import_AIS
import AIS_parameters
from tqdm import tqdm
import AIS_downsample
pd.set_option('display.max_columns', 20, 'display.width', 1500)

fname = os.path.split( inspect.getfile(inspect.currentframe()) )[1]
st_mtime = pathlib.Path(fname).stat().st_mtime

arrs = None

fnc = lambda arr, latitude, longitude: arr[
    np.round(np.interp(latitude, [-90, 90], [arr.shape[0], 0])).astype(int),
    np.round(np.interp(longitude, [-180, 180], [0, arr.shape[1]])).astype(int)]


def get_df( date = None, df = None, freq = None, multicolumn = False, files = ['shipping_intensity', 'coast_distance_km', 'earth_elevation_m'], pbar = None ):
    if isinstance( date, list ):
        return { date_: get_df(date_, files = files, freq = freq ) for date_ in tqdm( date, desc = fname )}
    if df is None:
        if freq:
            df, _ = AIS_downsample.get_df(date, freq = freq )
        else:
            df, _ = import_AIS.get_df(date)

    vals = {}
    global arrs
    if arrs is None:
        arrs = {file: np.load(AIS_parameters.files[file]) for file in files}
    if pbar:
        pbar.set_postfix_str( fname )
    for name, arr in arrs.items():
        vals[name] = np.array(fnc( arr, df['latitude'].to_numpy(float),  df['longitude'].to_numpy(float) ))

    df_  = pd.DataFrame(vals)
    if multicolumn:
        df_.columns = pd.MultiIndex.from_product( [[fname[:-3]], df_.columns] )
    return df_, None

if __name__ == '__main__':
    if 'stephan' == AIS_parameters.user:
        # print( get_df(date='2017_06_29', files = ['shipping_intensity', 'coast_distance_km', 'earth_elevetion_m'] ) )
        dates = ['2020_02_01', '2020_02_02']
        date = '2022_01_01'
        print( get_df(date=date, files = ['shipping_intensity', 'coast_distance_km', 'earth_elevation_m'], freq = 'H' ) )