'''
adding neighbor features to time-resampled AIS dataframe (universal clock)
this builds on functions_basic_sak.df_time_resample
and GriSPy
'''
import os.path
import numpy as np
import functions_basic_sak as funsak
import pandas as pd
from tqdm import tqdm
import AIS_parameters
import inspect
import pathlib
import re
import AIS_interp
from grispy import GriSPy
from AIS_GriSpy import haversine4GriSpy_radians
import AIS_GriSpy

fname = os.path.split( inspect.getfile(inspect.currentframe()) )[1]
st_mtime = pathlib.Path(fname).stat().st_mtime

pd.set_option('display.max_columns', 20, 'display.width', 1500)
##
def get_neighborhood(df_resample,  n_nearest = None, upper_radius_m = None, pbar = None, N_cells = 100, prec_format='%.1G' ):
    df_resample['longitude radians'] = np.radians( df_resample['longitude'] )
    df_resample['latitude radians'] = np.radians(df_resample['latitude'])
    df_resample['neighbor count within %g m' % upper_radius_m] = 0
    arr_count = np.zeros(  df_resample.shape[0] ).astype(np.int16)
    arr_dist = np.tile( np.NaN, (df_resample.shape[0], n_nearest ) ).astype(np.float16)
    arr_mmsi = -np.ones( ( df_resample.shape[0], n_nearest ) ).astype(np.int32)
    ##
    if pbar:
        pbar_ = df_resample.groupby('timestamp')
    else:
        pbar_ = tqdm(df_resample.groupby('timestamp'),
                    desc='%s: n_nearest = %i, upper_radius_m = %g' % ( fname,  n_nearest, upper_radius_m))
        pbar = pbar_
    distance_upper_bound = upper_radius_m/funsak.earth_radius_m
    N_cells_dict = dict()
    for timestamp_, df_ in pbar_:
        pbar.set_postfix({'timestamp': timestamp_, '# rows': df_.shape[0]})
        ##
        lon_lat = df_[['longitude radians', 'latitude radians']].to_numpy(float)
        ##
        #use optimal N_cells in GriSPy, which is about 200 for 1E4 rows in lon_lat with MarineCadastre
        approx_len_lon_lat = funsak.extract_int(prec_format % lon_lat.shape[0])
        if not approx_len_lon_lat in N_cells_dict:
            try:
                N_cells_dict[approx_len_lon_lat] = int( AIS_GriSpy.best_N_cells_self_neighbors(len_data=lon_lat.shape[0],
                                            distance_upper_bound_m=upper_radius_m, prec_format=prec_format, calculate=False ).index.get_level_values(2).to_numpy()[0] )
            except:
                N_cells_dict[approx_len_lon_lat] = 200
        ##
        grid = GriSPy(lon_lat, N_cells=N_cells_dict[approx_len_lon_lat], periodic={0: (-np.pi, np.pi), 1: (-np.pi / 2, np.pi / 2)},
                      metric=haversine4GriSpy_radians)
        bubble_dist, bubble_ind = grid.bubble_neighbors(lon_lat, distance_upper_bound=distance_upper_bound, sorted=True)
        ##
        for r, dist, ind in zip( range(len(bubble_dist)), bubble_dist, bubble_ind ):
            n = min( len(dist)-1, n_nearest )
            if n > 0:
                ##
                arr_count[r] = len(dist)-1
                arr_dist[r, :n] = dist[1:n+1]
                arr_mmsi[r, :n] = df_['mmsi'].to_numpy(int)[ind[1:n+1]]
        ##
    df_neighbors = pd.concat( [pd.DataFrame( arr_dist*funsak.earth_radius_m, columns = [ '%i dist m'%i for i in range( n_nearest )]),
                               pd.DataFrame( arr_mmsi, columns=['%i mmsi' % i for i in range(n_nearest)]),
                               pd.Series( arr_count, name = '# within %.3G m'%upper_radius_m)], axis = 1 )
    return df_neighbors
##
# def neighborhood( lon_lat_radians, lon_lat_centers_radians=None, n_nearest=None, upper_radius= .0001 ):
def get_df( date = '2022_06_29', time = None, freq = None, n_nearest = 5, upper_radius_m = 5, format = 'feather',
            pbar = None, multicolumn = False, nrows = None, overwrite = False, join = None ):
    '''

    Parameters
    ----------
    date: either string or list of strings representing desired date(s), e.g. '2022_06_29' or ['2022_06_29', '2021_06_29', '2020_06_29']
    format: 'feather', 'h5' or 'pkl'
    MarineCadaster_URL: this is intended for Marinecadastre for now, will update with Danish_URL later on.
    download_dir

    Returns either single df or dict of desired dataframes
    -------

    '''
    ##
    if isinstance(date, list):
        df_dict = {}
        filename_dict = {}
        pbar = tqdm( date, desc = '%s get_df'%fname )
        for date_ in pbar:
            pbar.set_postfix( {'date': date_} )
            df_dict[date_], filename_dict[date_] = get_df( date_, freq= freq, n_nearest = n_nearest, upper_radius_m = upper_radius_m, format = format,
               overwrite=overwrite, pbar=pbar, multicolumn=multicolumn, nrows=nrows, join = join )
        return df_dict, filename_dict
    else:
        date_ = pd.to_datetime(re.sub( '_', ' ', date))
        ##
        out_file = os.path.join( AIS_parameters.dirs['root'], 'downsample', '%s (%s %s %i %g).%s'%(date_.strftime('%Y_%m_%d'), fname[:-3], freq, n_nearest, upper_radius_m, format) )
        if not os.path.isdir( os.path.split( out_file)[0] ):
            os.mkdir(  os.path.split( out_file)[0] )

        if time is None:
            time = pd.date_range(date_, date_ + pd.to_timedelta(1, 'd'), freq = freq )[:-1]
        df = None
        if os.path.exists(out_file) and not overwrite and ( pd.read_feather(out_file).columns[0] == 'empty' ):
            df_neighbors = pd.read_feather(out_file)
            ##
        else:
            pd.DataFrame([0], columns=['empty']).to_feather(out_file)

            df = AIS_interp.interpolator().get_df( time=time, columns = ['timestamp', 'mmsi', 'latitude', 'longitude'],
                  only_valid = True, pbar=pbar)

            df_neighbors = get_neighborhood( df,
                         n_nearest = n_nearest, upper_radius_m = upper_radius_m, pbar = pbar )
            df_neighbors.to_feather( out_file )
        ##
        if not join is None:
            if df is None:
                df = AIS_interp.interpolator().get_df( time=time,
                                                      columns=['timestamp', 'mmsi', 'latitude', 'longitude'],
                                                      only_valid=True, pbar=pbar)
            if join ==True:
                df_neighbors = pd.concat([df, df_neighbors], axis=1)
        if multicolumn:
            df_neighbors.columns = pd.MultiIndex.from_product([[fname[:-3]], df_neighbors.columns])
        return df_neighbors, out_file
        ##
##
# check why mmsi # missing when there is a distance
if __name__ == '__main__':
    if 'stephan' == AIS_parameters.user:
        df, file = get_df( date = '2022_01_01', freq = 'H',  n_nearest = 3, upper_radius_m = 5E3, nrows = None, overwrite = True, join = False )
        print(df)
##
