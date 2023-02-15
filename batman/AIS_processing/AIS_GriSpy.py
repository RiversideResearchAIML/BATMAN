import matplotlib.pyplot as plt
import numpy as np
from grispy import GriSPy
import datetime
from tqdm import tqdm
# import import_AIS
import os
import AIS_parameters
import import_AIS
import pandas as pd
import inspect
import itertools
import functions_basic_sak as funsak

fname = os.path.split( inspect.getfile(inspect.currentframe()) )[1]

convert2list = lambda val: val if isinstance( val, list) else list(val)
def haversine4GriSpy_radians(c0, centres, dim):
    """Distance using the haversine formulae.
    The haversine formula determines the great-circle distance between two
    points on a sphere given their longitudes and latitudes. Important in
    navigation, it is a special case of a more general formula in spherical
    trigonometry, the law of haversines, that relates the sides and angles of
    spherical triangles. More info:
    https://en.wikipedia.org/wiki/Haversine_formula
    """
    # lon1 = np.deg2rad(c0[0])
    # lat1 = np.deg2rad(c0[1])
    # lon2 = np.deg2rad(centres[:, 0])
    # lat2 = np.deg2rad(centres[:, 1])
    # in radians!
    lon1 = c0[0]
    lat1 = c0[1]
    lon2 = centres[:, 0]
    lat2 = centres[:, 1]

    sdlon = np.sin((lon2 - lon1) / 2.0)
    sdlat = np.sin((lat2 - lat1) / 2.0)
    clat1 = np.cos(lat1)
    clat2 = np.cos(lat2)
    num1 = sdlat ** 2
    num2 = clat1 * clat2 * sdlon ** 2
    sep = 2 * np.arcsin(np.sqrt(num1 + num2))
    return sep

def runtime_bubble_neighbors(len_data, distance_upper_bound, N_cells, df = None ):
    if df is None:
        rng = np.random.default_rng(seed=0)
        lon_lat = np.concatenate(
            [rng.uniform(-np.pi, np.pi, size=(len_data, 1)), rng.uniform(-np.pi / 2, np.pi / 2, size=(len_data, 1))],
            axis=1)
    else:
        lon_lat = np.radians( df[['longitude', 'latitude']].sample( len_data ).to_numpy(float) )

    start = datetime.datetime.now()
    grid = GriSPy(lon_lat, N_cells=N_cells, periodic={0: (-np.pi, np.pi), 1: (-np.pi / 2, np.pi / 2)},
                  metric=haversine4GriSpy_radians)
    bubble_dist, bubble_ind = grid.bubble_neighbors(lon_lat, distance_upper_bound=distance_upper_bound, sorted=True)
    return (datetime.datetime.now() - start).total_seconds()


##
def df_self_neighbors( date_str = None ):
    if date_str is None:
        out_file = os.path.join( AIS_parameters.dirs['scratch'], fname[:-3], 'parametric self-neighbors.csv' )
    else:
        out_file = os.path.join(AIS_parameters.dirs['scratch'], fname[:-3], 'parametric self-neighbors %s.csv'%date_str)
    if not os.path.isdir(os.path.split(out_file)[0]):
        os.mkdir(os.path.split(out_file)[0])
    if os.path.exists( out_file ):
        df = pd.read_csv( out_file ).set_index( ['len_data', 'distance_upper_bound', 'N_cells'])
    else:
        df = pd.DataFrame( columns = ['len_data', 'distance_upper_bound', 'N_cells', 'runtime']).set_index( ['len_data', 'distance_upper_bound', 'N_cells'])
    return df, out_file


def param_self_neighbors( len_data = np.power( 10, [1,2,3,4,5]),
    distance_upper_bound_m = [1E3, 3E3, 5E3, 1E4],
    N_cells = range( 20, 300, 10), date_str = None  ):
    df, out_file = df_self_neighbors(date_str)
    if date_str is None:
        df_AIS = None
    else:
        df_AIS = import_AIS.get_df( date_str )[0]
    pbar = tqdm(itertools.product(len_data, distance_upper_bound_m, N_cells ), total = len(len_data)*len(distance_upper_bound_m)*len(N_cells))
    for len_data_, distance_upper_bound_m_, N_cells_ in pbar:
        if not df.index.isin( [(len_data_, distance_upper_bound_m_, N_cells_)]).any():
            pbar.set_postfix({'len_data_':len_data_, 'distance_upper_bound_m_': distance_upper_bound_m_, 'N_cells_': N_cells_})
            df.loc[(len_data_, distance_upper_bound_m_, N_cells_), 'runtime'] = runtime_bubble_neighbors(len_data_, distance_upper_bound_m_/funsak.earth_radius_m, N_cells_, df_AIS )
            df.to_csv( out_file )

##
def best_N_cells_self_neighbors( len_data = 1E5,
    distance_upper_bound_m = 1E4,
    N_cells_range = range(10, 200, 10),
    prec_format = '%.1G',
    date_str = '2022_01_01',
    calculate = False):
    ##
    df, out_file = df_self_neighbors(date_str = date_str)
    ##
    len_data = funsak.extract_int(prec_format%len_data)
    distance_upper_bound_m = funsak.extract_int(prec_format % distance_upper_bound_m)
    param_dist = np.hypot(np.log10( df.index.get_level_values(0) / len_data ), np.log10( df.index.get_level_values(1) / distance_upper_bound_m) )
    closest_param_dist = np.min( param_dist )
    best_time = df.loc[param_dist==closest_param_dist, 'runtime'].min()
    if (closest_param_dist > np.log10(2)) and calculate:
        pbar = tqdm( N_cells_range )
        for N_cells_ in pbar:
            if not df.index.isin([(len_data, distance_upper_bound_m, N_cells_)]).any():
                pbar.set_postfix(
                    {'len_data_': len_data, 'distance_upper_bound_m_': distance_upper_bound_m, 'N_cells_': N_cells_})
                df.loc[(len_data, distance_upper_bound_m, N_cells_), 'runtime'] = runtime_bubble_neighbors(len_data,
                                                                                                             distance_upper_bound_m / funsak.earth_radius_m,
                                                                                                             N_cells_)
                df.to_csv(out_file)
        param_dist = np.hypot(np.log10(df.index.get_level_values(0) / len_data),
                              np.log10(df.index.get_level_values(1) / distance_upper_bound_m))
        closest_param_dist = np.min(param_dist)
        best_time = df.loc[param_dist == closest_param_dist, 'runtime'].min()
    return df.loc[( param_dist==closest_param_dist ) & ( df['runtime'] == best_time ),:]
##
def optimize_GriSpy( data = None, centres = None, len_data = None, lenX = 10, overwrite = False, range_N_cells = [20, 300, 10], n_nearest = 2 ):
    if data is None:
        rng = np.random.default_rng(seed=0)
        data = np.concatenate([rng.uniform(-np.pi, np.pi, size=(len_data, 1)), rng.uniform(-np.pi/2, np.pi/2, size=(len_data, 1))], axis = 1)
        centres = np.concatenate([rng.uniform(-np.pi, np.pi, size=(lenX*len_data, 1)), rng.uniform(-np.pi/2, np.pi/2, size=(lenX*len_data, 1))], axis = 1)
    else:
        if isinstance( data, pd.core.frame.DataFrame):
            data = np.radians( data[['longitude', 'latitude']] ).values
        len_data = int((data.shape[0] // np.power(10, np.floor(np.log10(data.shape[0])))) * np.power(10, np.floor(
            np.log10(data.shape[0]))))
        data = data[np.random.choice( data.shape[0], size = len_data, replace = False ).reshape(-1)]
        if isinstance( centres, pd.core.frame.DataFrame):
            centres = np.radians( centres[['longitude', 'latitude']] ).values
        centres = centres[np.random.choice( centres.shape[0], size = (lenX*data.shape[0], 1), replace = False ).reshape(-1)]
    assert data.shape[0] < centres.shape[0]
    data = data[:len_data, :]
    if centres.shape[0] > lenX*data.shape[0]:
        N = lenX*data.shape[0]
    else:
        N = centres.shape[0]
    filename = os.path.join( 'optimize_GriSpy', 'data.shape[0] %.3G centres.shape[0] %.3G n_nearest %g.npy'%( data.shape[0],  N, n_nearest ) )
    if not os.path.isdir( 'optimize_GriSpy' ):
        os.mkdir('optimize_GriSpy')
    if os.path.exists( filename ) and not overwrite:
        seconds = np.load( filename )
    else:
        seconds = np.tile( np.NaN, range_N_cells[1])
        pbar = tqdm( range(*range_N_cells), desc = filename )

        for N_cells in pbar:
            start = datetime.datetime.now()
            grid = GriSPy(data, N_cells = N_cells, periodic = {0: (-np.pi, np.pi), 1: (-np.pi/2, np.pi/2) }, metric = haversine4GriSpy_radians)
            bubble_dist, bubble_ind = grid.nearest_neighbors(
                centres, n=n_nearest )
            seconds[N_cells] = (datetime.datetime.now() - start).total_seconds()
            pbar.set_postfix({'N_cells': N_cells, 'best time': np.nanmin(seconds), 'best N_cells': np.nanargmin( seconds) })
            if ( sum(seconds[np.nanargmin(seconds):N_cells] > np.nanmin(seconds)) > 5 ):
                break
        np.save(filename, seconds )
    return np.nanargmin(seconds), seconds[np.nanargmin(seconds)]


def GriSpy_neighbors( date_str = '01-01-2022', lon_lat = None, n_nearest = 3, distance_upper_bound = .1, N_cells = 40, degrees = False  ):
    ##
    if lon_lat is None:
        df, filename = import_AIS.get_df(date_str)
        lon_lat = np.radians( df[['longitude', 'latitude']].values.to_numpy(float) )
    elif degrees:
        lon_lat = np.radians( lon_lat )
    if lon_lat.shape[0] < 100:
        N_cells = 2
    grid = GriSPy(lon_lat, N_cells=N_cells, periodic={0: (-np.pi, np.pi), 1: (-np.pi / 2, np.pi / 2)},
                  metric=haversine4GriSpy_radians)
    bubble_dist, bubble_ind = grid.bubble_neighbors( lon_lat, distance_upper_bound=distance_upper_bound, sorted=True )
    ##
    count = []
    dist_neighbors = []
    ind_neighbors = []
    for i in range( len(bubble_dist ) ):
        count.append(len( bubble_dist[i] ) )
        if len( bubble_dist[i] ) > n_nearest:
            ind_neighbors.append( bubble_ind[i][1:n_nearest+1] )
            dist_neighbors.append( bubble_dist[i][1:n_nearest + 1] )
        else:
            ind_neighbors.append(np.concatenate( [bubble_ind[i][1:], -np.ones( 1+n_nearest - len(bubble_ind[i] ), dtype=int)], axis = 0))
            dist_neighbors.append(np.concatenate( [bubble_dist[i][1:], np.tile(np.nan, 1+n_nearest - len(bubble_ind[i] ))], axis = 0))
    count = np.array( count )-1
    dist_neighbors = np.array( dist_neighbors )
    ind_neighbors = np.array(ind_neighbors)
    return count, dist_neighbors, ind_neighbors
    ##

def optimize_GriSpy_neighbors( data = None, n_nearest = 2, distance_upper_bound = .1, len_data = None, overwrite = False, range_N_cells = [20, 400, 10] ):
    filename = os.path.join( 'optimize_GriSpy_neighbors', 'data.shape[0] %.3G n_nearest %g distance_upper_bound %g.npy'%( data.shape[0], n_nearest, distance_upper_bound ) )
    if not os.path.isdir( 'optimize_GriSpy_neighbors' ):
        os.mkdir('optimize_GriSpy_neighbors')
    if os.path.exists( filename ) and not overwrite:
        seconds = np.load( filename )
    else:
        if data is None:
            rng = np.random.default_rng(seed=0)
            data = np.concatenate([rng.uniform(-np.pi, np.pi, size=(len_data, 1)),
                                   rng.uniform(-np.pi / 2, np.pi / 2, size=(len_data, 1))], axis=1)
        else:
            if isinstance(data, pd.core.frame.DataFrame):
                data = np.radians(data[['longitude', 'latitude']]).values
            data = data[np.random.choice(data.shape[0], size=len_data, replace=False).reshape(-1)]
        seconds = np.tile( np.NaN, range_N_cells[1])
        pbar = tqdm( range(*range_N_cells), desc = filename )
        for N_cells in pbar:
            start = datetime.datetime.now()
            GriSpy_neighbors( lon_lat=data, n_nearest=n_nearest, distance_upper_bound=distance_upper_bound,
                              N_cells=N_cells)
            seconds[N_cells] = (datetime.datetime.now() - start).total_seconds()
            pbar.set_postfix({'N_cells': N_cells, 'best time': np.nanmin(seconds), 'best N_cells': np.nanargmin( seconds) })
            if ( sum(seconds[np.nanargmin(seconds):N_cells] > np.nanmin(seconds)) > 2 ):
                break
        np.save(filename, seconds )
    return np.nanargmin(seconds), seconds[np.nanargmin(seconds)]

##
if __name__ == '__main__':
    # param_self_neighbors(date_str='2022_01_01')

    best_N_cells_self_neighbors(len_data=12000, distance_upper_bound_m=1.5E3, calculate=False)
    ##
    # %timeit GriSpy_neighbors(lon_lat=np.array([lon1, lat1] ).T, N_cells = 10)
    # print( optimize_GriSpy_neighbors( data = import_AIS.get_df( '01-01-2022')[0], n_nearest = 3, distance_upper_bound= 0.001, overwrite = True, len_data = 10000, range_N_cells = [180, 400, 10] ) )


##

