import numpy as np
from tqdm import tqdm
import pandas as pd
import ciso8601
from pandas.api.types import union_categoricals
from grispy import GriSPy #https://grispy.readthedocs.io/en/latest/api.html
import re
import AIS_parameters
import collections
import os

earth_radius_km = 6378
earth_radius_m = 6378E3
# Distance = acos(SIN(lat1) * SIN(lat2) + COS(lat1) * COS(lat2) * COS(lon2 - lon1)) * 6371
# https://gis.stackexchange.com/questions/4906/why-is-law-of-cosines-more-preferable-than-haversine-when-calculating-distance-b
cosine_formula = lambda dlon, lat1, lat2:  np.arccos(np.maximum( -1, np.minimum(1, np.sin(lat1)*np.sin(lat2)+np.cos(lat1)*np.cos(lat2)*np.cos(dlon)) ) )
precomp_cosine_formula = lambda dlon, sin_lat1, sin_lat2, cos_lat1, cos_lat2:  np.arccos(np.maximum( -1, np.minimum(1, sin_lat1*sin_lat2+cos_lat1*cos_lat2*np.cos(dlon)) ) )

class file_lookup():
    '''
    class for finding filenames within path & all subdirs (os.walk.path)
    '''
    def __init__(self, path = os.path.join( AIS_parameters.dirs['google-earth-eo-data'], 'images' ),
                 startswith = None, endswith = None, contains = None ):
        '''

        Parameters
        ----------
        path: the parent path for search

        filters filename for search
        startswith
        endswith
        contains
        '''
        self.jpg_lookup = collections.defaultdict(lambda: None)
        self.jpg_lookup_list = collections.defaultdict(list)
        for root, dirs, files in os.walk(path):
            for file in files:
                tf = True
                if endswith:
                    tf &= file.endswith(endswith)
                if startswith:
                    tf &= file.startswith(startswith)
                if contains:
                    try:
                        tf &= file.index(contains)
                    except:
                        tf = False
                if tf:
                    self.jpg_lookup[file] = os.path.join( root, file )
                    self.jpg_lookup_list[file].append( os.path.join(root, file) )
    def get(self, file):
        '''

        Parameters
        ----------
        file: filename

        Returns str
        -------
        file with its full path
        '''
        return self.jpg_lookup[file]
    def get_all(self, file):
        '''
        useful if non-unique file names (i.e. different paths for the same name)
         Parameters
         ----------
         file: filename

         Returns list
         -------
         file with full paths
         '''
        return self.jpg_lookup_list[file]
    def duplicates(self):
        '''

        Returns dict
        -------
        of multiple paths for the same filename
        '''
        return { key: val for key, val in self.jpg_lookup_list.items() if len(val)>1 }


def create_dir4file(file):
    '''
    creates dir and subdir for file if necessary
    Parameters
    ----------
    file

    Returns
    -------
    directories created
    '''
    dir_created = []
    dir_ = ''
    for str_ in file.split(os.path.sep)[:-1]:
        if str_:
            dir_ += str_ + os.path.sep
            if not os.path.isdir(dir_):
                dir_created.append(dir_)
                os.mkdir(dir_)
    return dir_created

def distance_cosine_formula(lon1 = None, lon2=None, lat1 = None, lat2 = None, sin_lat1=None, sin_lat2=None, cos_lat1=None, cos_lat2=None, bearing_too = False, meshgrid = False, units = 'm', convert2radians = False):
    if convert2radians:
        if lon1.any():
            lon1 = np.radians( lon1 )
        if lon2.any():
            lon2 = np.radians(lon2)
        if lat1.any():
            lat1 = np.radians( lat1 )
        if lat2.any():
            lat2 = np.radians( lat2 )
    if sin_lat1 is None:
        sin_lat1 = np.sin(lat1)
    if cos_lat1 is None:
        cos_lat1 = np.cos(lat1)
    if meshgrid:
        assert (lon2 is None) and (lat2 is None) and (sin_lat2 is None) and (cos_lat2 is None)
        dlon = np.subtract( * np.meshgrid( lon1, lon1 ) )
        sin_lat1, sin_lat2 = np.meshgrid( sin_lat1, sin_lat1 )
        cos_lat1, cos_lat2 = np.meshgrid( cos_lat1, cos_lat1 )
    else:
        dlon = lon1 - lon2
    if sin_lat2 is None:
        sin_lat2 = np.sin( lat2 )
    if cos_lat2 is None:
        cos_lat2 = np.cos( lat2 )
    distance = precomp_cosine_formula(dlon, sin_lat1, sin_lat2, cos_lat1, cos_lat2)
    if bearing_too: #in radians!
        bearing = np.arctan2(np.sin(dlon) * cos_lat2,
                             np.cos(lat1) * sin_lat2 - sin_lat1 * cos_lat2 * np.cos(dlon))
    if units == 'm':
        distance *= earth_radius_m
    elif units == 'km':
        distance *= earth_radius_km
    if bearing_too:
        return distance, bearing
    return distance

def distance_haversine(lon1 = None, lon2=None, lat1 = None, lat2 = None, bearing_too = False, meshgrid = False, units = 'm', convert2radians = False):
    if convert2radians:
        if not lon1 is None:
            lon1 = np.radians( lon1 )
        if not lon2 is None:
            lon2 = np.radians(lon2)
        if not lat1 is None:
            lat1 = np.radians( lat1 )
        if not lat2 is None:
            lat2 = np.radians( lat2 )
    if meshgrid:
        if lat2 is None:
            lat2 = lat1
        if lon2 is None:
            lon2 = lon1
        lon1, lon2 = np.meshgrid( lon1, lon2 )
        lat1, lat2 = np.meshgrid( lat1, lat2 )

    dlon = lon2 - lon1
    sdlon = np.sin(dlon / 2.0)
    sdlat = np.sin((lat2 - lat1) / 2.0)
    clat1 = np.cos(lat1)
    clat2 = np.cos(lat2)
    num1 = sdlat ** 2
    num2 = clat1 * clat2 * sdlon ** 2
    distance = 2 * np.arcsin(np.sqrt(num1 + num2))
    if bearing_too:
        bearing = np.arctan2(np.sin(dlon) * np.cos(lat2),
                             np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
    if units == 'm':
        distance *= earth_radius_m
    elif units == 'km':
        distance *= earth_radius_km
    if bearing_too:
        return distance, bearing
    return distance


##
missed_dates = lambda fname: [file[4:14] for file in os.listdir(os.path.join(AIS_parameters.dirs['root'], 'ingested')) if ( not re.findall( '\(%s'%fname, file ) ) and (re.findall( '\(import_AIS\)', file )) ]

convert_SOG2m_sec = lambda series: series.values*0.5144

def extract_float(s):
    '''
    extracts floats from string s
    Parameters
    ----------
    s: string

    Returns
    -------
    list of floats if multiple found
    the only float if no multiples
    '''
    l = []
    for t in s.split():
        try:
            l.append(float(t))
        except ValueError:
            pass
    if len(l) == 1:
        return l[0]
    else:
        return l

def extract_int(s):
    '''
    extracts floats from string s
    Parameters
    ----------
    s: string

    Returns
    -------
    list of floats if multiple found
    the only float if no multiples
    '''
    l = []
    for t in s.split():
        try:
            l.append(int(float(t)))
        except ValueError:
            pass
    if len(l) == 1:
        return l[0]
    else:
        return l

def MultiIndex2single( df ):
    return df.rename(columns= lambda col: col if isinstance(col, str) else '\n'.join(col) )

def AIS_rename( df ):
    df = df.rename( columns={'BaseDateTime': 'timestamp', 'MMSI': 'mmsi', 'LAT': 'latitude', 'LON': 'longitude'} )
    if '\n' in ''.join(df.columns):
        df.columns = pd.MultiIndex.from_tuples([tuple(col.split('\n')) for col in df.columns])
    return df

def neighborhood_bad( lon_lat = None,  lon_lat_centers=None, df = None, df_centers = None, n_nearest=None, upper_radius= .0001, limit_n_nearest2upper_radius = False, convert2radians = True, N_cells_max = 256):
    '''
    neighborhood using haversine and grispy
    Parameters
    ----------
    lon_lat
    lon_lat_centers
    n_nearest
    upper_radius

    Returns
    -------
    bubble_dist, bubble_ind, bubble_count, near_dist, near_ind
    '''
    self_lon_lat_radians = False

    if lon_lat is None:
        lon_lat = df[['longitude', 'latitude']].to_numpy()
    if ( lon_lat_centers is None ) and ( df_centers is None ):
        lon_lat_centers = lon_lat
        self_lon_lat_radians = True
    elif lon_lat_centers is None:
        lon_lat_centers = df_centers[['longitude', 'latitude']].to_numpy(float)

    if convert2radians:
        lon_lat = np.radians(lon_lat)
        lon_lat_centers = np.radians( lon_lat )

    N_cells = min( N_cells_max, lon_lat_centers.shape[0] )
    gsp = GriSPy(lon_lat_centers, periodic={0: (-180, 180), 1: (-90, 90)}, metric='haversine', N_cells=N_cells)
    bubble_dist, bubble_ind, bubble_count, near_dist, near_ind = None, None, None, None, None
    if upper_radius is None:
        near_dist, near_ind = gsp.nearest_neighbors(lon_lat, n=n_nearest + self_lon_lat_radians)
        if self_lon_lat_radians:
            return near_dist[:, 1:], near_ind[:, 1:]
        else:
            return near_dist, near_ind
    else:
        bubble_dist, bubble_ind = gsp.bubble_neighbors(lon_lat, distance_upper_bound=upper_radius, sorted=True)
        bubble_count = np.zeros( len(bubble_dist), dtype = int )
        if not n_nearest is None:
            near_ind = -np.ones( (lon_lat.shape[0], n_nearest ), dtype = int)
            near_dist = np.tile( np.nan, (lon_lat.shape[0], n_nearest))
            missing = []
            for i in range( lon_lat.shape[0] ):
                if self_lon_lat_radians:
                    bubble_dist[i] = bubble_dist[i][1:]
                    bubble_ind[i] = bubble_ind[i][1:]
                bubble_count[i] = len(bubble_dist[i])
                if bubble_count[i] >= n_nearest:
                    near_dist[i] = bubble_dist[i][:n_nearest]
                    near_ind[i] = bubble_ind[i][:n_nearest]
                elif limit_n_nearest2upper_radius:
                    near_dist[i, :bubble_count[i]] = bubble_dist[i]
                    near_ind[i, :bubble_count[i]] = bubble_ind[i]
                else:
                    missing.append( i )
            n = min(lon_lat_centers.shape[0], n_nearest + self_lon_lat_radians )
            if any( missing ):
                if self_lon_lat_radians:
                    near_dist[missing, :n-1], near_ind[missing, :n-1] =  map( lambda matrix: np.array(matrix)[:, 1:], gsp.nearest_neighbors(lon_lat[missing], n = n ) )
                else:
                    near_dist[missing, :n], near_ind[missing, :n] = map( np.array, gsp.nearest_neighbors(lon_lat[missing], n = n ) )
        return bubble_dist, bubble_ind, bubble_count, near_dist, near_ind

def get_datetime( series, pbar = None ):
    '''
    ## very fast extraction of datetime from series using ciso8601
    will check if value is out-of-bounds for pandas
    :param series:
    :return:
    '''
    if pbar is None:
        val = [ciso8601.parse_datetime(string) if (isinstance(string, str) and len(string) > 5) else pd.NaT for string
               in tqdm(series, desc='ciso8601 time convert') ]
    else:
        pbar.set_postfix_str('%s ciso8601 time convert' % pbar.postfix)
        val = [ciso8601.parse_datetime(string) if (isinstance(string, str) and len(string) > 5) else pd.NaT for string in
           series]
    val = np.nan_to_num(val, pd.NaT)
    val[(val>pd.Timestamp.max) | (val < pd.Timestamp.min)] = pd.NaT
    return pd.Series( val, name =  series.name ).astype( 'datetime64[ns]' )

def df_time_resample_old( df, freq = 'H', time_column = 'timestamp', groupby_column = 'mmsi', columns = None, pbar = False ):
    '''
    time-resamples dataframe using pandas.DataFrame.interpolate('time', limit_area = 'inside')
    option for doing groupby before interpolate
    Parameters
    ----------
    df
    freq
    time_column
    groupby_column
    columns for interpolation (note: interpolation only for numeric data types)
        if is None, then will just restrict to numeric columns

    Returns
    -------
    time-interpolated dataframe for freq using columns
    '''
    assert False
    data_range = pd.DataFrame( pd.date_range(start= df[[time_column]].min().dt.ceil(freq)[time_column], end=df[[time_column]].max().dt.floor(freq)[time_column], freq = freq), columns = [time_column])
    if columns is None:
        columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    else:
        assert all( [pd.api.types.is_numeric_dtype(df[col]) for col in columns if col != time_column] )
    if not time_column in columns:
        columns.append( time_column )
    data_range['flag'] = True
    if groupby_column is None:
        df_ = df[columns].copy()
        columns.append('flag')
        df_['flag'] = False
        df_interp = pd.concat( [df_, data_range], axis = 0 ).sort_values( time_column).set_index(time_column).interpolate('time', limit_area = 'inside')
    else:
        df_interp = []
        if pbar:
            pbar_ = df[columns].groupby(df[groupby_column])
            pbar.set_postfix_str( 'functions_basic_sak.df_time_resample( freq = %s )'%freq )
        else:
            pbar_ = tqdm( df[columns].groupby( df[groupby_column] ), desc='functions_basic_sak.df_time_resample( freq = %s )'%freq )
        for _, df_ in pbar_:
            df_['flag'] = False
            df_interp.append( pd.concat( [df_, data_range], axis = 0 ).sort_values( time_column).set_index(time_column).interpolate('time', limit_area = 'inside') )
        df_interp = pd.concat( df_interp, axis = 0 )
    return df_interp[df_interp['flag']].dropna().drop( columns=['flag']).astype( {key: df[key].dtype for key in df if key in df_interp})

def df_time_resample( df, freq = 'H', time_column = 'timestamp', groupby_column = 'mmsi',  pbar = False ):
    '''
    time-resamples dataframe using pandas.DataFrame.interpolate('time', limit_area = 'inside')
    option for doing groupby before interpolate
    Parameters
    ----------
    df
    freq
    time_column
    groupby_column
    columns for interpolation (note: interpolation only for numeric data types)
        if is None, then will just restrict to numeric columns

    Returns
    -------
    time-interpolated dataframe for freq using columns
    '''
    assert False
    data_range = pd.DataFrame( pd.date_range(start= df[[time_column]].min().dt.floor(freq)[time_column], end=df[[time_column]].max().dt.ceil(freq)[time_column], freq = freq), columns = [time_column])
    def data_range_subset(t):
        tf1 = np.roll(data_range['timestamp'] >= t[0], -1)
        tf1[-1] = True
        tf2 = np.roll(data_range['timestamp'] <= t[-1], 1)
        tf2[0] = True
        return data_range[tf1 & tf2]

    numeric_columns = [col for col, dtype in df.dtypes.items() if ( col != 'index' ) and ( pd.api.types.is_numeric_dtype(dtype) or (col == time_column) )]
    non_numeric_columns = [col for col, dtype in df.dtypes.items() if not pd.api.types.is_numeric_dtype(dtype) or (col == time_column)]
    df_interp_numeric = []
    df_interp_non_numeric = []
    if pbar:
        pbar_ = df.groupby(df[groupby_column])
        pbar.set_postfix_str( 'functions_basic_sak.df_time_resample( freq = %s )'%freq )
    else:
        pbar_ = tqdm( df.groupby( df[groupby_column] ), desc='functions_basic_sak.df_time_resample( freq = %s )'%freq )
    for _, df_ in pbar_:
        data_range_ = data_range_subset(df_['timestamp'].to_numpy())
        ##
        df_interp_numeric.append(
            pd.concat([df_[numeric_columns], data_range_], axis=0).set_index(time_column).interpolate(
                'time').iloc[df_.shape[0]:] )
        df_interp_non_numeric.append(
            pd.concat([df_[non_numeric_columns], data_range_], axis=0).set_index(time_column).interpolate(
                'ffill').iloc[df_.shape[0]:] )
        ##
    ##
    df_interp = pd.concat( [pd.concat( df_interp_numeric, axis = 0 ),
    pd.concat(df_interp_non_numeric, axis=0)], axis = 1 )
    df_interp = df_interp.astype( {key: df[key].dtype for key in df if key in df_interp})
    ##
    return df_interp

def df_time_aggregate( df, freq = 'H', time_column = 'timestamp', groupby_column = 'mmsi', columns = None):
    '''
    time-aggregates dataframe using pandas.DataFrame.interpolate('time', limit_area = 'inside')
    option for doing groupby before interpolate
    Parameters
    ----------
    df
    freq
    time_column
    groupby_column
    columns for interpolation (note: interpolation only for numeric data types)
        if is None, then will just restrict to numeric columns

    Returns
    -------
    time-interpolated dataframe for freq using columns
    '''
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if columns is None:
        columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and not col in [groupby_column, time_column]]
    else:
        assert all( [pd.api.types.is_numeric_dtype(df[col]) for col in columns] )
    df_aggregate = []
    for _, df_ in tqdm( df[columns].groupby( [df[groupby_column], df['timestamp'].dt.round('H')] ), desc='functions_basic_sak.df_time_aggregate( freq = %s )'%freq ):
    # count = 0
    # for _, df_ in df[columns].groupby([df[groupby_column], df['timestamp'].dt.round('H')]):
        ##
        if df_.shape[0] > 0:
            df_aggregate.append(pd.concat( [pd.Series(_, index = [groupby_column, time_column]), df_.mean().rename(index = lambda s: 'mean '+s), df_.std().rename(index = lambda s: 'std '+s), df_.min().rename(index = lambda s: 'min '+s), df_.max().rename(index = lambda s: 'max '+s)], axis = 0 ) )
            df_aggregate[-1]['count'] = df_.shape[0]

        ##
    return pd.concat( df_aggregate, axis = 1).T


def concat_categoricals(list_data_frames):
    '''
    extension of pandas.concat for axis = 0 (vertical concat) where the categorical columns are concatenated using union_categoricals
    :param list_data_frames:
    :param axis:
    :return: concatenated dataframe
    '''
    if len( list_data_frames ) == 1:
        return list_data_frames[0]
    df_concat = pd.concat([df_[[c for c in df_.columns if df_[c].dtype != 'category']] for df_ in list_data_frames],
                          axis=0)
    pbar = tqdm([ col for col in list_data_frames[0] if list_data_frames[0][col].dtype == 'category'], disable=True)
    for col in pbar:
        pbar.set_postfix({'union_categoricals for column': col})
        df_concat[col] = union_categoricals([df_[col] for df_ in list_data_frames])
    return df_concat

def haversine_km(lon1, lat1, lon2, lat2, bearing_too = False, meshgrid = False, convert2radians = False):
    """
    Calculate the great circle distance between two points
    on the earth (specified in radians!)

    All args must be of equal length.

    """
    # lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    if convert2radians:
        lat1, lon1, lat2, lon2 = map( np.radians, [lat1, lon1, lat2, lon2])
    if meshgrid:
        lon1, lon2 = np.meshgrid( lon1, lon2)
        lat1, lat2 = np.meshgrid( lat1, lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = earth_radius_km * c
    if bearing_too:
        #https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
        bearing = np.arctan2(np.sin(dlon) * np.cos(lat2), np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
        return km, np.degrees(bearing)
    else:
        return km

def haversine_m(lon1, lat1, lon2, lat2, bearing_too = False, meshgrid = False, convert2radians = False):
    if bearing_too:
        km, bearing = haversine_km(lon1, lat1, lon2, lat2, bearing_too, meshgrid, convert2radians)
        return km*1000, bearing
    else:
        return haversine_km(lon1, lat1, lon2, lat2, bearing_too, meshgrid, convert2radians)*1000

if __name__ == '__main__':
    if 'stephan' == AIS_parameters.user:
        # lon_lat = {'Chicago': (-87.6298, 41.8781 ), 'Munich': (11.582, 48.1351) }
        # print( haversine_km( *lon_lat['Chicago'], *lon_lat['Munich'], convert2radians=True, bearing_too = True) )
        # bubble_dist, bubble_ind, bubble_count, near_dist, near_ind = neighborhood(df[['latitude', 'longitude']].to_numpy(float), n_nearest=3, upper_radius=.0001,
        #              limit_n_nearest2upper_radius=False )
        # bubble_dist_, bubble_ind_, bubble_count_, near_dist_, near_ind_ = neighborhood(df[['latitude', 'longitude']].to_numpy(float), n_nearest=3, upper_radius=.0001,
        #              limit_n_nearest2upper_radius=True )
        import import_AIS
        df = df_time_resample( import_AIS.get_df(date='2022_01_1', nrows = None)[0] )
        print(df)