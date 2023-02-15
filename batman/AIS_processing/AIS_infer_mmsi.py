import collections
import datetime

import pandas as pd
import inspect
import os
import numpy as np
import itertools
import tqdm
import AIS_interp
import functions_basic_sak as funsak
from sklearn.cluster import DBSCAN
from scipy.interpolate import interp1d
import AIS_parameters
import re
import PIL
import matplotlib.pyplot as plt
##
pd.set_option('display.max_columns', 20, 'display.width', 1500)
fname = os.path.split( inspect.getfile(inspect.currentframe()) )[1]

pd.set_option('display.max_columns', 20, 'display.width', 1500)

def lonlat2xy(df_, lon=None, lat=None):
    x = interp1d(df_[ ['lon1', 'lon2']].to_numpy()[0,:], [0, df_[ 'image_w'].to_numpy()[0]], bounds_error=False,fill_value='extrapolate')(lon)
    y = interp1d(df_[['lat1', 'lat2']].to_numpy()[0, :], [0, df_['image_h'].to_numpy()[0]], bounds_error=False,
                 fill_value='extrapolate')(lat)
    return x, y

def xy2lonlat(df_, x=None, y=None):
    lon = interp1d([0, df_[ 'image_w'].to_numpy()[0]], df_[ ['lon1', 'lon2']].to_numpy()[0,:], bounds_error=False,
                 fill_value='extrapolate')(x)
    lat = interp1d([0, df_['image_h'].to_numpy()[0]], df_[['lat1', 'lat2']].to_numpy()[0, :], bounds_error=False,
                 fill_value='extrapolate')(y)
    return lon, lat

def pd_height(df_or_S):
    if isinstance(df_or_S, pd.core.series.Series):
        return 1
    else:
        return df_or_S.shape[0]

def get_values(series_or_scalar, ind):
    if isinstance(series_or_scalar, pd.core.series.Series):
        values = series_or_scalar.values
    else:
        values = np.array([series_or_scalar])
    return values[ind]

def infer_mmsi( file_no_mmsi = None, threshold_sec = 300, threshold_m = 10E3,  out_file = None, max_permutations = 1E3,
            clustering_eps_m = 1E4, overwrite = False, nrows = None, multicolumn = False, pbar = None ):
    '''
    will create a dataframe for found waypoints with mmsi from MarineCadastre based upon timestamp, latitude and longitude
    
    Parameters
    ----------
    file_no_mmsi: dataframe with columns 'datetime' or 'timestamp', 'latitude', and 'longitude' that has missing mmsi
    threshold_sec: maximum time interval (seconds) between found mmsi waypoint and non-mmsi waypoint
    threshold_m: maximum distance (meters) between found mmsi waypoint and non-mmsi waypoint
    out_file: if left None, then saves to sister_file with suffix " AIS_infer_mmsi.feather"
    overwrite: set to true to redo calculations
    nrows: to limit the number of rows processed ( decrease turn-around time )
    Returns
    -------
    dataframe with colums  ['mmsi', 'timestamp nearest', 'interp interval sec', 'interp distance m'] that can be index-joined to DF
    '''
    assert max_permutations > 0, 'minimum # permutations is 1'
    import warnings
    overflow_distance = funsak.earth_radius_m * 1000
    assert ( threshold_m is None ) or ( threshold_m < overflow_distance ), 'use more reasonable (smaller) threshold_m'
    warnings.simplefilter(action='ignore', category=FutureWarning) # deal with stupid warning on np.sum( sep2[ rows[[order]], cols[[order]] ] )
    if out_file is None:
        root = os.path.join( os.path.split(file_no_mmsi)[0], '%s %s'%(os.path.splitext(os.path.split(file_no_mmsi)[1])[0], fname[:-3] ) )
        out_file = ''
        if not threshold_sec is None:
            out_file += ' %.2G sec'%threshold_sec
        if not threshold_m is None:
            out_file += ' %.2G m'%threshold_m
        if not nrows is None:
            out_file += ' %.2G nrows' % nrows
        out_file += ' %.2G max_perm %.2G eps m.feather'%(max_permutations, clustering_eps_m )
        out_file = os.path.join( root, out_file[1:] )

    # used in permutation search
    ##
    funsak.create_dir4file(out_file)
    if ( os.path.exists(out_file) and ( pd.read_feather(out_file).columns[0] == 'empty' ) )  or \
        overwrite or not os.path.exists(out_file):
        if file_no_mmsi.endswith('.csv'):
            DF = pd.read_csv(file_no_mmsi).rename(
                columns={'datetime': 'timestamp'})
        elif file_no_mmsi.endswith('.feather'):
            DF = pd.read_feather(file_no_mmsi).rename(columns={'datetime': 'timestamp' })
        if nrows:
            DF = DF.iloc[:nrows]
        if os.path.exists( out_file ):
            os.remove( out_file )
        pd.DataFrame([0], columns = ['empty']).to_feather(out_file)
        DF = DF.sort_values( 'timestamp')

        if not DF.iloc[0]['timestamp'].tzinfo is None:
            DF['timestamp']  = DF['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
        ##
        DF[ 'interp distance m' ] = np.NaN
        DF[ 'cluster#' ] = np.NaN
        DF[ 'greedy count'] = np.NaN
        DF[ 'last improvement'] = np.NaN
        DF['last iter'] = np.NaN
        if (np.abs(DF[['latitude', 'longitude']]).max().to_numpy() > np.array([1 / 2, 1]) * np.pi).any():
            DF[['latitude', 'longitude']] = np.radians(DF[['latitude', 'longitude']])
            ##
        if pbar is None:
            pbar = tqdm.tqdm( DF[['interp mmsi', 'timestamp', 'latitude', 'longitude']].groupby(  DF['timestamp'].dt.strftime('%Y-%m-%d') ),
              desc= '%s.infer_mmsi( threshold_sec = %.1E, threshold_m = %.1E,  max_permutations = %.1E, clustering_eps_m = %.1E'%( fname[:-3], threshold_sec, threshold_m,max_permutations,clustering_eps_m ) )
            pbar_ = pbar
        else:
            pbar_ = DF[['timestamp', 'latitude', 'longitude']].groupby( DF['timestamp'].dt.strftime('%Y-%m-%d') )
        if isinstance( pbar, tqdm.std.tqdm):
            postfix_str = str(pbar.postfix)
        else:
            postfix_str = None

        # import warnings
        # warnings.simplefilter('error')
        greedy_count = 0

        for date_, DF_ in pbar_:
            ##
            if isinstance(pbar, tqdm.std.tqdm):
                pbar.set_postfix_str( postfix_str + ' ... date = %s, greedy_count = %d' % (date_, greedy_count))
            df_interp = AIS_interp.get_interp(time=DF_['timestamp'].drop_duplicates(), return_df=False)
            df_interp = df_interp.set_index( 'timestamp' ).rename( columns = lambda col: col if col in ['index', 'interp index', 'timestamp nearest'] else 'interp %s'%col )
            if not 'interp mmsi' in DF:
                DF[df_interp.columns] = np.NaN
            df_interp[['interp latitude', 'interp longitude']] = np.radians(df_interp[['interp latitude', 'interp longitude']].to_numpy() )
            for TIMESTAMP, DF__ in DF_.groupby( 'timestamp'):
                # if any(DF__['timestamp'] != '2018-03-21 18:00:00'):
                #     continue
                ##
                if threshold_sec is None:
                    df__ = df_interp.loc[TIMESTAMP]
                else:
                    df__ = df_interp[(df_interp[
                                           'timestamp nearest'] - TIMESTAMP).dt.total_seconds().abs() < threshold_sec].loc[
                        TIMESTAMP]
                if df__.shape[0] == 0:
                    continue
                avail_mmsi = set( df__['interp mmsi'].to_numpy() )
                delta_m = funsak.distance_haversine( lon1 = DF__.longitude.values,
                                                    lat1 = DF__.latitude.values,
                       meshgrid=True, convert2radians=False, units = 'm' )
                clustering = DBSCAN(eps=clustering_eps_m, min_samples=1, metric='precomputed').fit(delta_m)
                DF.loc[DF__.index.to_flat_index(), 'cluster#'] = clustering.labels_
                ##
                for label_ in np.unique( clustering.labels_ ):
                    tf = label_ == clustering.labels_
                    DF___ = DF__[tf]
                    df___ = df__[df__['interp mmsi'].isin( avail_mmsi )]
                    ##
                    ROWS, rows = np.meshgrid( range(pd_height( DF___ )), range(pd_height(df___) ) )
                    distance_m=funsak.distance_haversine(lon1 = get_values(DF___['longitude'], ROWS ), lat1 = get_values(DF___['latitude'], ROWS ),
                                              lon2=get_values(df___['interp longitude'], rows ), lat2=get_values(df___['interp latitude'], rows ), convert2radians=False, units = 'm' )
                    ##
                    if threshold_m:
                        distance_m[distance_m > threshold_m] = overflow_distance
                        close_enough = np.any( distance_m <= threshold_m, axis = 1 )
                        df___ = df___[close_enough]
                        distance_m = distance_m[close_enough]
                    ##
                    if df___.shape[0] == 0:
                        continue
                    if distance_m.shape[0] < distance_m.shape[1]:
                        distance_m = np.concatenate( [distance_m, np.tile( overflow_distance, (distance_m.shape[1]-distance_m.shape[0], distance_m.shape[1] ))], axis = 0 )
                    ##
                    depth = distance_m.shape[1]
                    if distance_m.shape[1] > 1:
                        while ( ( np.prod(depth+ np.arange(1, distance_m.shape[1]+1)) ) < max_permutations ) and ( depth < distance_m.shape[0]):
                            depth +=1
                    rows_remap = []
                    ROWS_remap = []
                    rows_additional = dict()
                    argsort = np.argsort(distance_m, axis = None)
                    for row, ROW in zip( *np.unravel_index( argsort, distance_m.shape) ):
                        # pbar.set_postfix({ 'len(ROWS_remap)':len(ROWS_remap), 'len( rows_additional )':len( rows_additional )})
                        if ( not row in rows_remap ):
                            if (not ROW in ROWS_remap ):
                                ROWS_remap.append( ROW )
                                rows_remap.append( row )
                                if row in rows_additional:
                                    rows_additional.pop(row)
                            elif not row in rows_additional:
                                rows_additional[row] = None
                                if ( len(ROWS_remap) == distance_m.shape[1] ) and ( len( rows_additional ) >= depth ):
                                    break
                    rows_additional = list( rows_additional )
                    distance_m_remap = distance_m[:,ROWS_remap]
                    distance_m_remap = distance_m_remap[rows_remap+rows_additional, :]
                    ##
                    index_remap = np.arange(np.prod(distance_m.shape)).reshape( distance_m.shape)
                    index_remap = index_remap[:,ROWS_remap]
                    index_remap = index_remap[rows_remap+rows_additional, :]
                    ##
                    best_mean_distance = overflow_distance
                    greedy_count___ = -1
                    last_improvement = -1
                    for i, permutation in enumerate( zip(itertools.permutations( range( min(distance_m.shape+np.array([0, depth])) ), distance_m.shape[1] )) ):
                        if i == max_permutations:
                            break
                        mean_distance = np.mean( [ distance_m_remap[p, i] for i, p in enumerate( permutation[0] ) ] )
                        if mean_distance < best_mean_distance:
                            greedy_count___ += 1
                            last_improvement = i
                            best_mean_distance = mean_distance
                            best_permutation = permutation[0]
                            if isinstance(pbar, tqdm.std.tqdm):
                                pbar.set_postfix_str(
                                    postfix_str + ' ... date = %s, cluster# = %d, count = %d, improvements = %d, total above greedy = %d' % (
                                        date_, label_, DF___.shape[0], greedy_count___, greedy_count) )
                    ##
                    greedy_count += greedy_count___
                    r, R = np.unravel_index( [index_remap[p, i] for i, p in enumerate(best_permutation)], distance_m.shape )
                    tf = distance_m[r, R] < overflow_distance
                    r = r[tf]
                    R = R[tf]
                    if isinstance(pbar, tqdm.std.tqdm):
                        pbar.set_postfix_str(
                            postfix_str + ' ... date = %s, cluster# = %d, count = %d, improvements = %d, total above greedy = %d' % (
                                date_, label_, DF___.shape[0], greedy_count___, greedy_count ) )
                    DF.loc[DF___.index.values[R], df___.columns] =  df___.iloc[r].values
                    avail_mmsi -= set(df___['interp mmsi'].iloc[r].to_numpy())
                    DF.loc[DF___.index.values[R], 'greedy count'] = greedy_count___
                    DF.loc[DF___.index.values[R], 'last improvement'] = last_improvement
                    DF.loc[DF___.index.values[R], 'last iter'] = i
                    DF.loc[DF___.index.values[R], 'interp distance m'] = distance_m[r, R]
                    ##
        DF.sort_index(inplace=True)
        DF[['latitude', 'longitude', 'interp latitude', 'interp longitude']] = np.degrees( DF[['latitude', 'longitude', 'interp latitude', 'interp longitude']] )
        assert not any( DF['interp distance m'] > overflow_distance/2 )
        DF.to_feather( out_file )
    else:
        DF = pd.read_feather( out_file )

    if multicolumn:
        DF.columns = pd.MultiIndex.from_product([[fname[:-3]], DF.columns])
    return DF, out_file

def parametric_sweep( overwrite = False, **kwargs):
    '''
    flexible parametric sweep for infer_mmsi
    Parameters
    ----------
    overwrite
    kwargs:
        threshold_sec = [10, 100, 1000, 10000],
        threshold_m = [10, 30, 100, 300, 1E3, 3E3, 10E3],
        max_permutations = [1, 1E2, 1E4, 1E6, 1E8],
        clustering_eps_m = [ lambda kwargs: kwargs['threshold_m']*3 ] which makes clustering_eps_m a function
    Returns
    -------
    parameter sweep
    '''
    ##
    keys = ['max_permutations', 'clustering_eps_m', 'threshold_m', 'threshold_sec']
    kwargs = { key: np.unique( kwargs[key]) if type(kwargs[key]) in [float, int] else kwargs[key] for key in keys }
    total = np.prod( [len(val) for val in kwargs.values()] )

    pbar = tqdm.tqdm(
        itertools.product( *[val for val in kwargs.values()] ),
        total = total )
    for i, vals in enumerate( pbar ):
        kwargs_ = { k: val for k, val in zip( kwargs.keys(), list(vals) ) }
        pbar.set_postfix( kwargs_ )
        ##
        df, filename = get_df(file_no_mmsi=AIS_parameters.files['ship_truth_data'],
            overwrite=overwrite, nrows=None, pbar = False, **kwargs_ )


##
def get_df_stats( path = os.path.join( AIS_parameters.dirs['google-earth-eo-data'], 'ship_truth_data AIS_infer_mmsi' ) ):
    stats = collections.defaultdict(list)
    fullfile = lambda file: os.path.join(path, file)
    pbar = tqdm.tqdm(list( filter( lambda file: file.endswith( '.feather') and pd.read_feather(fullfile(file)).columns[0] != 'empty', os.listdir( path )) ), desc= 'building df_stats')
    for file in pbar:
        df = pd.read_feather( fullfile(file) )
        stats['file'].append(file)
        stats['comp time sec'].append(os.path.getmtime(fullfile(file)) - os.stat(fullfile(file)).st_birthtime)
        ##
        flsp = os.path.splitext(re.sub( 'eps m', 'eps', file))[0].split()
        if 'eps' in flsp:
            stats['clustering_eps_m'].append( float( flsp[flsp.index('eps')-1] ) )
        else:
            stats['clustering_eps_m'].append( np.inf )

        if 'sec' in flsp:
            stats['threshold_sec'].append( float( flsp[flsp.index('sec')-1] ) )
        else:
            stats['threshold_sec'].append( np.inf )
        ##
        if 'm' in flsp:
            stats['threshold_m'].append( float( flsp[flsp.index('m')-1] ) )
        else:
            stats['threshold_m'].append( np.inf )
        if 'max_perm' in flsp:
            stats['max_perm'].append( float(flsp[flsp.index('max_perm')-1] ) )
        else:
            stats['max_perm'].append( np.inf )
        stats['mean success pairing'].append((df['interp mmsi'] >= 0).mean())
        stats['greedy count'].append(df['greedy count'].sum())
        stats['max greedy count'].append(df['greedy count'].max())
        stats['std interval sec'].append((df['timestamp nearest'] - df['timestamp']).dt.total_seconds().std())
        stats['max interval sec'].append(
            (df['timestamp nearest'] - df['timestamp']).dt.total_seconds().abs().max())
        interval_m = funsak.distance_haversine(lon1=df['longitude'].to_numpy(), lon2=df['interp longitude'].to_numpy(),
                                               lat1=df['latitude'].to_numpy(), lat2=df['interp latitude'].to_numpy(),
                                               convert2radians=True)
        stats['std interval m'].append(
            np.nanstd(interval_m))
        stats['mean interval m'].append(
            np.nanmean(interval_m))
        stats['max interval m'].append(
            np.nanmax(interval_m))
    df_stats = pd.DataFrame(stats)
    return df_stats

def get_df( file_no_mmsi = None, threshold_sec = None, threshold_m = None, out_file = None, max_permutations = None, clustering_eps_m = None,
            overwrite = False, nrows = None, multicolumn = False, pbar = None, filename = None, clustering_eps_m_default = 2 ):
    '''

    Parameters
    ----------
    file_no_mmsi: either a string or list of strings for files
    threshold_sec: threshold for time interval to allow pairing
    threshold_m: threshold for separation to allow pairing
    out_file
    max_permutations: the number of permutations tested
    overwrite
    nrows: convenience for faster debugging
    multicolumn: will create two-column multi-index
    Returns
    -------
    dataframe, Ymd where
        dataframe has joined closest AIS data
        and Ymd are the unique dates
    '''
    if filename:
        ##
        filename = os.path.split(filename)[1]
        assert filename.split()[1::2][:-1] == ['sec', 'm', 'max_perm', 'eps']
        df_out, filename_out = get_df( file_no_mmsi=AIS_parameters.files['ship_truth_data'],
               threshold_sec=float(filename.split()[0]), threshold_m=float(filename.split()[2]),
               max_permutations=float(filename.split()[4]), clustering_eps_m = float(filename.split()[6]), overwrite=overwrite,
               nrows=None, pbar=None)
    else:
        if clustering_eps_m is None:
            clustering_eps_m = clustering_eps_m_default*threshold_m
        df_out, filename_out = infer_mmsi( file_no_mmsi = file_no_mmsi, threshold_sec = threshold_sec, threshold_m = threshold_m,
                out_file = out_file, max_permutations = max_permutations, clustering_eps_m = clustering_eps_m,
                overwrite = overwrite, nrows = nrows, multicolumn = multicolumn, pbar = pbar )
    ##
    return df_out, filename_out
##
if __name__ == '__main__':
    if True:
        parametric_sweep(
            threshold_sec=[10, 100, 1000, 10000],
            threshold_m=[10, 30, 100, 300, 1E3, 3E3, 10E3],
            max_permutations=[1, 1E2, 1E4, 1E6, 1E8],
            clustering_eps_m=[ None ], overwrite = False
            )
    elif True:
        df, filename = get_df(file_no_mmsi=AIS_parameters.files['ship_truth_data'],
                              threshold_sec=1E3, threshold_m=1E3,
                              max_permutations=1E6, overwrite=True,
                              nrows=None, pbar=None)
    elif False:
        ##
        df, filename = get_df(file_no_mmsi=AIS_parameters.files['ship_truth_data'],
                              threshold_sec=None, threshold_m=None,
                              max_permutations=1E4, overwrite=False,
                              nrows=None, pbar=None)
        ##
        assert False
        lon, lat = xy2lonlat( df, x = df.loc[0, 'image_w'], y = 0)
        lon - df.loc[0, 'lon2']
        ##
        for jpg_filename, df_ in df.groupby( 'filename'):
            if df_.shape[0] > 1:
                break
        for root, dirs, files in os.walk( AIS_parameters.dirs['google-earth-eo-data']):
            if jpg_filename in files:
                break
        jpg_filename = os.path.join( root, jpg_filename )
        ## checking
        # xy2lonlat(df_, (df_.x1+df_.x2)/2, (df_.y1+df_.y2)/2 )[0] - df_.longitude
        # xy2lonlat(df_, (df_.x1+df_.x2)/2, (df_.y1+df_.y2)/2 )[1] - df_.latitude
        lonlat2xy( df_, lon = df_['original longitude'], lat = df_['original longitude'])[0] - (df_.x1+df_.x2)/2
        lonlat2xy(df_, lon=df_.longitude, lat=df_.latitude)[1] - (df_.y1+df_.y2)/2
        ##
        im = PIL.Image.open(jpg_filename)
        plt.imshow(im)
        plt.plot( [df_.x1, df_.x1, df_.x2, df_.x2, df_.x1], [df_.y1, df_.y2, df_.y2, df_.y1, df_.y1], '-r' )

        # x, y = lonlat2xy( df_, df_['interp longitude'].to_numpy(), lat = df_['interp latitude'].to_numpy() )
        # for x_, y_, xi, yi in zip( (df_.x1+df_.x2)/2, (df_.y1+df_.y2)/2, x, y ):
        #     plt.plot( [x_, xi], [y_, yi], '--r')
        x = (df_.x1 + df_.x2) / 2
        y = (df_.y1 + df_.y2) / 2
        lon, lat = xy2lonlat( df_, x = (df_.x1+df_.x2)/2, y = (df_.y1+df_.y2)/2 )
        # x, y = lonlat2xy( df_, lon = lon, lat = lat )
        x, y = lonlat2xy( df_, lon = df_['longitude'], lat = df_['latitude'] )

        plt.plot( x, y, 'dg')
        # plt.plot( *lonlat2xy( df_, lon = -64.746, lat = 17.675 ), 'dg')
        # plt.plot( *lonlat2xy( df_, df_['longitude'].to_numpy(), lat = df_['latitude'].to_numpy() ), 'sr')
        # plt.plot( x, y, '+r', markersize = 20)
        plt.gca().set_aspect(1/np.cos(np.radians(df_.latitude.values[0])))
        plt.xticks(ticks = [0, df_.loc[0,'image_w' ]], labels=np.round(df_.loc[0, ['lon1','lon2']].to_numpy(float), decimals=3) )
        plt.yticks(ticks=[0, df.loc[0, 'image_h']], labels=np.round(df_.loc[0, ['lat1', 'lat2']].to_numpy(float), decimals = 3) )
        plt.show()
        ##
        plt.show()
        # plt.plot( xy2lonlat(df_, [0, 10000], 0 )[0], xy2lonlat(df_, [0, 10000], 0 ))
        ##
        lat1, lat2 = np.meshgrid( df_.latitude, df_['interp latitude'])
        lon1, lon2 = np.meshgrid(df_.longitude, df_['interp longitude'])
        dist_m = funsak.distance_haversine(lon1 = lon1, lon2 = lon2, lat1 = lat1, lat2 = lat2, convert2radians=True)

        best_dist = np.inf
        for p in itertools.permutations(range(3), 3):
            tmp = np.sum([dist_m[p[i], i] for i in range(3)])
            if tmp < best_dist:
                best_dist = tmp
                best_perm = p
        ##
    elif False:

        from multiprocessing.pool import ThreadPool
        from time import sleep
        from random import randint


        def dosomething(var):
            sleep(randint(1, 5))
            print(var)

        array = ["a", "b", "c", "d", "e"]
        with ThreadPool(processes=2) as pool:
            pool.map(dosomething, array)

    elif False: #https://stackoverflow.com/questions/54954075/how-to-multi-thread-with-for-loop
        from multiprocessing.pool import ThreadPool
        from time import sleep
        from random import randint

        threshold_sec = [100, 1000, 10000]
        threshold_m = [100, 1E3, 10E3]
        max_permutations = [1, 1E1, 1E2, 1E4, 1E6,1E8]
        clustering_eps_m = [1E1, 1E2, 1E3, 1E4]
        clustering_eps_m = [1E2, 1E3, 1E4]

        def dosomething(sec_m_perm_eps):
            print('threshold_sec = %g, threshold_m = %g, max_permutations = %g, clustering_eps_m = %g'%(sec_m_perm_eps) )
            start = datetime.datetime.now()
            get_df(file_no_mmsi=AIS_parameters.files['ship_truth_data'],
                   threshold_sec=sec_m_perm_eps[0], threshold_m=sec_m_perm_eps[1],
                   max_permutations=sec_m_perm_eps[2], clustering_eps_m=sec_m_perm_eps[3],
                   overwrite=False, nrows=None, pbar = False )
            print('threshold_sec = %g, threshold_m = %g, max_permutations = %g, clustering_eps_m = %g'%(sec_m_perm_eps), datetime.datetime.now() - start )


        array = itertools.product( threshold_sec, threshold_m, max_permutations, clustering_eps_m )
        with ThreadPool(processes=2) as pool:
            pool.map(dosomething, array)
    elif False:
        stats = get_df_stats()
    elif True:
        threshold_sec = [100, 1000, 10000]
        threshold_m = [100, 1E3, 10E3]
        max_permutations = [1, 1E1, 1E2, 1E4, 1E6]
        clustering_eps_m = [1E1, 1E2, 1E3, 1E4]
            ##
        ##
    elif 'stephan' == AIS_parameters.user:
        df_master, Ymd = get_df( file_no_mmsi= AIS_parameters.files['ship_truth_data'],
                threshold_sec=300, threshold_m=None, max_permutations=1E6, overwrite=False,
                nrows=None)

        ##
        print(df_master)
        df_previous = pd.read_feather(None)
        ##
        if False:
        ##
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots( 1, 2, figsize = (10, 5))
            df_master['interp distance m'].plot( kind = 'hist', ax = ax[0], bins = np.logspace(1, 7, 20), logy = True, logx = True)
            ax[0].set_xlabel( 'pairing error, m')
            df_master['interp interval sec'].plot( kind = 'hist', ax = ax[1], bins = np.logspace(0, 3, 20), logy = True, logx = True)
            ax[1].set_xlabel( 'pairing error, sec')
            plt.show()
            fig.savefig('ship pairing results.pdf')
        ##
    else:
            print(infer_mmsi(file_no_mmsi=AIS_parameters.files['ship_truth_data'], threshold_sec=300, threshold_m=10E3,
                         overwrite=False, nrows=None))