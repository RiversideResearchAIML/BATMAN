import numpy as np
import functions_basic_sak as funsak
import pandas as pd
from tqdm import tqdm
import os
from sklearn.cluster import DBSCAN
import inspect
import AIS_parameters
import re
import import_AIS
fname = os.path.split( inspect.getfile(inspect.currentframe()) )[1]


def get_timestamp_branch( df, df_prev = None, time_column = 'timestamp', groupby_column = 'mmsi', pbar = False, speed_threshold_m_sec = 35, time_threshold_sec = 3600 ): #80 mph, which is the fastest yacht
    '''
    identifies trajectory lat/lon clusters based on speed, threshold_m_sec,
    if mmsi has no spoofing, then spoofing cluster values = -1,
    if there is spoofing, the largest cluster is assigned 0, then next-largest 1, ...
    '''
    if 'index' in df:
        df = df.drop( columns = 'index')
    ##
    if df_prev is None:
        df['branch start'] = pd.Timestamp.max
        df['branch time'] = 0
        df['current'] = True
        ##
    else:
        if 'index' in df_prev:
            df_prev = df_prev.drop(columns='index')
        df = pd.concat( [df_prev[df_prev['timestamp'] > df['timestamp'].min() - pd.to_timedelta(time_threshold_sec, 'sec')], df], axis = 0 )
        df['current'] = df['branch start'].isna()
    df = df[['mmsi', 'timestamp', 'latitude', 'longitude', 'current', 'branch start', 'branch time']].copy()
    df[['latitude_radians', 'longitude_radians']] = np.radians( df[['latitude', 'longitude']] )
    df['cos_latitude']= np.cos( df['latitude_radians'] )
    df['sin_latitude'] = np.sin(df['latitude_radians'])
    df['dist_m'] = funsak.distance_cosine_formula(lon1=df['longitude_radians'].to_numpy(), lon2=df['longitude_radians'].shift(-1).to_numpy(),
                                   lat1 =df['latitude_radians'].to_numpy(), lat2=df['latitude_radians'].shift(-1).to_numpy(),
                                   sin_lat1 = df['sin_latitude'].to_numpy(), sin_lat2 = df['sin_latitude'].shift(-1).to_numpy(),
                                   cos_lat1= df['cos_latitude'].to_numpy(), cos_lat2 = df['cos_latitude'].shift(-1).to_numpy(), units = 'm' )
    ##
    grouped = df.groupby('mmsi')
    df['branch start'] = grouped['branch start'].transform(min)
    df['branch time'] = grouped['branch time'].transform(max)
    df = df[df['current']].drop( columns='current').copy()
    df_out = pd.DataFrame( columns=['branch start', 'branch time'], index = df.index)
    ##
    df['delta_sec'] = df['timestamp'].diff().to_numpy(int)/1E9
    tf = df['mmsi']!= df['mmsi'].shift(1)
    df.loc[ tf, ['dist_m']] = np.NaN
    df['speed_m_sec'] = df['dist_m']/df['delta_sec']
    df.loc[tf, ['delta_sec']] = 0
    timestamp = df['timestamp'].to_numpy()
    delta_sec = df['delta_sec'].to_numpy(int)
    branch_start = df['branch start'].to_numpy()
    branch_time = df['branch time'].to_numpy()
    df['tf_broken'] = np.logical_or( df['speed_m_sec'] > speed_threshold_m_sec, df['delta_sec'] > time_threshold_sec )
    df_group = df.groupby( 'mmsi')
    df_broken = df[df_group['tf_broken'].transform( sum ) > 0]
    df_group['delta_sec'].transform(pd.Series.cumsum) + df['branch time']
    ##
    df_broken_grouped = df_broken.groupby( 'mmsi')
    ##
    if pbar:
        pbar_ = df_broken_grouped
        pbar.set_postfix_str( '%s speed_threshold_m_sec %.3G, time_threshold_sec %.3G'%(fname, speed_threshold_m_sec, time_threshold_sec ) )
    else:
        pbar_ = tqdm( df_broken_grouped, desc = '%s speed_threshold_m_sec %.3G, time_threshold_sec %.3G'%(fname, speed_threshold_m_sec, time_threshold_sec ) )
    for mmsi_, df_ in pbar_:
    ##
        pbar_.set_postfix({'mmsi': str(mmsi_)})
        mesh_dist_m = funsak.distance_cosine_formula(lon1=df_['longitude_radians'].to_numpy(),
                                       lat1 =df_['latitude_radians'].to_numpy(),
                                       sin_lat1 = df_['sin_latitude'].to_numpy(),
                                       cos_lat1= df_['cos_latitude'].to_numpy(),meshgrid=True, units = 'm' )

        mesh_dt = np.abs(np.subtract(*np.meshgrid( df_['timestamp'].to_numpy(int), df_['timestamp'].to_numpy(int) ))+1)/1E9
        mesh_speed = mesh_dist_m / mesh_dt
        metric = mesh_speed
        metric[mesh_dt > time_threshold_sec] = 2*speed_threshold_m_sec
        clustering = DBSCAN(eps=speed_threshold_m_sec, min_samples=1, metric = 'precomputed').fit(metric)
        ##
        ind_ = df_.index.to_numpy()
        index_ = [ind_[0], ind_[-1] + 1]
        timestamp_ = timestamp[index_[0]:index_[1]]
        # delta_sec_ = delta_sec[index_[0]:index_[1]]
        # cum_delta_sec_ =branch_time[index_[0]:index_[1]]
        branch_start_ = branch_start[index_[0]:index_[1]]
        for label_ in clustering.labels_:
            tf = label_ == clustering.labels_
            branch_start_[tf] = np.nanmin( [timestamp_[tf][0], branch_start_[tf][0]] )
            ##
            # tmp = np.concatenate([[False], tf[1:] & tf[:-1]])
            # if any(tmp):
            #     cum_delta_sec_[tmp] += np.cumsum( delta_sec_[tmp] )
        ##

        branch_start[index_[0]:index_[1]] = branch_start_
        # branch_time[index_[0]:index_[1]] = cum_delta_sec_

        ##
    df_out['branch start'] = branch_start
    df_out['branch time'] = branch_time
    return df_out

def get_df(date='2022_06_29', time_column='timestamp', groupby_column = 'mmsi',
           pbar=None, multicolumn=False, nrows=None, overwrite=False, speed_threshold_m_sec = 35, time_threshold_sec = 3600, join = False, no_create = False ): #80 mph, which is the fastest yacht
    '''
    set timestamp marking the beginning of a clustering branch
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
    out_file = os.path.join(AIS_parameters.dirs['root'], 'ingested',
                                    '%s (%s %.3G m_sec %.3G sec).feather' % (date, fname[:-3], speed_threshold_m_sec, time_threshold_sec ))
    if not os.path.isdir(os.path.split(out_file)[0]):
        os.mkdir(os.path.split(out_file)[0])
    if no_create and not os.path.exists(out_file):
        return None, out_file
    if join or overwrite or not os.path.exists(out_file):
        df, _ = import_AIS.get_df(date)
            ##
    if os.path.exists(out_file) and not overwrite:
        df_out = pd.read_feather(out_file)
            ##
    else:
        prev_day = (pd.to_datetime(re.sub('_', '-', date)) - pd.to_timedelta(1, 'd')).strftime('%Y_%m_%d')
        df_prev, _prev = get_df( prev_day, join = True, no_create = True)
        df_out = get_timestamp_branch( df, df_prev, time_column=time_column, groupby_column = groupby_column, pbar=pbar, speed_threshold_m_sec = speed_threshold_m_sec, time_threshold_sec = time_threshold_sec )
        df_out.to_feather(out_file)
        ##
    if multicolumn:
        df_out.columns = pd.MultiIndex.from_product([[fname[:-3]], df_out.columns.to_list()])
    if join:
        if multicolumn:
            df.columns = pd.MultiIndex.from_product([['import_AIS'], df.columns.to_list()])
        df_out = pd.concat([df, df_out], axis=1)

        return df_out, out_file


if __name__ == '__main__':
    if 'stephan' == AIS_parameters.user:
        df, _ = get_df( date = '2022_01_01', nrows = None, overwrite = True, join = True )
        print(df)