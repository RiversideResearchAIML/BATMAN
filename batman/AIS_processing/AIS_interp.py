import datetime
import itertools
import numpy as np
import pandas as pd
import tqdm
import os

import inspect
import AIS_parameters_sak
import AIS_parameters
import import_AIS
fname = os.path.split( inspect.getfile(inspect.currentframe()) )[1]
from scipy.interpolate import interp1d as interp1d

period_msec = np.int64(pd.to_timedelta(365*20, 'd').to_numpy(int) / 1E6) #20 years
start_msec = np.int64( pd.to_datetime('2010-01-01' ).to_numpy().astype(int)/1E6 ) #AIS only started 2010

from collections.abc import Iterable
# verifying there is enough room in int64 to fold mmsi and timestamp into a single vector
# print( (period_msec*1E6 - (np.int64( pd.to_datetime('2022-01-01' ).to_numpy().astype(int)/1E6 ) - start_msec))/(2**63) )


def get_df_edges( start_date = '2021-12-30', end_date = '2022-02-04', pbar=None ):
    date_range = pd.date_range( pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date() )
    ##
    df_edge_filename = os.path.join(AIS_parameters_sak.dirs['root'], '%s %s to %s'%(fname[:-3], date_range[0].strftime('%Y-%m-%d'), date_range[-1].strftime('%Y-%m-%d')), 'df_edges.feather' )
    if not os.path.exists( os.path.split(  os.path.split( df_edge_filename )[0] )[0] ):
        os.mkdir( os.path.split(  os.path.split( df_edge_filename )[0] )[0] )
    if not os.path.exists( os.path.split( df_edge_filename )[0] ):
        os.mkdir( os.path.split( df_edge_filename )[0] )

    if os.path.exists( df_edge_filename ):
        df_edges = pd.read_feather(df_edge_filename).drop( columns = 'index')
    else:
        df_edges = []
        if isinstance(pbar, tqdm.std.tqdm):
            pbar_ = date_range
            postfix_str = str(pbar.postfix)
        elif isinstance(pbar, type(None)) or (isinstance(pbar, bool) and pbar):
            pbar = tqdm.tqdm(date_range, desc='%s loading edge rows' % fname[:-3])
            pbar_ = pbar
            postfix_str = ''

        for date in pbar_:
            if isinstance(pbar, tqdm.std.tqdm):
                pbar.set_postfix_str( postfix_str + ' generating df_edges from date = %s' % (str(date)))
            df, _ = import_AIS.get_df( date, pbar = pbar )
            last = df['mmsi'] != df['mmsi'].shift(-1)
            first = df['mmsi'] != df['mmsi'].shift()
            df_edges.append( df[last | first] )
        df_edges = pd.concat(df_edges).sort_values( ['mmsi', 'timestamp']).reset_index(drop = True)
        df_edges.to_feather(df_edge_filename)
    return df_edges, df_edge_filename

def augment( df = None, file = None, save = False ):
    augmented_file = os.path.splitext(file)[0] + ' augmented.feather'
    if os.path.exists( augmented_file):
        return pd.read_feather(augmented_file)
    else:
        if df is None:
            df = pd.read_feather(file)
        assert not (save and ( file is None ) )

        flat_columns = list(itertools.chain(*[col.split() for col in df.columns]))
        tmp = []
        if file:
            if save:
                pbar = tqdm.tqdm(df.groupby('timestamp'), desc='augment & save all interp columns for "%s"'%os.path.split(augmented_file)[1])
            else:
                pbar = tqdm.tqdm(df.groupby('timestamp'),
                                 desc='augment all interp columns for "%s"' % os.path.split(file)[1])
        else:
            pbar = tqdm.tqdm(df.groupby('timestamp'), desc='augment all interp columns')
        col2add = None
        for time_, df_ in pbar:
            df__ = get_interp(time=time_, mmsi=df_.loc[
                df_['interp mmsi'] >= 0, 'interp mmsi'])
            if col2add is None:
                col2add = [col for col in df__.columns if not any([col_ in flat_columns for col_ in col.split()])]
                col_numbers2add = [r for r, col in enumerate( df__.columns ) if not any([col_ in flat_columns for col_ in col.split()])]
            ##
            if df__.shape[0]:
                try:
                    tmp.append(df__[col2add])
                except: #necessary because of typo in 2021-01-04 : TranscieverClass - should be TransceiverClass
                    tmp.append(pd.DataFrame( df__.iloc[:, col_numbers2add], columns = col2add ))
        ##
        df.loc[df['interp mmsi'] >= 0, col2add] = pd.concat(tmp, axis=0).values
    if save:
        df.to_feather( augmented_file )
    return df


def get_interp( df = None, mmsi = None, time = None, return_df = False, how = 'nearest' ):
    if isinstance( time, str ):
        time = [pd.to_datetime(time)]
    elif not isinstance(time, Iterable):
        time = [time]
    if df is None:
        try:
            date = np.unique([pd.to_datetime(time_).date() for time_ in time])
            assert len(date) == 1
            date = date[0]
        except:
            date = pd.to_datetime(time).date()
        df = get_source_df( date )
    if mmsi is None:
        mmsi = df.mmsi.cat.categories
    else:
        mmsi = list( set(df.mmsi.cat.categories) & set( mmsi ) )
    mmsi2code = dict(zip(df.mmsi.cat.categories, range(len(df.mmsi.cat.categories))))
    columns_numeric = ['mmsi'] + [col for col, dtype in df.dtypes.items() if pd.api.types.is_numeric_dtype(dtype)]
    mesh_TIMES, mesh_MMSI = map(np.ravel, np.meshgrid(
        np.array([time__.to_datetime64().astype(int) for time__ in time]) / 1E6,
        mmsi))

    CODES_w_MSEC = np.array([mmsi2code[mmsi_] for mmsi_ in mesh_MMSI]) * period_msec + (
            mesh_TIMES - start_msec)
    data_numeric = interp1d(df.index.to_flat_index(), df[columns_numeric].to_numpy(),
                        axis=0, fill_value=np.NaN, bounds_error=False, assume_sorted=True)(CODES_w_MSEC)
    valid = data_numeric[:, columns_numeric.index('mmsi')] == mesh_MMSI
    if how == 'nearest':
        DF_ = pd.DataFrame(np.concatenate([mesh_TIMES.reshape(-1, 1) * 1E6,
                                           data_numeric], axis=1)[valid],
                           columns=['timestamp'] + columns_numeric)
        ##
        index = DF_['interp index'].round().astype(int)
        DF_['timestamp nearest'] = df.iloc[index]['timestamp'].values
        columns = [col for col in df.columns if not col in columns_numeric and not col == 'timestamp']
    else:
        assert False, 'fix later'
    # elif how == 'previous':
        index = np.floor(data_numeric[:-1]).astype(int)
        DF_ = pd.DataFrame(np.concatenate([mesh_TIMES.reshape(-1, 1) * 1E6,
                                           data_numeric], axis=1)[valid],
                           columns=['timestamp'] + columns_numeric)
    ##
    DF_[columns] = df.iloc[index][columns].values
    DF_ = DF_.astype({col: np.int32 if col == 'mmsi' else dtype for col, dtype in df.dtypes.items()})
    if return_df:
        return DF_, df[df.mmsi.isin(mmsi)]
    return DF_

def get_source_df( date_, df_edges = None ):
    ##
    if isinstance( df_edges, pd.core.frame.DataFrame ):
        df = pd.concat([df_edges[df_edges.timestamp.dt.date < date_].drop_duplicates( subset = 'mmsi', keep = 'last'),
                import_AIS.get_df(date_)[0],
                df_edges[df_edges.timestamp.dt.date > date_].drop_duplicates(subset='mmsi', keep='first') ], axis=0)
    else:
        df = import_AIS.get_df(date_)[0]
    df.mmsi = pd.Categorical(df.mmsi, ordered=True, categories=np.unique(df.mmsi.to_numpy(int)))
    ##
    delta_times_msec = np.int64(df.timestamp.to_numpy(int) / 1E6) - start_msec
    df['codes_w_msec'] = df.mmsi.cat.codes.to_numpy(int) * period_msec + delta_times_msec
    ##
    df.set_index('codes_w_msec', inplace=True)
    df.sort_index(inplace=True)
    df['interp index'] = np.arange(df.shape[0], dtype=float)
    return df


def append_columns( aux_columns, DF_, df, pbar = None ):
    if isinstance(pbar, tqdm.std.tqdm):
        postfix_str = str(pbar.postfix)
    date_ = DF_.iloc[0]['timestamp'].date()
    for adj, index in zip( ['previous', 'next', 'nearest'],[DF_['interp index'].to_numpy(int),np.ceil(DF_['interp index'].to_numpy()), np.round(DF_['interp index'].to_numpy() ) ] ):
        columns2add = [ col for col in aux_columns if ( not col in DF_) and ( adj in col) ]
        if any(columns2add ):
            if isinstance(pbar, tqdm.std.tqdm):
                pbar.set_postfix_str(
                    postfix_str + ' date = %s, appending %s %s' % ( adj,
                        str(date_), str(columns2add)))
            DF_[columns2add] = df.iloc[index][[col.replace( ' %s'%adj, '') for col in columns2add]].values
    return DF_

def resample_date_range( start_date = None, end_date = None, date_range = None, buffer = None,
             aux_columns= None, pbar=None, save = True, return_source_df = False, remove_df_edge = False ):
    ##
    if isinstance( date_range, list):
        date_range = np.array([pd.to_datetime(date_range_) for date_range_ in date_range])
    elif isinstance( date_range, str ):
        date_range = np.array(pd.to_datetime(str))

    if start_date is None:
        start_date = date_range[0].date() - buffer.round('d')
    if end_date is None:
        end_date = date_range[-1].date() + buffer.round('d')
    start_date = str( start_date )
    end_date = str( end_date )

    save &= isinstance( date_range, pd.core.indexes.datetimes.DatetimeIndex )
    root = os.path.join(AIS_parameters_sak.dirs['root'], '%s %s to %s' % ( fname[:-3], start_date, end_date ))
    if not os.path.isdir( root ):
        os.mkdir( root )
    if not os.path.isdir(os.path.join(root, date_range.freqstr)):
        os.mkdir(os.path.join(root, date_range.freqstr))
    create_filename = lambda date: os.path.join( root, date_range.freqstr, '%s.feather'%str( date ) )

    ##
    DF = dict()
    df_source = dict()

    vals, index = np.unique([x.date() for x in date_range], return_inverse=True)
    if isinstance( pbar, tqdm.std.tqdm):
        pbar_ = enumerate( vals )
        postfix_str = str(pbar.postfix)
    elif isinstance( pbar, type(None) ) or ( isinstance( pbar, bool) and pbar ):
        pbar = tqdm.tqdm( enumerate( vals ), desc = fname[:-3], total = len(vals) )
        pbar_ = pbar
        postfix_str = ''
    df_edges, df_edge_filename = get_df_edges(start_date, end_date, pbar=pbar)
    for i, date_ in pbar_:
        if  not os.path.exists( create_filename(date_)):
            ##
            time_ = date_range[i==index]
            if isinstance(pbar, tqdm.std.tqdm):
                pbar.set_postfix_str( postfix_str + ' date = %s, # times = %d'%( str(date_), len(time_) ) )
            df = get_source_df( date_, df_edges )
            DF_ = get_interp(df=df, mmsi = None, time=time_ )
            if save:
                pbar.set_postfix_str(postfix_str + ' date = %s, # times = %d, save to %s' % (str(date_), len(time_), create_filename(date_)))
                DF_.to_feather(create_filename(date_))
            ##
        else:
            DF_ = pd.read_feather(create_filename((date_)))
            df = None
        if not aux_columns is None:
            for adj in ['previous', 'next', 'nearest']:
                if adj in aux_columns:
                    aux_columns.remove(adj)
                    aux_columns.extend( ['%s %s'%(col, adj) for col in df_edges.columns if not col == 'mmsi'])
            if not (set(DF_.columns.to_list()) >= set(aux_columns)):
                if df is None:
                    df = get_source_df(date_, df_edges)
                append_columns( aux_columns, DF_, df, pbar )
        ##
        if return_source_df:
            if df is None:
                df = get_source_df(date_, df_edges)
            df_source[date_] = df
        if remove_df_edge:
            if df is None:
                df = get_source_df(date_, df_edges)
            ##
            good = (df.iloc[DF_['interp index'].to_numpy(int)].timestamp.dt.date == date_).to_numpy(bool) & (df.iloc[np.ceil(DF_['interp index'].to_numpy())].timestamp.dt.date == date_).to_numpy(bool)
            DF_ = DF_[good]

        DF[date_] = DF_

    ##
    if len(DF) == 1:
        DF = DF_
        df_source = df
    if return_source_df:
        return DF, df_source
    else:
        return DF

if __name__ == '__main__':
    if 'stephan' == AIS_parameters.user:
        # DF, df_source = resample_date_range( date_range = ['2022-01-01 12:00', '2022-01-01 13:00'], aux_columns = ['next', 'previous'], return_source_df= True)
        # DF, df_source = resample_date_range( date_range = [pd.to_datetime('2022-01-01 12:00')], aux_columns = ['next', 'previous'], return_source_df= True)
        #
        # DF = resample_date_range( start_date= '2021-12-30', end_date='2022-02-03', date_range =  pd.date_range('2022-01-01', '2022-01-02', freq = 'H')[:-1], aux_columns = ['nearest'] )
        # DF = resample_date_range( date_range=pd.date_range('2022-01-01', '2022-01-02', freq='H')[:-1], aux_columns=['nearest'] )
        # DF = resample_date_range( date_range = pd.date_range('2022-01-01', '2022-02-01', freq = 'H'), buffer = pd.to_timedelta(2, 'd' ), aux_columns = ['nearest'], return_source_df= False)

        DF, df = get_interp( time = pd.date_range('2022-01-01', '2022-01-02', freq='H')[:-1], mmsi = range(100), return_df = True, aux_columns = ['previous'])
        print(DF)

