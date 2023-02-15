import pandas as pd
import numpy as np
import functions_basic_sak as funsak
import inspect
import os
import import_AIS
import re
from tqdm import tqdm
import AIS_parameters
import AIS_interp

pd.set_option('display.max_columns', 20, 'display.width', 1500)
##
dates = pd.date_range('2022-01-01', '2022-02-01', freq = 'D' )[:-1]
dates
##
interpolator = AIS_interp( )

fname = os.path.split( inspect.getfile(inspect.currentframe()) )[1]
##
def traj_calc(df):
    tf1 = df['mmsi'].diff() != 0
    tf2 = df['mmsi'].diff(-1) != 0

    df_ = pd.DataFrame(df['timestamp'].diff().dt.total_seconds()).rename(columns={'timestamp': 'delta sec'})
    df_.loc[tf1, 'delta sec'] = np.NaN
    df_['delta m'], df_['bearing deg'] = funsak.haversine_m(df['longitude'], df['latitude'], df['longitude'].shift(),
                                                              df['latitude'].shift(), bearing_too=True,
                                                              convert2radians=True)
    df_.loc[tf1, ['delta m', 'bearing deg']] = np.NaN
    avg_dl = (df_['delta m'] + df_['delta m'].shift(-1)) / 2
    df_['curvature deg/m'] = df_['bearing deg'].diff(-1) / (avg_dl + 1E-10)
    df_['speed m/sec'] = df_['delta m'] / df_['delta sec']
    df_['accel m/sec^2'] = df_['speed m/sec'].diff(-1) / df_['delta sec'].shift(-1) * 1E3
    df_['ang vel deg/sec'] = df_['bearing deg'].diff() / df_['delta sec']
    df_['ang accel deg/sec^2'] = df_['ang vel deg/sec'].diff(-1) / df_['delta sec'].shift(-1)
    df_.loc[tf2, ['accel m/sec^2', 'curvature deg/m', 'speed m/sec', 'ang vel deg/sec', 'ang accel deg/sec^2']] = np.NaN
    return df_

#check iloc[1315:1325], 'end2end m', excursion from mean m for mmsi 210959000
def get_df( date = '2022_06_29', freq= None, overwrite = False, pbar = None, multicolumn = False, join = False ):
    '''

    Parameters
    ----------
    date: either string or list of strings representing desired date(s), e.g. '2022_06_29' or ['2022_06_29', '2021_06_29', '2020_06_29']
    MarineCadaster_URL: this is intended for Marinecadastre for now, will update with Danish_URL later on.
    download_dir

    Returns either single df or dict of desired dataframes
    -------

    '''
    ##
    if isinstance(date, list):
        df_dict = {}
        pbar = tqdm( date, desc = '%s get_df'%fname )
        for date_ in pbar:
            pbar.set_postfix( {'date': date_} )
            df_dict[date_] = get_df( date = date_, freq= freq, overwrite = overwrite, pbar = pbar, multicolumn = multicolumn, join = join )
        return df_dict
    else:
        date_str = pd.to_datetime(re.sub( '_', '-', date)).strftime('AIS_%Y_%m_%d')
        if freq:
            out_file = os.path.join(AIS_parameters.dirs['root'], 'downsample',
                                    '%s (%s %s).feather' % (date_str, fname[:-3], freq))
        else:
            out_file = os.path.join( AIS_parameters.dirs['root'], 'ingested', '%s (%s).feather'%(date_str, fname[:-3]) )
        if not os.path.isdir( os.path.split( out_file)[0] ):
            os.mkdir(  os.path.split( out_file)[0] )
        df_res1 = None
        if overwrite or not os.path.exists( out_file ):
            df, df_filename = import_AIS.get_df(date, pbar = pbar )
            if freq:
                df = df[['mmsi', 'timestamp', 'latitude', 'longitude']]
                ##
                df_traj = pd.concat( [df[['mmsi', 'timestamp']], traj_calc(df)], axis = 1 )
                ##

                ##
                df_traj_grouped = df_traj.groupby([df_traj['mmsi'], pd.Grouper(key='timestamp', freq=freq)])
                ##
                flag = False
                if pbar:
                    pbar_ = {'mean', 'std', 'min', 'max'}
                else:
                    pbar_ = tqdm({'mean', 'std', 'min', 'max'}, desc='aggregate trajectory' )
                    pbar = pbar_
                    flag = True
                df_append = []# [df_traj_grouped[['timestamp']].count().rename( columns = {'timestamp':'count'})]
                for s in pbar_:
                    pbar.set_postfix({'calulation': s})
                    df_append.append( eval( 'df_traj_grouped.%s().rename( columns = lambda s: "%s "+s )'%(s, s) ) )
                df_A = pd.concat( df_append, axis = 1 )
                ##
                if flag:
                    pbar = None
                ##
                df_res1, _1 = AIS_downsample.get_df(date, freq=freq, time_column='timestamp', groupby_column='mmsi',
                                          pbar=pbar )
                ##
                df_res2, _2 = AIS_downsample.get_df(date, freq=freq, time_column='timestamp', groupby_column='mmsi',
                                          pbar=pbar, offset = (-1, 'n'))

                df_res2 = df_res2[df_res2['timestamp'] > df_res1['timestamp'].min()]
                ##
                df_aug = pd.concat([df, df_res1, df_res2], axis=0).drop_duplicates(
                    subset=['mmsi', 'timestamp']).sort_values(['mmsi', 'timestamp'])
                ##
                df_aug = pd.concat( [df_aug, traj_calc(df_aug)], axis = 1 )
                df_aug['longitude radians'] = np.radians( df_aug['longitude'])
                df_aug['latitude radians'] = np.radians(df_aug['latitude'])
                df_aug['cos latitude'] = np.cos(df_aug['latitude radians'])
                df_aug['sin latitude'] = np.sin(df_aug['latitude radians'])
                ##
                grouped = df_aug.groupby([df_aug['mmsi'], pd.Grouper(key='timestamp', freq=freq)])
                ##
                df_B = pd.concat( [grouped['latitude'].count().rename( 'count'),
                                   grouped['longitude'].mean().rename('mean longitude'),
                                   grouped['latitude'].mean().rename('mean latitude')], axis = 1 )
                ##
                if pbar:
                    pbar_ = grouped
                else:
                    pbar_ = tqdm(grouped, desc='aggregate trajectory', total = df_B.shape[0] )
                    pbar = pbar_
                df_append = []
                # mmsi_timestamp_df_res1 = df_res1[['mmsi', 'timestamp']].to_numpy()
                for mmsi_timestamp, df_gpb in pbar_:
                    pbar.set_postfix( {'mmsi': '%i'%(mmsi_timestamp[0]), 'timestamp': mmsi_timestamp[1]} )
                    if (df_gpb.shape[0] > 1):
                        ##
                        # if mmsi_timestamp[0] == 210959000:
                        #     print(3)
                        end2end_length = funsak.distance_cosine_formula(
                            lon1=df_gpb['longitude radians'].values[0], lon2=df_gpb['longitude radians'].values[-1],
                            lat1=df_gpb['latitude radians'].values[0], lat2=df_gpb['latitude radians'].values[-1],
                            sin_lat1=df_gpb['sin latitude'].values[0], sin_lat2=df_gpb['sin latitude'].values[-1],
                            cos_lat1=df_gpb['cos latitude'].values[0], cos_lat2=df_gpb['cos latitude'].values[-1],
                            meshgrid=False, bearing_too=False, units='m', degrees=False)
                        ##
                        _, bearing_first = funsak.distance_cosine_formula(
                            lon1=df_gpb['longitude radians'].values[0], lon2=df_gpb['longitude radians'].values[1],
                            lat1=df_gpb['latitude radians'].values[0], lat2=df_gpb['latitude radians'].values[1],
                            sin_lat1=df_gpb['sin latitude'].values[0], sin_lat2=df_gpb['sin latitude'].values[1],
                            cos_lat1=df_gpb['cos latitude'].values[0], cos_lat2=df_gpb['cos latitude'].values[1],
                            meshgrid=False, bearing_too=True, units='m', degrees=False)
                        _, bearing_last = funsak.distance_cosine_formula(
                            lon1=df_gpb['longitude radians'].values[-2], lon2=df_gpb['longitude radians'].values[-1],
                            lat1=df_gpb['latitude radians'].values[-2], lat2=df_gpb['latitude radians'].values[-1],
                            sin_lat1=df_gpb['sin latitude'].values[-2], sin_lat2=df_gpb['sin latitude'].values[-1],
                            cos_lat1=df_gpb['cos latitude'].values[-2], cos_lat2=df_gpb['cos latitude'].values[-1],
                            meshgrid=False, bearing_too=True, units='m', degrees=False)
                        end2end_rotation = np.degrees(bearing_last - bearing_first ) %360
                        lon_mean, lat_mean = df_gpb['longitude radians'].mean(),df_gpb['latitude radians'].mean()
                        excursion_matrix = funsak.distance_cosine_formula(
                            lon1=lon_mean, lon2=df_gpb['longitude radians'],
                            lat1=lat_mean, lat2=df_gpb['latitude radians'],
                            sin_lat1=np.sin(lat_mean), sin_lat2=df_gpb['sin latitude'],
                            cos_lat1=np.cos(lat_mean), cos_lat2=df_gpb['cos latitude'],
                            meshgrid=False, bearing_too=False, units='m', degrees=False)
                        excursion = np.max( excursion_matrix )
                        distance_matrix_m = funsak.distance_cosine_formula(
                            lon1=df_gpb['longitude radians'].values, lat1=df_gpb['latitude radians'].values[-2],
                            sin_lat1=df_gpb['sin latitude'].values,
                            cos_lat1=df_gpb['cos latitude'].values,
                            meshgrid=True, bearing_too=False, units='m', degrees=False )
                        path_length = np.sum( np.diag( distance_matrix_m, k = 1))
                        span = np.max(distance_matrix_m )
                        ##
                        df_append.append( [ end2end_length, end2end_rotation, excursion, path_length, span ] )
                        ##
                    else:
                        df_append.append(np.tile(np.NaN, 5))
                ##
                df_C = pd.DataFrame(df_append, columns = [ 'end2end m',  'end2end deg', 'excursion from mean m', 'path m', 'span m' ] )
                df_out = df_res1[['mmsi', 'timestamp']].merge(pd.concat( [df_B.reset_index(), df_C], axis = 1), on = ['mmsi', 'timestamp'], how = 'left' ).merge( df_A, on = ['mmsi', 'timestamp'], how = 'left' )
                df_out = df_out.iloc[:,2:]
                ##
                df_out = df_out.astype({ col: np.int16 if dtype == int else np.float32 for col, dtype in df_out.dtypes.items()})
                ##
                df_out.to_feather(out_file)
                ##
        else:
            if out_file.endswith('feather'):
                df_out = pd.read_feather(out_file)
    if multicolumn:
        df_out.columns = pd.MultiIndex.from_product([[fname[:-3]], df_out.columns])
    if join:
        if df_res1 is None:
            df_res1, _1 = AIS_downsample.get_df(date, freq=freq, time_column='timestamp', groupby_column='mmsi',
                                                pbar=pbar)
        if multicolumn:
            df_res1.columns = pd.MultiIndex.from_product([['AIS_downsample'], df_res1.columns.to_list()])
        df_out = pd.concat( [df_res1, df_out], axis = 1 )

    return df_out, out_file
##
##
if __name__ == '__main__':
    user = 'stephan plots freq = H'
    user = 'stephan'
    if user == 'stephan big jump':
        ##
        import plotly.express as px
        datestr = '2022-01-01'
        df = pd.concat( [ import_AIS.get_df(datestr)[0], get_df( datestr, overwrite = False )[0]], axis = 1 )
        df['SOG m/sec'] = funsak.convert_SOG2m_sec(df['SOG'])
        ##
        import seaborn as sns
        import matplotlib.pyplot as plt
        df_ = df.loc[ (df['SOG m/sec']>.1) & (df['speed m/sec']>.1), ['SOG m/sec', 'speed m/sec']]
        # fig, ax = plt.subplots(1)
        g = sns.displot(data = df_.loc[:1000000], x='SOG m/sec', y = 'speed m/sec', log_scale = [True, True], cbar = True, bins = (100, 100))
        g.plot_marginals()
        # ax.plot(1, 1, marker='o')
        plt.show()
        ##
        j = 0
        argsort = np.argsort( -df['speed m/sec'].values)
        speed = 25
        # speed = df['speed m/sec'].max()
        i = np.argmin(np.abs(df['speed m/sec'].fillna(np.inf).values - speed))
        df_ = df.loc[( np.abs( df.loc[i, 'timestamp']-df['timestamp'] ) < pd.to_timedelta(5, 'H') ) & ( df['mmsi'] == df.loc[i, 'mmsi'] ), ['mmsi', 'timestamp', 'latitude', 'longitude', 'speed m/sec']].copy()
        df_.loc[df_['timestamp'] <= df.loc[i, 'timestamp'], 'when'] = 'before'
        df_.loc[df_['timestamp'] > df.loc[i, 'timestamp'], 'when'] = 'after'
        fig = px.line_mapbox(df_, lat="latitude", lon="longitude", color="when", height=200)

        fig.update_layout(mapbox_style="stamen-terrain", mapbox_zoom=10, mapbox_center_lat=df_.loc[i, 'latitude'], mapbox_center_lon=df_.loc[i, 'longitude'],
                          margin={"r": 0, "t": 0, "l": 0, "b": 0})

        # fig.write_html("test.html")
        fig.write_image('speed %g.png'%speed)

        df.loc[(np.abs(df.loc[i, 'timestamp'] - df['timestamp']) < pd.to_timedelta(.2, 'H')) & (
                    df['mmsi'] == df.loc[i, 'mmsi']), ['mmsi', 'timestamp', 'latitude', 'longitude', 'SOG', 'SOG m/sec',
                                                       'speed m/sec']]

        ##
        ##
        ##
        def auto_open(f_map, path):
            html_page = f'{path}'
            f_map.save(html_page)
            # open in browser.
            new = 2
            webbrowser.open(html_page, new=new)

        m = folium.Map([df.loc[argmax,'latitude'], df.loc[argmax,'longitude']], zoom_start=11)
        m.save('aee.html')
        ##
    elif user == 'stephan plots freq = H':
        ##
        import matplotlib.pyplot as plt

        datestr = '2022-01-01'
        df, filename = get_df( datestr, freq='H', overwrite = False )
        # df = df.rename( columns = {'max speed m/sec': 'speed m/sec', 'max accel m/sec^2': 'accel m/sec^2', 'max ang vel deg/sec': 'ang vel deg/sec', 'max ang accel deg/sec^2': 'ang accel deg/sec^2'})
        ##


        ##
        threshold_speed = 25
        bad_speed = df['max speed m/sec'] > threshold_speed
        fig, ax = plt.subplots(2, 2, figsize=[15, 10])
        df.loc[:, 'max speed m/sec'].abs().plot(kind='hist', logx=True, logy=True, ax=ax[0, 0], bins=np.logspace( -6, 5), color='black')
        df.loc[bad_speed, 'max speed m/sec'].abs().rename( 'threshold').plot(kind='hist', logx=True, logy=True, ax=ax[0, 0], bins=np.logspace( -6, 5), color='red', legend = True)
        ax[0, 0].set_xlabel('max speed m/sec')
        ax[0, 0].set_title('threshold speed %g'%threshold_speed, color = 'red')

        tf = bad_speed
        df.loc[:, 'max accel m/sec^2'].abs().plot(kind='hist', logx=True, logy=True, ax=ax[0, 1], bins=np.logspace( -6, 7), color='black')
        df.loc[tf, 'max accel m/sec^2'].abs().rename( 'threshold').plot(kind='hist', logx=True, logy=True, ax=ax[0, 1], bins=np.logspace( -6, 7), color='red', legend = True)
        ax[0, 1].set_xlabel('abs. max accel m/sec^2')
        ax[0, 1].set_title('threshold speed %g' % threshold_speed, color = 'red')

        #
        threshold_ang_vel= 12
        bad_ang_vel = df.loc[:, 'max ang vel deg/sec'].abs() > threshold_ang_vel
        df.loc[:, 'max ang vel deg/sec'].abs().plot(kind='hist', logx=True, logy=True, ax=ax[1, 0], bins=np.logspace( -5, 3), color='black')
        df.loc[bad_ang_vel, 'max ang vel deg/sec'].abs().rename( 'threshold').plot(kind='hist', logx=True, logy=True, ax=ax[1, 0], bins=np.logspace( -5, 3), color='red', legend = True)
        ax[1, 0].set_xlabel('abs. ang vel deg/sec')
        ax[1, 0].set_title('threshold abs. ang vel %g'%threshold_ang_vel, color = 'red')

        tf = bad_ang_vel
        df.loc[:, 'max ang accel deg/sec^2'].abs().plot(kind='hist', logx=True, logy=True, ax=ax[1, 1], bins=np.logspace( -8, 3), color='black')
        df.loc[tf, 'max ang accel deg/sec^2'].abs().rename( 'threshold').plot(kind='hist', logx=True, logy=True, ax=ax[1, 1], bins=np.logspace( -8, 3), color='red', legend = True)
        ax[1, 1].set_xlabel('abs. ang accel deg/sec^2')
        ax[1, 1].set_title('threshold abs. ang vel %g'%threshold_ang_vel, color = 'red')

        plt.show()
        fig.savefig(os.path.splitext(filename)[0]+'max speed_accel.pdf' )
        ##
    elif user == 'stephan plots':
        # print( get_df(date='2022_01_01', freq = 'H', update=True) )
        datestr = '2022-01-01'
        df_raw, filename = import_AIS.get_df( datestr )
        df, filename = get_df(datestr)
        ##
        import matplotlib.pyplot as plt
        ##
        threshold_speed = 25
        bad_speed = df['speed m/sec'] > threshold_speed
        fig, ax = plt.subplots(2, 3, figsize=[20, 10])
        df.loc[:, 'speed m/sec'].abs().plot(kind='hist', logx=True, logy=True, ax=ax[0, 0], bins=np.logspace( -6, 5), color='black')
        df.loc[bad_speed, 'speed m/sec'].abs().rename( 'threshold').plot(kind='hist', logx=True, logy=True, ax=ax[0, 0], bins=np.logspace( -6, 5), color='red', legend = True)
        ax[0, 0].set_xlabel('speed m/sec')
        ax[0, 0].set_title('threshold speed %g'%threshold_speed, color = 'red')

        tf = bad_speed
        df.loc[:, 'accel m/sec^2'].abs().plot(kind='hist', logx=True, logy=True, ax=ax[0, 1], bins=np.logspace( -6, 7), color='black')
        df.loc[tf, 'accel m/sec^2'].abs().rename( 'threshold').plot(kind='hist', logx=True, logy=True, ax=ax[0, 1], bins=np.logspace( -6, 7), color='red', legend = True)
        ax[0, 1].set_xlabel('abs. accel m/sec^2')
        ax[0, 1].set_title('threshold speed %g' % threshold_speed, color = 'red')

        tf = bad_speed|bad_speed.shift(-1)
        df.loc[:, 'accel m/sec^2'].abs().plot(kind='hist', logx=True, logy=True, ax=ax[0, 2], bins=np.logspace( -6, 7), color='black')
        df.loc[tf, 'accel m/sec^2'].abs().rename( 'threshold').plot(kind='hist', logx=True, logy=True, ax=ax[0, 2], bins=np.logspace( -6, 7), color='red', legend = True)
        ax[0, 2].set_xlabel('abs. accel m/sec^2')
        ax[0, 2].set_title('threshold speed %g w/prev' % threshold_speed, color = 'red')
        #
        threshold_ang_vel= 12
        bad_ang_vel = df.loc[:, 'ang vel deg/sec'].abs() > threshold_ang_vel
        df.loc[:, 'ang vel deg/sec'].abs().plot(kind='hist', logx=True, logy=True, ax=ax[1, 0], bins=np.logspace( -5, 3), color='black')
        df.loc[bad_ang_vel, 'ang vel deg/sec'].abs().rename( 'threshold').plot(kind='hist', logx=True, logy=True, ax=ax[1, 0], bins=np.logspace( -5, 3), color='red', legend = True)
        ax[1, 0].set_xlabel('abs. ang vel deg/sec^2')
        ax[1, 0].set_title('threshold abs. ang vel %g'%threshold_ang_vel, color = 'red')

        tf = bad_ang_vel
        df.loc[:, 'ang accel deg/sec^2'].abs().plot(kind='hist', logx=True, logy=True, ax=ax[1, 1], bins=np.logspace( -8, 3), color='black')
        df.loc[tf, 'ang accel deg/sec^2'].abs().rename( 'threshold').plot(kind='hist', logx=True, logy=True, ax=ax[1, 1], bins=np.logspace( -8, 3), color='red', legend = True)
        ax[1, 1].set_xlabel('abs. ang accel deg/sec^2')
        ax[1, 1].set_title('threshold abs. ang vel %g'%threshold_ang_vel, color = 'red')

        tf = bad_ang_vel|bad_ang_vel.shift(-1)
        df.loc[:, 'ang accel deg/sec^2'].abs().plot(kind='hist', logx=True, logy=True, ax=ax[1, 2], bins=np.logspace( -8, 3), color='black')
        df.loc[tf, 'ang accel deg/sec^2'].abs().rename( 'threshold').plot(kind='hist', logx=True, logy=True, ax=ax[1, 2], bins=np.logspace( -8, 3), color='red', legend = True)
        ax[1, 2].set_xlabel('abs. ang accel deg/sec^2')
        ax[1, 2].set_title('threshold abs. ang vel %g w/prev'%threshold_ang_vel, color = 'red')

        plt.show()
        fig.savefig(os.path.splitext(filename)[0]+' speed_accel.pdf' )
        ##
    elif user == 'stephan':
        dates = pd.date_range('2022-01-01', '2022-02-01').strftime('%Y-%m-%d').to_list()
        print( get_df(date=dates, freq = 'H', overwrite = True, multicolumn = False, join = False) )
