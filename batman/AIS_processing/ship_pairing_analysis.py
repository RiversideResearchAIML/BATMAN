import collections

import pandas as pd
import inspect
import re
import os
import numpy as np
import itertools
import tqdm
import AIS_interp
import functions_basic_sak as funsak
from sklearn.cluster import DBSCAN
from scipy.interpolate import interp1d
import AIS_infer_mmsi
import AIS_parameters
import matplotlib.pyplot as plt
import geopandas
import import_AIS
import AIS_description
import AIS_infer_mmsi
import PIL
import functions_basic_sak as fun_sak
import shutil
import subprocess
# import dropbox
import inspect
import AIS_parameters

fname = os.path.split( inspect.getfile(inspect.currentframe()) )[1]
##
path = os.path.join( AIS_parameters.dirs['google-earth-eo-data'], 'ship_truth_data AIS_infer_mmsi' )
merged_path = os.path.join( AIS_parameters.dirs['google-earth-eo-data'], 'ship_truth_data AIS_infer_mmsi merged data' )
stats = collections.defaultdict(list)
fullfile = lambda file: os.path.join(path, file)

def merge_EO(file = '1E+02 sec 1E+04 m 1E+04 max_perm 1E+03 eps m True correct_lon_lat.feather', pbar = None ):
    if not os.path.isdir(  path+' merged data' ):
        os.mkdir( path+' merged data')
    out_file = os.path.join( path+' merged data', file )
    if not os.path.exists( out_file ):
        df = pd.read_feather( fullfile(file) )
        df_merged = []
        if pbar is None:
            pbar = tqdm.tqdm( df.groupby(df.timestamp.dt.date ), desc = 'building %s'%file )
            pbar_ = pbar
            postfix = None
        else:
            pbar_ = df.groupby(df.timestamp.dt.date )
            postfix = pbar.postfix
        if postfix is None:
            postfix = ''
        for date_, df_ in pbar_:
            pbar.set_postfix_str(postfix + ' %s'%str(date_))
            tf = ~df_['interp index'].isna()
            df_merged.append(AIS_description.get_df(
                df=
                pd.concat( [df_[tf].rename(columns=lambda s: s + ' EO').reset_index(drop = True),
                           import_AIS.get_df(date_)[0].iloc[np.round( df_.loc[tf, 'interp index']).astype(int)].reset_index(drop = True)], axis = 1 ),
                multicolumn=True, join=True))
        df_merged = pd.concat( df_merged, axis = 0).rename( columns = lambda s: s if isinstance( s, str ) else '\n'.join(s) ).reset_index(drop=True)
        df_merged.to_feather( out_file )
    else:
        df_merged = pd.read_feather(out_file)
    return df_merged
##
def merge_EO_all():
    pbar = tqdm.tqdm(list( filter( lambda file: file.endswith( '.feather') and pd.read_feather(fullfile(file)).columns[0] != 'empty', os.listdir( path )) ), desc= 'building df_stats')
    for file in pbar:
        merge_EO( file, pbar )
        pbar.set_postfix_str(file)


def get_df_stats( path = os.path.join( AIS_parameters.dirs['google-earth-eo-data'], 'ship_truth_data AIS_infer_mmsi' ) ):
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
        if 'correct_lon_lat' in flsp:
            stats['correct_lon_lat'].append( bool(flsp[flsp.index('correct_lon_lat')-1] ))
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
##
def create_plot(jpg_file_, df_):
    ##
    im = PIL.Image.open(jpg_lookup[jpg_file_])
    fig, ax = plt.subplots(1, 1)
    ax.imshow(im)
    for i, (row__, df__) in enumerate(df_.iterrows()):
        if all(df__[['Width', 'Length']].fillna(0) > 0):
            ax.add_patch(
                plt.Circle((np.mean(df__[['x1 EO', 'x2 EO']]), np.mean(df__[['y1 EO', 'y2 EO']])), df__['Length'] / df_['meters_per_pixel EO'].mean(), color='r', lw=1,
                           fill=False))
        x1, y1 = AIS_infer_mmsi.lonlat2xy(
            df_[['lon1 EO', 'lon2 EO', 'lat1 EO', 'lat2 EO', 'image_w EO', 'image_h EO']].rename(
                columns=lambda s: s.replace(' EO', '')), lon=df_['longitude'], lat=df_['latitude'])
        # break
        label = 'mmsi %i, ' % df__['interp mmsi EO']
        if df__['VesselName']:
            label += ' %s, ' % df__['VesselName']
        else:
            label += ' NA, '
        if df__['AIS_description\nVesselType']  and not df__['AIS_description\nVesselType'].endswith( '(0)' ):
            label += ' %s, '%df__['AIS_description\nVesselType']
        else:
            label += ' NA, '
        label +='\n'
        if df__['AIS_description\nStatus']:# and not df__['AIS_description\nStatus'].endswith( '(0)' ):
            label += ' %s, '%df__['AIS_description\nStatus']
        else:
            label += ' NA, '
        label += ' %g m, %g sec' % (np.round(df__['interp distance m EO'], decimals=0),
                                    np.subtract(df__['timestamp nearest EO'], df__['timestamp EO']).total_seconds())
        plt.plot([x1[i], np.mean(df__[['x1 EO', 'x2 EO']])], [y1[i], np.mean(df__[['y1 EO', 'y2 EO']])], '-', lw=2, color='C%d' % (i), label=label)

    ax.set_xlim(df_['image_w EO'].to_numpy()[0] * np.array([-.1, 1.1]))
    ax.set_ylim(df_['image_h EO'].to_numpy()[0] * np.array([1.1, -.1]))
    ax.legend(loc=(.2, 1), fontsize='x-small')
    add_m = lambda x: ['%s m' % v for v in x]
    plt.xticks(ticks=[0, df_['image_w EO'].to_numpy()[0]],
               labels=add_m(np.round([0, df_['image_w EO'].mean()], decimals=0).astype(int)))
    plt.yticks(ticks=[0, df_['image_h EO'].to_numpy()[0]],
               labels=add_m(
                   np.round([0, df_['image_h EO'].mean() / df_['meters_per_pixel EO'].mean()], decimals=0).astype(int)))
    ax.set_title('%s\nlon = %f\nlat = %f' % (
    df_['timestamp EO'].mean().strftime('%Y-%m-%d, %H:%M'), df_['longitude EO'].mean(), df_['latitude EO'].mean()),
                 fontsize='small', loc='left', x= -.1, y=1)

    for X, Y, mmsi in zip(df_['x2 EO'].to_numpy(), df_['y2 EO'].to_numpy(), df_['interp mmsi EO']):
        ax.text(X, Y, mmsi, horizontalalignment='left',
                verticalalignment='top', fontsize='x-small', fontweight='bold', color='w')
        # ax.text( X+5, Y+5, mmsi, horizontalalignment = 'left',
        #               verticalalignment='top', fontsize = 'small', color = 'r')

    # plt.show()
    ##
    return fig

if __name__ == '__main__':
    # merge_EO_all()
    if True:
        ##
        df = get_df_stats()
        ##
        clustering_eps_m = 1000
        threshold_m = 1000
        threshold_sec = 10000
        fig, ax = plt.subplots( 1, 3, figsize = [12, 4])
        ax = ax.flatten()
        df[(df.clustering_eps_m == clustering_eps_m) & (df.threshold_m == threshold_m) & (df.threshold_sec == threshold_sec) & ( df['comp time sec']>10)].plot.scatter( 'max_perm', 'mean success pairing', logx = True, ax = ax[0])
        df[(df.clustering_eps_m == clustering_eps_m) & (df.threshold_m == threshold_m) & (df.threshold_sec == threshold_sec) & ( df['comp time sec']>10)].plot.scatter( 'max_perm', 'mean interval m', logx = True, ax = ax[1])
        df[(df.clustering_eps_m == clustering_eps_m) & (df.threshold_m == threshold_m) & (df.threshold_sec == threshold_sec) & ( df['comp time sec']>10)].plot.scatter( 'max_perm', 'comp time sec', logx = True, ax = ax[2])
        plt.show()
        ##
        out_file = os.path.join( AIS_parameters.dirs['root'], os.path.splitext(fname)[0], 'computation.tif')
        fun_sak.create_dir4file(out_file)
        fig.savefig(out_file)
        ##

    elif False:
        for file in filter(lambda file: file.endswith('.feather') and file.startswith( 'erroneous') and
            pd.read_feather(fullfile(file)).columns[0] != 'empty', os.listdir(path)):
            ##
            df_erroneous = pd.read_feather(os.path.join( path, file ))
            df = pd.read_feather(os.path.join(path, file.replace( 'erroneous distance ', '' )))
            ##
            all(df_erroneous[['interp latitude', 'interp longitude']].fillna(-1) == df[
                ['interp latitude', 'interp longitude']].fillna(-1))
            all(df_erroneous[['interp distance m']].fillna(-1) == df[
                ['interp distance m']].fillna(-1))

            ##
    elif True: #check distance between interp and found
        ##
        df_distance_check = collections.defaultdict(list)
        pbar = tqdm.tqdm( list(
            filter(lambda file: file.endswith('.feather') and not file.startswith( 'erroneous') and
            pd.read_feather(fullfile(file)).columns[0] != 'empty', os.listdir(path))),
            desc='checking distances')
        for file in pbar:
            df = pd.read_feather( os.path.join( path, file) )
            delta_m = (funsak.distance_haversine(lat1=df.latitude, lon1=df.longitude, lat2=df['interp latitude'],
                                      lon2=df['interp longitude'], convert2radians=True, units='m') - df['interp distance m'])
            df_distance_check['file'].append( file )
            # cmd = 'date - r  "%s" - R'%os.path.join( path, file )
            # proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True)
            # o, e = proc.communicate()
            # df_distance_check['last modified'] = pd.to_datetime(os.path.getmtime(os.path.join(path,file)), unit = 's')
            # df_distance_check['creation'] = pd.to_datetime(os.stat(os.path.join(path,file)).st_birthtime, unit='s')
            df_distance_check['max dist'].append( delta_m.max() )
            df_distance_check['min dist'].append(delta_m.min())
            df_distance_check['std dist'].append(delta_m.std())
        df_distance_check = pd.DataFrame(df_distance_check).sort_values('std dist', ascending = False)
        df, filename = AIS_infer_mmsi.get_df(filename= '1E+02 sec 1E+02 m 1E+04 max_perm 1E+04 eps m True correct_lon_lat.feather' )
        df_erroneous = pd.read_feather( os.path.join(path,'erroneous distance '+os.path.split(filename)[1] ) )
        df == df_erroneous
        # for file in df_distance_check.loc[df_distance_check['std dist'] > 1, 'file']:
        #     if os.path.exists(os.path.join(path,'erroneous distance '+file)):
        #         # shutil.copyfile(os.path.join(path,file), os.path.join(path,'erroneous distance '+file))
        #         # AIS_infer_mmsi.get_df( filename = file )
        #         print(file)
        ##
    elif False:
        pbar = tqdm.tqdm(list( filter( lambda file: file.endswith( '.feather' ), os.listdir( path+' merged data' ) ) ), desc='generate jpg overlays' )
        for file in pbar:
            pbar.set_postfix({'file': file})
            df = pd.read_feather(os.path.join(path+' merged data', file ))
            df['X'] = np.mean(df[['x1 EO', 'x2 EO']], axis=1)
            df['Y'] = np.mean(df[['y1 EO', 'y2 EO']], axis=1)
            for jpg_file_, df_ in df.groupby( 'filename EO'):
                if not os.path.isdir( os.path.join( path + ' merged data', file.replace( '.feather', '') ) ):
                    os.mkdir(os.path.join( path + ' merged data', file.replace( '.feather', '') ) )
                out_file = os.path.join( path + ' merged data', file.replace( '.feather', ''), jpg_file_ )
                if True:#not os.path.exists( out_file ):
                    pbar.set_postfix( {'file': file, 'jpg_file': jpg_file_ })
                    fig = create_plot(jpg_file_, df_)
                    fig.savefig(out_file)
                    plt.close(fig)

