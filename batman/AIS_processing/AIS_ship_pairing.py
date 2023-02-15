import ship_pairing_analysis

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
import PIL
import matplotlib.pyplot as plt
import geopandas
import import_AIS
import AIS_description
import AIS_infer_mmsi
import re
import PIL
fname = os.path.split( inspect.getfile(inspect.currentframe()) )[1]

#check resolution
##
df_truth = pd.read_feather( AIS_parameters.files['ship_truth_data'])
df_truth.drop_duplicates( 'filename')
for _, row_ in df_truth.iterrows():
    im =  PIL.Image.open(ship_pairing_analysis.jpg_lookup[row_.filename] )
    break
##


path_data = os.path.join( AIS_parameters.dirs['google-earth-eo-data'], 'ship_truth_data AIS_infer_mmsi merged data' )
path_plots = os.path.join( path_data, fname[:-3])

##
df_stats = ship_pairing_analysis.get_df_stats()
##
df_compare = []
for _, df_ in df_stats.groupby( ['threshold_sec', 'threshold_m','clustering_eps_m']):
    df_compare.append( df_.loc[df_['max_perm'] == 1E6, ['file', 'mean success pairing', 'mean interval m', 'greedy count', 'max greedy count', 'clustering_eps_m', 'threshold_sec', 'threshold_m', 'comp time sec']].reset_index(drop=True).join(
        df_.loc[df_['max_perm'] == 1, ['file', 'mean success pairing', 'mean interval m', 'max greedy count', 'comp time sec']].reset_index(drop=True),
          lsuffix='_1E6', rsuffix='_1') )
df_compare = pd.concat( df_compare, axis = 0 )
df_compare['improved # pairs'] = df_compare['mean success pairing_1'] < df_compare['mean success pairing_1E6']
df_compare['improved mean distance'] = df_compare['mean interval m_1'] > df_compare['mean interval m_1E6']
df_compare['successful pairing_1E6'] = df_compare['mean success pairing_1E6']*600
df_compare['successful pairing_1'] = df_compare['mean success pairing_1']*600
df_compare = df_compare.sort_values(['improved # pairs', 'improved mean distance','successful pairing_1E6',
                 'successful pairing_1', 'mean interval m_1E6','mean interval m_1', 'clustering_eps_m', 'threshold_sec', 'threshold_m','greedy count', 'max greedy count_1E6'],ascending=False).reset_index(drop = True)
##
for _, df_compare_ in df_compare.groupby( 'clustering_eps_m'):
    df_compare_[['threshold_m', 'threshold_sec', 'clustering_eps_m', 'improved # pairs', 'improved mean distance', 'comp time sec_1E6', 'comp time sec_1', 'successful pairing_1E6',
                 'successful pairing_1', 'mean interval m_1E6', 'mean interval m_1', 'greedy count', 'max greedy count_1E6']].sort_values(
    ['threshold_m', 'threshold_sec', 'clustering_eps_m', 'improved # pairs', 'improved mean distance'], ascending = False).to_excel( os.path.join( fname[:-3], 'clustering_eps_m %g.xls'%( _) ) )
##
colors = list('rcg')
XY_loc = [['x2 EO', 'y2 EO'], ['x2 EO', 'y1 EO']]
linestyle = ['-', '--', ':']
count = -1
def create_plot( df_, fig = None, ax = None ):
    ##
    global count
    ##
    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize = [12, 8])
        # try:
        im = PIL.Image.open(ship_pairing_analysis.jpg_lookup[df_['filename EO'].values[0]])
        # except:
        #     print(3)
        ax.imshow(im)
        count = 0
    else:
        count += 1
    color = colors[count]

    # plt.show()
    for i, (row__, df__) in enumerate(df_.iterrows()):
        if all(df__[['Width', 'Length']].fillna(0) > 0):
            ax.add_patch(
                plt.Circle((np.mean(df__[['x1 EO', 'x2 EO']]), np.mean(df__[['y1 EO', 'y2 EO']])), df__[['Length', 'Width']].max() / df_['meters_per_pixel EO'].mean(), color=color, lw=1,
                           fill=False))
        x1, y1 = AIS_infer_mmsi.lonlat2xy(
            df_[['lon1 EO', 'lon2 EO', 'lat1 EO', 'lat2 EO', 'image_w EO', 'image_h EO']].rename(
                columns=lambda s: s.replace(' EO', '')), lon=df_['longitude'].to_numpy(float), lat=df_['latitude'].to_numpy(float))
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
        # label +='\n'
        if df__['AIS_description\nStatus']:# and not df__['AIS_description\nStatus'].endswith( '(0)' ):
            label += ' %s, '%df__['AIS_description\nStatus']
        else:
            label += ' NA, '
        label += ' %g m, %g sec' % (np.round(df__['interp distance m EO'], decimals=0),
                                    np.subtract(df__['timestamp nearest EO'], df__['timestamp EO']).total_seconds())
        plt.plot([x1[i], np.mean(df__[['x1 EO', 'x2 EO']])], [y1[i], np.mean(df__[['y1 EO', 'y2 EO']])], linestyle[count], lw=2, color=color, label=label)
        # ax.text(.3*x1[i]+.7*np.mean(df__[['x1 EO', 'x2 EO']]), .3*y1[i]+.7*np.mean(df__[['y1 EO', 'y2 EO']]),
        #         '%i %s' % (df__['interp mmsi EO'], re.sub(',', '', df__['AIS_description\nVesselType'].split()[0])), horizontalalignment='left', backgroundcolor='w',
        #         verticalalignment='top', fontsize='x-small', fontweight='bold', color=colors[count])

    if count == 0:
        ax.set_xlim(df_['image_w EO'].to_numpy()[0] * np.array([-.1, 1.1]))
        ax.set_ylim(df_['image_h EO'].to_numpy()[0] * np.array([1.1, -.1]))
        add_m = lambda x: ['%s m' % v for v in x]
        plt.xticks(ticks=[0, df_['image_w EO'].to_numpy()[0]],
                   labels=add_m(np.round([0, df_['image_w EO'].mean()], decimals=0).astype(int)))
        plt.yticks(ticks=[0, df_['image_h EO'].to_numpy()[0]],
                   labels=add_m(
                       np.round([0, df_['image_h EO'].mean() / df_['meters_per_pixel EO'].mean()], decimals=0).astype(int)))
    #
    # try:
    for X, Y, mmsi, type in zip(df_[XY_loc[count][0]].to_numpy(), df_[XY_loc[count][1]].to_numpy(), df_['interp mmsi EO'], df_['AIS_description\nVesselType']):
        if type is None:
            ax.text(X, Y, '%i'%(mmsi ), horizontalalignment='left', backgroundcolor='w',
                    verticalalignment='top', fontsize='x-small', fontweight='bold', color=colors[count])
        else:
            ax.text(X, Y, '%i %s'%(mmsi, re.sub( ',', '', type.split()[0]) ), horizontalalignment='left', backgroundcolor='w',
                    verticalalignment='top', fontsize='x-small', fontweight='bold', color=colors[count])
    # except:
    #     print(3)

        # ax.text( X+5, Y+5, mmsi, horizontalalignment = 'left',

    return fig, ax
##
pbar = tqdm.tqdm(df_compare.iterrows(), desc = 'compare greedy w/ 1E6' )
for index_, row_ in pbar:
    df_1E6 = pd.read_feather( os.path.join( path_data, row_.file_1E6 ) )
    df_1E6 = df_1E6[df_1E6['interp mmsi EO']>= 0]
    df_1 = pd.read_feather( os.path.join( path_data, row_.file_1 ) )
    df_1 = df_1[ df_1['interp mmsi EO'] >= 0 ]
    if df_1.shape[0] == 0:
        continue
    for index__, row__ in df_1E6.sort_values( 'greedy count EO', ascending = False ).drop_duplicates( ['filename EO']).iterrows():
        pbar.set_postfix({'feather': row_['file_1E6'], 'jpg': row__['filename EO']})
    ##
        a =  df_1[ df_1['filename EO'] == row__['filename EO']]
        if a.shape[0] == 0:
            continue
        b = df_1E6[ df_1E6['filename EO'] == row__['filename EO']]
        b_ = []
        for index__, row__ in b.iterrows():
            tf = (row__['latitude EO'] == a['latitude EO']) & (row__['longitude EO'] == a['longitude EO']) & (
                    row__['interp mmsi EO'] == a['interp mmsi EO'])
            if not any(tf):
                b_.append( row__)
            # print( row__[['latitude EO', 'longitude EO', 'interp mmsi EO']].values- a.loc[tf, ['latitude EO', 'longitude EO', 'interp mmsi EO']].values )
        if len(b_) == 0:
            continue
        b_ = pd.concat( b_, axis = 1 ).T
        ##
        fig, ax = create_plot( a )
        create_plot( b_ , fig = fig, ax = ax )
        ax.legend(loc=(.2, .92), fontsize='x-small')
        title_str = '%s\nlon = %f\nlat = %f' % (
            a['timestamp EO'].mean().strftime('%Y-%m-%d, %H:%M'), a['longitude EO'].mean(), a['latitude EO'].mean())
        fig.text( .1, .9, title_str )
        fig.text( .8, .95, 'greedy:\n%i found, mean separation %.0f m'%(a.shape[0],a['interp distance m EO'].mean() ), color = 'r' )
        fig.text( .8, .85, '%.0E permutations:\n%i found, mean separation %.0f m'%(1E6, b.shape[0],b['interp distance m EO'].mean() ), color = 'c' )
        # df_1E6[ df_1E6['filename EO'] == df_1E6.iloc[df_1E6['greedy count EO'].argmax()]['filename EO']]
        plt.show()
        ##
        out_file = os.path.join( path_plots, row_.file_1E6.replace( '.feather', ''), row__['filename EO'])
        funsak.create_dir4file(out_file)

        fig.savefig(out_file)
        plt.close(fig)
        # break

##

