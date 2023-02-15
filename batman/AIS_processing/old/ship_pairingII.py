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
import PIL
import ship_pairing_analysis

import AIS_parameters
path = os.path.join( AIS_parameters.dirs['google-earth-eo-data'], 'ship_truth_data AIS_infer_mmsi' )
fullfile = lambda file: os.path.join( path+' merged data', file )
##
ship_pairing_analysis.merge_EO_all()
##
jpg_lookup = {}
dir = AIS_parameters.dirs['google-earth-eo-data']
for root, dirs, files in os.walk(dir):
    for file in files:
        if file.endswith( '.jpg'):
            jpg_lookup[file] = os.path.join( root, file )
##
def create_plot(jpg_file_, df_):
    im = PIL.Image.open(jpg_lookup[jpg_file_])
    fig, ax = plt.subplots(1, 1)
    ax.imshow(im)
    for i, (row__, df__) in enumerate(df_.iterrows()):
        if all(df__[['Width', 'Length']].fillna(0) > 0):
            ax.add_patch(
                plt.Circle((df__['X'], df__['Y']), df__['Length'] / df_['meters_per_pixel EO'].mean(), color='r', lw=1,
                           fill=False))
        x1, y1 = AIS_infer_mmsi.lonlat2xy(
            df_[['lon1 EO', 'lon2 EO', 'lat1 EO', 'lat2 EO', 'image_w EO', 'image_h EO']].rename(
                columns=lambda s: s.replace(' EO', '')), lon=df_['longitude'], lat=df_['latitude'])
        # break
        label = 'mmsi %i, ' % df__['interp mmsi EO']
        if df__['VesselName']:
            label += ' %s, ' % df__['VesselName']
        label += ' %g m, %g sec' % (np.round(df__['interp distance m EO'], decimals=0),
                                    np.subtract(df__['timestamp nearest EO'], df__['timestamp EO']).total_seconds())
        plt.plot([x1[i], df__.X], [y1[i], df__.Y], '-', lw=2, color='C%d' % (i), label=label)

    ax.set_xlim(df_['image_w EO'].to_numpy()[0] * np.array([-.1, 1.1]))
    ax.set_ylim(df_['image_h EO'].to_numpy()[0] * np.array([1.1, -.1]))
    ax.legend(loc=(.5, 0), fontsize='x-small')
    add_m = lambda x: ['%s m' % v for v in x]
    plt.xticks(ticks=[0, df_['image_w EO'].to_numpy()[0]],
               labels=add_m(np.round([0, df_['image_w EO'].mean()], decimals=0).astype(int)))
    plt.yticks(ticks=[0, df_['image_h EO'].to_numpy()[0]],
               labels=add_m(
                   np.round([0, df_['image_h EO'].mean() / df_['meters_per_pixel EO'].mean()], decimals=0).astype(int)))
    ax.set_title('%s\nlon = %f\nlat = %f' % (
    df_['timestamp EO'].mean().strftime('%Y-%m-%d, %H:%M'), df_['longitude EO'].mean(), df_['latitude EO'].mean()),
                 fontsize='small', loc='left')

    for X, Y, mmsi in zip(df_.X.to_numpy(), df_.Y.to_numpy(), df_['interp mmsi EO']):
        ax.text(X + 15, Y + 15, mmsi, horizontalalignment='left',
                verticalalignment='top', fontsize='x-small', fontweight='bold', color='w')
        # ax.text( X+5, Y+5, mmsi, horizontalalignment = 'left',
        #               verticalalignment='top', fontsize = 'small', color = 'r')

    # plt.show()
    return fig

pbar = tqdm.tqdm(list( filter( lambda file_: file_.endswith( '.feather' ), os.listdir( path+' merged data' ) ) ), desc='generate jpg overlays' )
for file in pbar:
    pbar.set_postfix({'file': file})
    df = pd.read_feather(fullfile( file ))
    df['X'] = np.mean(df[['x1 EO', 'x2 EO']], axis=1)
    df['Y'] = np.mean(df[['y1 EO', 'y2 EO']], axis=1)
    for jpg_file_, df_ in df.groupby( 'filename EO'):
        if not os.path.isdir( os.path.join( path + ' merged data', file.replace( '.feather', '') ) ):
            os.mkdir(os.path.join( path + ' merged data', file.replace( '.feather', '') ) )
        out_file = os.path.join( path + ' merged data', file.replace( '.feather', ''), jpg_file_ )
        if not os.path.exists( out_file ):
            pbar.set_postfix( {'file': file, 'jpg_file': jpg_file_ })
            fig = create_plot(jpg_file_, df_)
            fig.savefig(out_file)
            plt.close(fig)



