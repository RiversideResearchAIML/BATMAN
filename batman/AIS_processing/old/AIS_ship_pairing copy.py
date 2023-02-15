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

##
df_truth['meters_per_pixel_from_lon'] = df_truth.apply( lambda row_: funsak.distance_haversine(lon1=row_.lon1, lon2=row_.lon2, lat1=(row_.lat1+row_.lat2)/2, lat2=(row_.lat1+row_.lat2)/2, convert2radians=True, units='m')/row_.image_w, axis = 1 )
df_truth['meters_per_pixel_from_lat'] = df_truth.apply( lambda row_: funsak.distance_haversine(lon1=row_.lon1, lon2=row_.lon1, lat1=row_.lat1, lat2=row_.lat2, convert2radians=True, units='m')/row_.image_h, axis = 1 )
##
if False:
    for row_, df_ in df_truth.groupby('filename'):
        if (df_['meters_per_pixel'].std() > .1):
            print(ship_pairing_analysis.jpg_lookup_list[row_])
            break
    ##

    ##
    fig, ax = plt.subplots(2,1, figsize = [8, 11])
    for i, file in enumerate( ship_pairing_analysis.jpg_lookup_list[row_] ):
        ax[i].imshow(PIL.Image.open( file ))
        ax[i].set_title( os.path.sep.join(file.split(os.sep)[-4:]) + '\n meters_per_pixel %g'%df_.meters_per_pixel.values[i])
    fig.show()
    ##
    a = df_truth[['filename', 'meters_per_pixel', 'meters_per_pixel_from_lat', 'meters_per_pixel_from_lon']].drop_duplicates().reset_index()
    a['discrep w lat'] = np.abs(a['meters_per_pixel']/a[ 'meters_per_pixel_from_lat']-1)
    a['discrep w lon'] = np.abs(a['meters_per_pixel']/a[ 'meters_per_pixel_from_lon']-1)
    a.iloc[(np.max( a[['discrep w lat', 'discrep w lon']], axis = 1)).sort_values(ascending = False).index.values].to_csv('meters_per_pixel_check.csv')
    ##
    df_truth.groupby('filename')['meters_per_pixel'].std().fillna(0).sort_values()
    ##
    df_truth[['filename', 'meters_per_pixel', 'meters_per_pixel_from_lon', 'meters_per_pixel_from_lat']]

    ##
    for _, row_ in df_truth.iterrows():
        im =  PIL.Image.open(ship_pairing_analysis.jpg_lookup[row_.filename] )
        assert all( im.size == row_[['image_w', 'image_h']].values )
##
path_data = os.path.join( AIS_parameters.dirs['google-earth-eo-data'], 'ship_truth_data AIS_infer_mmsi merged data' )
path_plots = os.path.join( path_data, fname[:-3])

##
df_truth.drop_duplicates( 'filename' )

df_stats = ship_pairing_analysis.get_df_stats()
##
def generate_label( df__, prefix = 'both'):
    label = '%s: mmsi = %i; ' %(prefix, df__['mmsi'] )
    if df__['VesselName_1']:
        label += ' %s; ' % df__['VesselName_1']
    elif df__['VesselName_1E6']:
        label += ' %s: ' % df__['VesselName_2']
    else:
        label += ' NA: '
    try:
        label += ' %s: ' % df__['AIS_description\nVesselType_1']
    except:
        try:
            label += ' %s:' % df__['AIS_description\nVesselType_1E6']
        except:
            label += ' NA;'
    try:
        label += ' %s;' % df__['AIS_description\nStatus_1']
    except:
        try:
            label += ' %s;' % df__['AIS_description\nStatus_1E6']
        except:
            label += ' NA;'
    label = re.sub( 'nan', 'NA', re.sub( 'None', 'NA', re.sub( '\(.*\)', '', label ) ) )
    if np.isnan(df__['interp distance m EO_1E6']):
        label += ' Δ = (%g m, %g sec)' % ( np.round(df__['interp distance m EO_1'], decimals=0),
                                    np.subtract(df__['timestamp nearest EO_1'], df__['timestamp EO']).total_seconds())
    else:
        label += ' Δ = (%g m, %g sec)' % ( np.round(df__['interp distance m EO_1E6'], decimals=0),
                                    np.subtract(df__['timestamp nearest EO_1E6'],
                                                df__['timestamp EO']).total_seconds())

    return label

df_combined = dict()
df_combined_value_counts = dict()
for params, df_ in df_stats.groupby( ['threshold_sec', 'threshold_m','clustering_eps_m']):
    df1 = pd.read_feather(os.path.join(path_data, df_.loc[1 == df_.max_perm, 'file' ].values[0] ) )
    df1E6 = pd.read_feather(os.path.join(path_data, df_.loc[1E6 == df_.max_perm, 'file' ].values[0] ) )
    if not all(df1E6['interp distance m EO'] <= params[1]):
        continue
    df1E6['x'] = df1E6['ship_pixel_loc EO'].apply( lambda x: x[0]  )
    df1E6['y'] = df1E6['ship_pixel_loc EO'].apply( lambda x: x[1]  )
    df1['x'] = df1['ship_pixel_loc EO'].apply( lambda x: x[0]  )
    df1['y'] = df1['ship_pixel_loc EO'].apply( lambda x: x[1]  )
    keys = tuple(int(v) for v in  params)
    columns = ['timestamp EO', 'x', 'y', 'mmsi', 'filename EO', 'timestamp nearest EO', 'interp latitude EO', 'interp longitude EO', 'interp distance m EO', 'VesselName', 'AIS_description\nVesselType', 'AIS_description\nStatus', 'meters_per_pixel EO', 'lat1 EO', 'lon1 EO', 'lat2 EO', 'lon2 EO', 'Length', 'Width']
    df_combined[keys] = pd.merge( df1[columns].dropna( subset =[ 'mmsi']), df1E6[columns].dropna( subset =[ 'mmsi']),
                    on = ['timestamp EO', 'x', 'y', 'mmsi', 'filename EO', 'meters_per_pixel EO', 'lat1 EO', 'lon1 EO', 'lat2 EO', 'lon2 EO', 'Length', 'Width' ], how = 'outer', suffixes = ['_1', '_1E6'], indicator=True)
    df_combined_value_counts[keys] = df_combined[keys]._merge.value_counts()
df_combined_value_counts = pd.DataFrame( df_combined_value_counts).T
df_combined_value_counts.index.names = ['threshold_sec', 'threshold_m','clustering_eps_m']
df_combined_value_counts['delta'] = (df_combined_value_counts.left_only - df_combined_value_counts.right_only)
df_combined_value_counts.sort_values('delta', inplace=True)
df_combined_value_counts
##
for keys in df_combined_value_counts.index.values:
    for file_res, df__ in df_combined[keys].groupby( ['filename EO', 'meters_per_pixel EO'] ):
        if df__._merge.value_counts().right_only > df__._merge.value_counts().left_only:
            out_file = os.path.join( path_plots, 'threshold_sec %g threshold_m %g clustering_eps_m %g'%(keys[0], keys[1], keys[2]), file_res[0] )
            print( out_file )
            im = PIL.Image.open(ship_pairing_analysis.jpg_lookup[df__['filename EO'].values[0]])
            fig, ax = plt.subplots(1,1, figsize = [10, 6])
            ax.imshow(im)

            df__['x_1'] = interp1d(df__[ ['lon1 EO', 'lon2 EO']].to_numpy()[0,:], [0, im.size[0]], bounds_error=False,fill_value='extrapolate')(df__['interp longitude EO_1'])
            df__['x_1E6'] = interp1d(df__[ ['lon1 EO', 'lon2 EO']].to_numpy()[0,:], [0, im.size[0]], bounds_error=False,fill_value='extrapolate')(df__['interp longitude EO_1E6'])
            df__['y_1'] = interp1d(df__[['lat1 EO', 'lat2 EO']].to_numpy()[0, :], [0, im.size[1]], bounds_error=False, fill_value='extrapolate')(df__['interp latitude EO_1'])
            df__['y_1E6'] = interp1d(df__[['lat1 EO', 'lat2 EO']].to_numpy()[0, :], [0, im.size[1]], bounds_error=False, fill_value='extrapolate')(df__['interp latitude EO_1E6'])

            for _, df___ in  df__[df__._merge=='left_only'].iterrows():
                ax.plot( [df___.x, df___.x_1], [df___.y, df___.y_1], color = 'r', marker = '*', ms = 5, linestyle='--', alpha = .5)
                ax.scatter( df___.x, df___.y, marker = 's', color = 'r', s = 30,
                            label = generate_label( df___, prefix = 'max_perm = 1') )
                ax.text(df___.x+5, df___.y-5, '%i'%df___.mmsi, color='r', verticalalignment = 'top')
                if not np.isnan(df___.Length):
                    ax.add_patch(plt.Circle((df___.x, df___.y), df___.Length/file_res[1], color='r', linestyle = '-', fill=False, alpha = .6))
            for _, df___ in  df__[df__._merge=='right_only'].iterrows():
                ax.plot( [df___.x, df___.x_1E6], [df___.y, df___.y_1E6], color = 'c', marker = '*', ms = 5, linestyle='--', alpha = .5)
                ax.scatter( df___.x, df___.y, marker = '.', color = 'c', s = 20,
                            label = generate_label( df___, prefix = 'max_perm = 1E6') )
                ax.text(df___.x+5, df___.y-5, '%i'%df___.mmsi, color='c', verticalalignment = 'bottom')
                if not np.isnan(df___.Length):
                    ax.add_patch(plt.Circle((df___.x, df___.y), df___.Length/file_res[1], color='c', linestyle = '-', fill=False, alpha = .6))

            for _, df___ in  df__[df__._merge=='both'].iterrows():
                ax.plot( [df___.x, df___.x_1], [df___.y, df___.y_1], color = 'b', marker = '*', ms = 5, linestyle='--', alpha = .5)
                ax.scatter( df___.x, df___.y, marker = '.', color = 'b',  s = 30,
                            label = generate_label( df___, prefix = 'both') )
                ax.text(df___.x+5, df___.y, '%i'%df___.mmsi, color='b', verticalalignment = 'center', label = 'both')
                if not np.isnan(df___.Length):
                    ax.add_patch(plt.Circle((df___.x, df___.y), df___.Length/file_res[1], color='b', linestyle = '-', fill=False, alpha = .6))

            l = ax.legend(loc=(0.1, .9), fontsize='small', facecolor='grey', framealpha = 1)

            for text in l.get_texts():
                if text.get_text().startswith( 'max_perm = 1E6'):
                    text.set_color("cyan")
                elif text.get_text().startswith('max_perm = 1'):
                    text.set_color("red")
                elif text.get_text().startswith('both'):
                    text.set_color("blue")

            ax.set_xlim([-.25*im.size[0], 1.25*im.size[0]])
            ax.set_ylim([1.25*im.size[1],-.25*im.size[1]])

            box = ax.get_position()
            box.x0 = box.x0 + 0.06
            box.x1 = box.x1 + 0.06
            ax.set_position(box)
            ax.set_xticks(ticks=[0, im.size[0]],
                               labels=['0 m, %s deg'%df__['lon1 EO'].mean(), '%.0f m, %s deg'% (im.size[0]*file_res[1], df__['lon2 EO'].mean())])
            ax.set_yticks(ticks=[0, im.size[1]],
                               labels=['0 m, %s deg'%df__['lat1 EO'].mean(), '%.0f m, %s deg'% (im.size[1]*file_res[1], df__['lat2 EO'].mean())])

            plt.show()
            funsak.create_dir4file(out_file)
            fig.savefig(out_file)
            plt.close(fig)
##