'''
behavioral classification for hourly time resampling

I know these loitering distances (5 km) may not be the best - will change in future versions!

successive loitering 5E+03 m: successive mean waypoints are within 5E3m and within the hour window the excursion (from the mean lat/lon) is also less than 5E3m
pipeline loitering by dredging diving: only for loitering ships within 5E3 m 'Dredging or underwater operations (33)', 'Diving operations (34)'
port loitering 5E+03 m: within 5 km of a port
on land loitering 5E+03 m: loitering on land
sea loitering 5E+03 m: loitering at sea over 5km from coast line
coast loitering 5E+03 m: loitering at sea with 5km of coast line
OOB 35 m/sec 45 deg/sec: out-of-bounds speed or ang vel
transshipment mmsi 20 m: mmsi of successive co-loitering within 20 m, -1 if no such ships
'''

import pandas as pd
from tqdm import tqdm
import os
import inspect
import AIS_parameters
import numpy as np
import AIS_harvest
import datetime
import AIS_description
from skimage import data, util, measure
from scipy.ndimage.measurements import label
import re

fname = os.path.split( inspect.getfile(inspect.currentframe()) )[1]

##
def from_df(df,  searches = ['port', 'portname'] ):
    if isinstance( searches, str):
        searches = [searches]
    return pd.concat( [ df[col] for col in df.columns if all( [any(re.findall( search, col)) for search in searches ] ) ], axis = 1 )

def from_col(df,  searches = ['port', 'portname'] ):
    if isinstance( searches, str):
        searches = [searches]
    return [ col for col in df.columns if all( [any(re.findall( search, col)) for search in searches ] ) ]


##

def get_df( date, freq = 'H', threshold_mks = None, overwrite = False, multicolumn = False, pbar = None ):
    ##
    out_file = os.path.join( AIS_parameters.dirs['root'], 'training', datetime.datetime.now().strftime('%m-%d'), 'AIS_%s (%s %s).feather'%(re.sub( '-', '_', date ), fname[:-3], ', '.join([ '%s %.3G'%(col, val ) for col, val in threshold_mks.items()]) ) )
    if not os.path.isdir( os.path.split(out_file)[0] ):
        os.mkdir( os.path.split(out_file)[0] )
    if not overwrite and os.path.exists( out_file ):
        return pd.read_feather( out_file ), out_file
    ##
    df, filename = AIS_harvest.get_df(date, freq=freq, overwrite=False, multicolumn=False)
    ##
    successive_loitering = np.concatenate( [df['AIS_loitering\nloit time for %g m'%threshold_mks['loitering']].to_numpy() >= 0, [False]] )
    successive_loitering = successive_loitering[1:] & successive_loitering[:-1] & ( df['AIS_trajectory\nexcursion from mean m'].to_numpy() < threshold_mks['loitering'] )
    away_from_coast_loitering = successive_loitering & ( df['AIS_numpy_tagging\ncoast_distance_km'].values*1E3 > threshold_mks['from coast'] )
    ##
    behavior_dict = dict()
    behavior_dict['successive loitering at sea within %.3G m'%threshold_mks['loitering']] = successive_loitering & ( df['AIS_trajectory\nexcursion from mean m'].to_numpy() > 0 )
    ##
    behavior_dict['successive loitering within 2E3 m of pipelines by dredging or diving vessel']  = df['GEM_Oil_Gas_Pipelines_2022-10.geojson\ngeometry\nPipelineName'].apply( lambda v: not v is None ) &\
          successive_loitering & df['AIS_description\nVesselType'].isin( ['Dredging or underwater operations (33)', 'Diving operations (34)'])
    ##
    behavior_dict['successive loitering within 10E3 m of port']  = df['wld_trs_ports_wfp.csv\nlat_lon\nportname'].apply( lambda v: not v is None ) & successive_loitering
    ##
    behavior_dict['successive loitering > %.3G m away from coast'%threshold_mks['from coast']]  = ( df['AIS_numpy_tagging\ncoast_distance_km'].values*1E3 > threshold_mks['from coast'] ) & successive_loitering
    ##
    behavior_dict['successive loitering < %.3G m near coast'%threshold_mks['from coast']]  = ( df['AIS_numpy_tagging\ncoast_distance_km'].values*1E3 < threshold_mks['from coast'] ) & successive_loitering
    ##
    behavior_dict['OOB at sea > %.3G m/sec %.3G deg/sec' % (threshold_mks['OOB speed'], threshold_mks['OOB ang vel'])] = (df['AIS_numpy_tagging\ncoast_distance_km'].values*1E3>0) & \
    ( df['AIS_trajectory\nmax speed m/sec'].values > threshold_mks['OOB speed'] ) | ( df['AIS_trajectory\nmax ang vel deg/sec'].abs().values > threshold_mks['OOB ang vel'] )
    ##
    behavior_dict['OOB within coast > %.3G m/sec %.3G deg/sec' % (threshold_mks['OOB speed'], threshold_mks['OOB ang vel'])] = (df['AIS_numpy_tagging\ncoast_distance_km'].values*1E3<0) &\
    ( df['AIS_trajectory\nmax speed m/sec'].values > threshold_mks['OOB speed'] ) | ( df['AIS_trajectory\nmax ang vel deg/sec'].abs().values > threshold_mks['OOB ang vel'] )
    ##
    behavior_dict['successive loitering at sea and fishing vessel and Fishing Prohibited']  =  (df['AIS_numpy_tagging\ncoast_distance_km'].values*1E3 >0 ) &\
            ( df['NOAA_MPAI_v2020.gdb\ngeometry\nFish_Rstr'] == 'Commercial Fishing Prohibited and Recreational Fishing Restricted' ) &\
            successive_loitering & (df['AIS_description\nVesselType']=='Fishing (30)')
    behavior_dict['successive loitering at sea and fishing vessel dark'] =  (df['AIS_numpy_tagging\ncoast_distance_km'].values*1E3 >0 ) &\
                                        ( df['AIS_trajectory\nmax delta sec'] > 3600 ) & (df['AIS_numpy_tagging\ncoast_distance_km'].values*1E3 >0 ) &\
                                        successive_loitering & (df['AIS_description\nVesselType']=='Fishing (30)')
    behavior_dict['successive loitering at sea and fishing vessel and Fishing Prohibited and dark'] = behavior_dict['successive loitering at sea and fishing vessel and Fishing Prohibited'] & \
        (df['AIS_trajectory\nmax delta sec'] > 3600)

    ##
    df['anamolous routing'] = ( df['AIS_numpy_tagging\ncoast_distance_km'].values*1E3 > threshold_mks['from coast'] ) &\
                             (( df['AIS_description\nVesselType'].apply( lambda s: ( s.find('Cargo') >= 0 ) ) ) & \
                             (df['AIS_numpy_tagging\nshipping_intensity'] == 0) )
    ##
    behavior_dict['anamolous routing %.3G m away from coast for cargo ships at sea, where > %g%% waypoints are un-traveled'%(threshold_mks['from coast'], threshold_mks['anamolous routing']*100 )] = \
        (df['anamolous routing'].groupby( df['AIS_downsample\nmmsi']).transform('mean') > threshold_mks['anamolous routing'])& df['anamolous routing']
    ##
    # transshipment close together <100 m and loitering at sea
    behavior_dict['mmsi of transshipment within %.3G m %.3G m away from coast'%(threshold_mks['transshipment'], threshold_mks['from coast'] )] = np.zeros(df.shape[0]).astype(int)
    if not pbar is None:
        pbar.set_postfix_str( '%s transshipment'%pbar.postfix)
    from itertools import groupby
    from operator import itemgetter
    for mmsi_, df_ in df.groupby( 'AIS_downsample\nmmsi'):
        mmsis =  df_[from_col(df, ['neighbor', 'mmsi'])].values
        mmsis[ df_[from_col(df, ['neighbor', 'dist m'])].values > threshold_mks['transshipment'] ] = -1
        mmsis[ ~away_from_coast_loitering[df_.index[0]:df_.index[-1]+1] ]= -1
        ##
        distances = df_[from_col(df, ['neighbor', 'dist'])].to_numpy()
        row_dist = dict()
        weight = dict()
        mmsis_dict = dict()
        closest = [-1]*mmsis.shape[0]
        count = 0
        for uni in np.unique( mmsis ):
            if uni > -1:
                rows, cols = np.where(mmsis == uni)
                for k, g in groupby(enumerate(zip(rows, distances[mmsis==uni]) ), lambda x: x[0] - x[1][0]):
                    row_dist[count] = list( map( itemgetter(1), g ) )
                    weight[count] = len( row_dist[count] ) - sum(map( itemgetter(1), row_dist[count] ))*1E-10
                    mmsis_dict[count] = uni
                    count += 1
        if count == 1:
            for i in map(itemgetter(0), row_dist[0]):
                closest[i] = mmsis_dict[0]
        else:
            for count in sorted(weight, key= weight.get, reverse=True):
                # closest[map( itemgetter(0), row_dist[k] )] = k
                if all([closest[i]==-1 for i in map( itemgetter(0), row_dist[count])] ):
                    for i in map(itemgetter(0), row_dist[count]):
                        closest[i] = mmsis_dict[count]
        behavior_dict['mmsi of transshipment within %.3G m %.3G m away from coast'%(threshold_mks['transshipment'], threshold_mks['from coast'] )][df_.index[0]:df_.index[-1]+1] = closest
        ##

    df_out = pd.DataFrame( behavior_dict, index = df.index )
    df_out.to_feather( out_file )
    if False:
        ##
        for col in df_out:
            if df_out[col].dtype == bool:
                print( '%s : %g%%'%(col, df_out[col].mean()*100 ) )
            elif df_out[col].dtype == object:
                print( '%s : %g%%'%(col, df_out[col].apply( lambda s: not s is None).mean()*100 ) )
            else:
                print('%s : %g%%' % (col, (df_out[col]>0).mean() * 100) )
        ##
    if False:
        ##
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1)

        # df_out['transshipment mmsi 20 m'].plot( kind = 'hist', xlim = [1, df_out['transshipment mmsi 20 m'].max()], bins = 1000, logy = True )
        # plt.show()
        # ##
        mmsi, index, counts = np.unique( df_out['transshipment mmsi 20 m'], return_index=True, return_counts=True )
        plt.bar( range(sum(mmsi>0)), sorted( counts[mmsi>0]) )
        ax.set_ylabel('hours ship was involved in transshipment')
        plt.show()


        ##
        mmsi0 = mmsi[counts==1]
        mmsi1 = 367566940
        ( (df['AIS_downsample\nmmsi'] == mmsi0)&np.any( from_df(df, ['neighbor', 'dist']) < 20, axis = 1 ) ).sum()

        df.loc[(df['AIS_downsample\nmmsi'] == mmsi0)&np.any( from_df(df, ['neighbor', 'dist']) < 20, axis = 1 ), ['AIS_downsample\nmmsi', 'AIS_downsample\ntimestamp']+from_col( df, 'neighbor')]
        ##
        df_out['transshipment mmsi 20 m'][(df['AIS_downsample\nmmsi'] == mmsi1)&np.any( from_df(df, ['neighbor', 'dist']) < 20, axis = 1 )]
        ##
    return df_out, out_file

##
pbar = tqdm( pd.date_range('2022-01-01', '2022-02-01').strftime('%Y-%m-%d').to_list()[:-1], desc = fname )
for date in pbar:
    pbar.set_postfix({'date':date})
    df, filename = get_df( date, freq = 'H', overwrite = True, multicolumn = False,
           threshold_mks = {'loitering':5E3, 'OOB speed': 35, 'OOB ang vel': 45, 'transshipment': 20, 'from coast': 10E3, 'anamolous routing': .5 }, pbar = pbar )
    # break
import shutil
shutil.copy2( fname, os.path.join( os.path.split( filename)[0], fname ) )
##