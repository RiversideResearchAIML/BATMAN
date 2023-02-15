import numpy as np
import functions_basic_sak as funsak
import pandas as pd
from tqdm import tqdm
import os
from skimage import data, util, measure
from scipy.ndimage.measurements import label
import inspect
import pathlib
import AIS_parameters
import AIS_interp
import re
import import_AIS
import matplotlib.pyplot as plt

fname = os.path.split( inspect.getfile(inspect.currentframe()) )[1]
st_mtime = pathlib.Path(fname).stat().st_mtime

pd.set_option('display.max_columns', 20, 'display.width', 1500)

convert_list = lambda list_: list_ if isinstance( list_, list ) else [list_]
##
def loitering(df = None,  radii_m = 1E3, pbar = False ):
    # df = df.iloc[:10000].copy()
    df_ = dict()
    radii_m = convert_list(radii_m)
    for radius_m in radii_m:
        ##
        df_['loit time for %g m' % (radius_m) ] = np.tile( np.NaN, df.shape[0] ).astype(np.float32)
        df_['loit dist for %g m' % (radius_m) ] = np.tile( np.NaN, df.shape[0] ).astype(np.float32)
        df_['loit index for %g m' % (radius_m)] = -np.ones( df.shape[0] ).astype( np.int32 )

    ##
    df['longitude_radians'] = np.radians( df['longitude'].to_numpy(float) )
    df['latitude_radians'] = np.radians(df['latitude'].to_numpy(float))
    df['sin_latitude'] = np.sin( df['latitude_radians'] )
    df['cos_latitude'] = np.cos( df['latitude_radians'] )
    df['cum time sec'] = np.maximum(0, df['timestamp'].diff().to_numpy(int) / 1E9).cumsum()
    h_structure = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
    if pbar:
        pbar_ = df.groupby( 'mmsi')
        flag_pbar = False
        pbar.set_postfix_str( '%s radii_m = %s'%(pbar.postfix, str(radii_m) ) )
    else:
        pbar_ = tqdm(df.groupby('mmsi'), desc='loitering( radii_m = %s )' % str(radii_m))
        pbar = pbar_
        flag_pbar = True
    for mmsi_, df_gb in pbar_:
        if flag_pbar:
            pbar.set_postfix({'mmsi': '%i'%mmsi_, '# rows': df_gb.shape[0]})
        ##
        dist_m = funsak.distance_cosine_formula(lon1=df_gb['longitude_radians'], lat1=df_gb['latitude_radians'],
                sin_lat1=df_gb['sin_latitude'], cos_lat1=df_gb['cos_latitude'],
                                meshgrid=True, units='m', convert2radians=False)
        ##
        for radius_m in radii_m:
            ##
            tf = dist_m < radius_m
            labeled, _ = label(tf, h_structure)
            cum_time = df_gb['cum time sec'].to_numpy(float)
            cum_time -= cum_time[0]
            ##
            if False:
                ##
                import matplotlib.pyplot as plt

                plt.imshow(labeled)
                plt.show()
            ##
            props = [props_ for props_ in measure.regionprops(labeled, intensity_image=np.tile(cum_time,  (df_gb.shape[0], 1)) ) if ( props_.bbox[0] == props_.bbox[1] ) and ( props_.area>1 )]
            dt_props = np.array([ props_.mean_intensity * props_.area for props_ in props])
            ##
            for i, r in enumerate( np.argsort(-dt_props)):
                a = props[r].bbox[1]+df_gb.index.values[0]
                b = props[r].bbox[3]+df_gb.index.values[0]
                if all( df_['loit index for %g m' % (radius_m)][a:b] == -1):
                    df_['loit time for %g m' % (radius_m )][a:b] = cum_time[props[r].bbox[1]:props[r].bbox[3]] - cum_time[props[r].bbox[1]]
                    df_['loit dist for %g m' % (radius_m )][a:b] = dist_m[props[r].bbox[0], props[r].bbox[1]:props[r].bbox[3]]
                    df_['loit index for %g m' % (radius_m)][a:b] = i
        ##
    ##
    df_ = pd.DataFrame( df_, index = df.index )
    return df_
##
def get_df( date = '2022_06_29', radii_m = [10], freq = None, format = 'feather', overwrite = False,
            pbar = None, multicolumn = False, nrows = None, join = False ):
    '''

    Parameters
    ----------
    date: either string or list of strings representing desired date(s), e.g. '2022_06_29' or ['2022_06_29', '2021_06_29', '2020_06_29']
    format: 'feather', 'h5' or 'pkl'
    MarineCadaster_URL: this is intended for Marinecadastre for now, will overwrite with Danish_URL later on.
    download_dir

    Returns either single df or dict of desired dataframes
    -------

    '''
    ##
    radii_m = sorted( convert_list( radii_m) )
    if isinstance(date, list):
        df_dict = {}
        filename_dict = {}
        pbar = tqdm( date, desc = '%s get_df'%fname )
        for date_ in pbar:
            pbar.set_postfix( {'date': date_} )
            df_dict[date_], filename_dict[date_] = get_df( date_,  radii_m = radii_m, format = format, freq = freq,
               overwrite=overwrite, pbar=pbar, multicolumn=multicolumn, nrows=nrows )
        return df_dict, filename_dict
    else:
        date_str = pd.to_datetime(re.sub( '_', ' ', date)).strftime('AIS_%Y_%m_%d')
        ##
        if freq is None:
            out_file = os.path.join( AIS_parameters.dirs['root'], 'ingested', '%s (%s %s m).%s' % (date_str, fname, ', '.join( ['%.2G'%radius_m for radius_m in radii_m]), format)  )
        else:
            out_file = os.path.join(AIS_parameters.dirs['root'], 'resampled',
                '%s (%s %s %s m).%s' % (date_str, fname, freq, ', '.join(['%.2G' % radius_m for radius_m in radii_m]), format)  )
        if not os.path.isdir( os.path.split( out_file)[0] ):
            os.mkdir(  os.path.split( out_file)[0] )

        if join or overwrite or not os.path.exists(out_file):
            if freq:
                df, _ = AIS_interp.get_interp(date, freq=freq, pbar=pbar)
            else:
                df, _ = import_AIS.get_df( date, pbar = False, nrows=nrows )


        if os.path.exists(out_file) and not overwrite:
            df_ = pd.read_feather(out_file)
        else:
            df_ = loitering( df, radii_m = radii_m, pbar = pbar )
            df_.to_feather(out_file)
        if multicolumn:
            df_.columns = pd.MultiIndex.from_product([[fname[:-3]], df_.columns])
        if join:
            df_ = df.join(df_)
        return df_, out_file
        ##
##
if __name__ == '__main__':
    if 'stephan' == AIS_parameters.user:
        dates = ['2022_01_01', '2022_01_02']
        date = '2022_01_02'
        df, filename = get_df( date = date, radii_m = 5E3, nrows = None, overwrite = False, freq = None, join = True )
        print(filename)
        ##
        for mmsi_, df_ in df.groupby( 'mmsi'):
            if (df_.shape[0] > 400) and any( df_.iloc[:, -3] > 1E4 ) and any( df_.iloc[:, -1] > 1 ):
                ##
                from matplotlib.collections import LineCollection
                fig, ax = plt.subplots(1,1, figsize = [10, 6])
                points = df_[['longitude', 'latitude']].to_numpy().reshape(-1, 1, 2)
                t = df_.timestamp.to_numpy(int)
                segs = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segs, cmap=plt.get_cmap('jet'))
                lc.set_array(t)  # color the segments by our parameter
                # plot the collection
                plt.gca().add_collection(lc)  # add the collection to the plot
                ax.set_xlim( df_.longitude.min(), df_.longitude.max())
                ax.set_ylim( df_.latitude.min(), df_.latitude.max())
                km_lon = funsak.distance_haversine(lon1 = df_.longitude.min(), lon2 = df_.longitude.max(),
                                               lat1 = df_.latitude.min(), lat2 = df_.latitude.min(), convert2radians=True)/1000
                ax.set_xticks([df_.longitude.min(), df_.longitude.max()],
                              labels=['%.4f deg' % df_.longitude.min(),
                                      '%.4f deg\n%.1f km' % (df_.longitude.max(),
                                           km_lon)],
                              horizontalalignment='right',
                              verticalalignment='top')
                km_lat = funsak.distance_haversine(lon1 = df_.longitude.min(), lon2 = df_.longitude.min(),
                                               lat1 = df_.latitude.min(), lat2 = df_.latitude.max(), convert2radians=True)/1000
                ax.set_yticks([ df_.latitude.min(), df_.latitude.max()],
                              labels=['%.4f deg' % df_.latitude.min(),
                                      '%.4f deg\n%.1f km' % (df_.latitude.max(), km_lat)],
                              horizontalalignment='right',
                              verticalalignment='bottom')

                cb = plt.colorbar(lc, ticks=[t[0], t[-1]])
                cb.ax.set_yticklabels([df_.timestamp.min().strftime( '%H:%M'), df_.timestamp.max().strftime( '%H:%M')])
                cb.ax.set_ylabel('time (%s)'%df_.timestamp.min().strftime( '%Y-%m-%d'))
                for loit_, df__ in df_.groupby( 'loit index for 5000 m'):
                    if loit_ >= 0:
                        label = '%i: %s - %s, %g km'%(loit_, df__.iloc[0].timestamp.strftime( '%H:%M'), df__.iloc[-1].timestamp.strftime( '%H:%M'), df__.iloc[-1,-2]/1E3 )
                        ax.plot(df__.longitude, df__.latitude, 'o', linewidth=3, label = label)
                        ax.text(df__.longitude.mean(), df__.latitude.mean(), loit_ )
                    distance = funsak.distance_haversine(lat1=df__.latitude, lon1=df__.longitude, convert2radians=True, meshgrid=True,units='m' )
                    print( loit_, distance.flatten().max() )
                ax.set_aspect( km_lat/km_lon )
                ax.legend(loc=[0.2, 1.1], fontsize='small',
                          title='{}, {}'.format( *df_.iloc[0][['mmsi', 'VesselName']].values), handlelength=4)
                ax.set_xlabel('longitude')
                ax.set_ylabel('latitude')
                plt.show()
                out_file = os.path.join( fname[:-3], '%s %i.tif'%(df_.iloc[0].timestamp.strftime('%Y-%m-%d'), df_.mmsi.values[0]) )
                funsak.create_dir4file(out_file)
                fig.savefig(out_file, dpi=100)

        ##
##
