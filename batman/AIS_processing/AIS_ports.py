import pandas as pd
import numpy as np
import functions_basic_sak as funsak
import import_AIS
from tqdm import tqdm
import os
import inspect
import pathlib
import re
import AIS_parameters

pd.set_option('display.max_columns', 20, 'display.width', 1500)

fname = os.path.split( inspect.getfile(inspect.currentframe()) )[1]
st_mtime = pathlib.Path(fname).stat().st_mtime

##
df_world_ports = None
n_nearest = 1
upper_radius_km = 10
def port_neighborhood( df, n_nearest = 2, upper_radius_km = upper_radius_km,  multicolumn = False ):
    bubble_dist, bubble_ind, bubble_count, near_dist, near_ind = funsak.neighborhood(df =df, df_centers=df_world_ports,
        n_nearest = n_nearest, convert2radians=True, upper_radius=upper_radius_km/funsak.earth_radius_km,
        limit_n_nearest2upper_radius = True)
    df_ = df[[]].copy()
    df_['port count within %g km' % ( upper_radius_km)] = np.uint16(bubble_count)
    for i in range( n_nearest ):
        df_['port #%i within %g km dist'%(i, upper_radius_km)] = np.float16( near_dist[:,i]*funsak.earth_radius_km )
        df_['port #%i within %g km index'%(i, upper_radius_km)] = np.int16(near_ind[:,i])
    ##
    if multicolumn:
        df_.columns = pd.MultiIndex.from_product([[fname[:-3]], df_.columns])
    return df_
    ##

def get_df( date = None,  df = None, n_nearest = 2, upper_radius_km = 10, freq = None, columns =  ['portname', 'country'], format = 'feather', update = False,
            pbar = None, multicolumn = False, nrows = None ):
    '''

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
    global df_world_ports
    if df_world_ports is None:
        df_world_ports = pd.read_csv(AIS_parameters.files['world_ports']).dropna( how = 'any', subset = ['latitude', 'longitude'])

    if isinstance(date, list):
        df_dict = {}
        filename_dict = {}
        pbar = tqdm( date, desc = '%s get_df'%fname )
        for date_ in pbar:
            pbar.set_postfix( {'date': date_} )
            df_dict[date_], filename_dict[date_] = get_df( date_,  n_nearest = n_nearest, upper_radius_km = upper_radius_km, freq= freq, columns =  columns, format = format,
               update=update, pbar=pbar, multicolumn=multicolumn, nrows=nrows )
        return df_dict, filename_dict
    else:
        date_str = pd.to_datetime(re.sub( '_', ' ', date)).strftime('AIS_%Y_%m_%d')
        ##
        if freq:
            out_file = os.path.join( AIS_parameters.dirs['root'], 'resampled', '%s (%s %s %g %g).%s'%(date_str, fname[:-3], freq, n_nearest, upper_radius_km, format) )
        else:
            out_file = os.path.join( AIS_parameters.dirs['root'], 'ingested', '%s (%s %g %g).%s'%(date_str, fname[:-3], n_nearest, upper_radius_km, format) )
        if not os.path.isdir( os.path.split( out_file)[0] ):
            os.mkdir(  os.path.split( out_file)[0] )
        if os.path.exists(out_file) and not (update and (st_mtime > pathlib.Path(out_file).stat().st_mtime)):
            df_ = pd.read_feather(out_file)
        else:
            if df is None:
                df, filename = import_AIS.get_df(date, pbar=False)
                if nrows:
                    df = df.iloc[:nrows]

            if pbar:
                pbar.set_postfix_str( 'calculating neighborhood' )
            else:
                print( 'calculating neighborhood' )
            df_ = port_neighborhood( df, n_nearest = n_nearest, upper_radius_km = upper_radius_km, multicolumn=False )
            if format == 'feather':
                df_.to_feather(out_file)
        if multicolumn:
            df_.columns = pd.MultiIndex.from_product([[fname[:-3]], df_.columns])
        return df_, out_file
        ##

##
def port_plotly():
    import plotly.express as px
    import plotly

    fig = px.scatter_mapbox(df_world_ports, lat=df_world_ports.latitude, lon=df_world_ports.longitude, hover_name='portname',
                            hover_data=['prttype', 'prtsize', 'country'], zoom=1, mapbox_style="carto-positron")
    plotly.offline.plot(fig, filename='world_ports.html')

if __name__ == '__main__':
    user = 'stephan all'
    user = 'stephan'
    if 'stephan' == AIS_parameters.user:
        df, filename = get_df( funsak.missed_dates(fname), upper_radius_km=10, n_nearest=2, nrows = None)
    elif user == 'stephan':
            ##
            dates = ['2020_02_01', '2020_02_02']
            df, filename = get_df(date = dates, upper_radius_km=10, freq = 'H', n_nearest=2, nrows=None)
            # port_plotly()
    elif user == 'bayley':
        pass
##
