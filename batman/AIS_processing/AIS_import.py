import pandas as pd
import numpy as np
import ciso8601
import functions_basic_sak as funsak
import inspect
import os
import pathlib
import wget
import requests
import re
from tqdm import tqdm
import AIS_parameters
import AIS_function_wrapper

pd.set_option('display.max_columns', 20, 'display.width', 1500)


fname = os.path.split( inspect.getfile(inspect.currentframe()) )[1]
st_mtime = pathlib.Path(fname).stat().st_mtime

URL_cache = ''
requests_ = ''
def download_from_URL( date = '2022_06_30', pbar = None ):
    ##
    date_str = pd.to_datetime(re.sub( '_', ' ', date)).strftime('AIS_%Y_%m_%d')
    ##
    global URL_cache
    global requests_
    ##
    if pbar is None:
        print(  'download %s from MarineCadaster'%date )
    elif isinstance(pbar, bool ) and ( pbar != False ):
        pbar.set_postfix( {'download from MarineCadaster': date } )
    URL = AIS_parameters.dirs['MarineCadaster_URL']+pd.to_datetime(re.sub( '_', ' ', date)).strftime('%Y')
    if URL != URL_cache:
        requests_ = requests.get(URL).text
        URL_cache = URL
    ##
    link = set([os.path.join(URL, f) for f in re.findall(date_str, requests_)])
    ##
    out = os.path.join(AIS_parameters.dirs['root'], 'download', os.path.split(list(link)[0])[1])+'.zip'
    while not os.path.exists( out ):
        try:
            wget.download( list(link)[0]+'.zip', out = out )
        except:
            import time
            time.sleep( 5 )
            print( 'trying again')
    return out
    ##

def extract_df( source = None, format = 'feather', update = False, overwrite = False, delete_source = False, nrows = None, pbar = None, return_filename = False ):
    out_file = os.path.join( AIS_parameters.dirs['root'], fname[:-3], os.path.splitext(os.path.split(source)[1])[0]+'.'+format)
    if return_filename:
        return out_file
    if not os.path.isdir(os.path.split(out_file)[0]):
        os.mkdir( os.path.split(out_file)[0])

    if overwrite or not os.path.exists( out_file ):# or (update and ( st_mtime > pathlib.Path(out_file).stat().st_mtime )):
        if source.endswith( '.zip' ) or source.endswith( '.csv' ):
            try:
                df = pd.read_csv(source, engine='c', low_memory=False, nrows=1000 )
            except:
                assert False, '%s does not exist'%source
            ##
            time_columns = []
            dtypes = df.dtypes.to_dict()
            for col, dtype in dtypes.items():
                if col.lower() == 'mmsi':
                    dtypes[col] = float
                elif col.lower().startswith( 'lat') or col.lower().startswith( 'lon'):
                    pass
                elif dtype in [int, float]:
                    dtypes[col] = np.float16
                else:
                    try:
                        datetimes = [ciso8601.parse_datetime( val ) for val in df[col].dropna().iloc[:2]]
                        dtypes[col] = str
                        time_columns.append( col )
                    except:
                        dtypes[col] = 'category'
                    ##
            if pbar is None:
                print( 'read_csv %s'%source )
            else:
                pbar.set_postfix({'read_csv': source})
            ## to deal with corrupted data
            df = pd.read_csv( source, engine='c', dtype=dtypes, nrows = nrows, on_bad_lines = 'skip' )

            ##
            for col in time_columns:
                df[col] = funsak.get_datetime(df[col], pbar = pbar )
            ## here safely convert mmsi from float to np.int32
        df_log = pd.DataFrame( df.isna().sum(), columns = ['isna().sum()'] ).rename_axis( 'orig. name')
        total_count = [df.shape[0], '']
        df = funsak.AIS_rename(df).dropna( subset = ['latitude', 'longitude', 'mmsi', 'timestamp'], how = 'any' ).drop_duplicates(subset=['mmsi', 'timestamp']).sort_values( ['mmsi', 'timestamp'] ).reset_index().astype( {'mmsi': np.int32})
        df_log['new name'] = df.drop(columns='index').columns
        df_log['new isna().sum'] = df.drop(columns='index').isna().sum().to_list()
        total_count.append( df.shape[0] )
        df_log = pd.concat([df_log, pd.DataFrame(total_count, index=df_log.columns, columns=['total count']).T], axis=0)
        df_log.to_csv( os.path.splitext(out_file)[0]+ ' log.csv' )
        if format.endswith('feather'):
            df.to_feather( out_file )
        elif format.endswith('pkl'):
            df.to_pickle(out_file)
        elif format.lower().endswith('h5'):
            df.to_hdf(out_file, key = 'df', format = 'table', mode = 'w', complevel = 9)

    elif out_file.endswith('feather'):
        df = pd.read_feather( out_file )
    elif out_file.endswith( '.pkl'):
        df = pd.read_pickle( out_file )
    elif out_file.lower().endswith( '.h5'):
        df = pd.read_hdf( out_file, 'df' )
    if delete_source:
        os.remove(source)
    return df, out_file


def get_df( date = '2022_06_29', format = 'feather', pbar = None, return_filename = False ):
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
    if isinstance( date, str ):
        date_str = pd.to_datetime(re.sub( '_', ' ', date)).strftime('AIS_%Y_%m_%d')
    elif isinstance( date, np.datetime64 ):
        date_str = pd.Timestamp(date).strftime('AIS_%Y_%m_%d')
    else:
        date_str = date.strftime('AIS_%Y_%m_%d')
    ##
    if not os.path.isdir( os.path.join( AIS_parameters.dirs['root'], 'download') ):
        os.mkdir(os.path.join( AIS_parameters.dirs['root'], 'download'))
    source_file = [file for file in os.listdir(os.path.join( AIS_parameters.dirs['root'], 'download')) if file.startswith(date_str) and ( file.endswith( '.csv') or file.endswith( '.zip'))]
    if return_filename:
        return extract_df(source=source_file, format=format, pbar=pbar, return_filename=return_filename)
    if source_file == []:
        source_file = download_from_URL( date_str[4:], pbar  )
    else:
        source_file = os.path.join( AIS_parameters.dirs['root'], 'download', source_file[0])
    df, out_file = extract_df( source = source_file, format = format, pbar = pbar, return_filename = return_filename )
    return df, out_file

##

if __name__ == '__main__':
    if 'stephan' == AIS_parameters.user:
        get_df( '2022-06-03')
        # print( get_df(date=pd.date_range('2022-01-01', '2022-02-01').strftime('%Y-%m-%d').to_list()[:-1], format='feather')[0].keys() )