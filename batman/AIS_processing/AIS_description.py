import pandas as pd
import os
import inspect
import AIS_parameters
import AIS_import
import numpy as np
import re

fname = os.path.split( inspect.getfile(inspect.currentframe()) )[1]
##
df_vessel_types = None
df_vessel_status = None


def clean_vessel_type_str(str_):
    sub_list = ['all ships of this type', 'not available or no ship, default', '\(.*\)',
                'for assignment to local vessel',
                'reserved for future use', 'no additional information', 'reserved for regional use', 'null',
                'no designation']
    if isinstance(str_, str):
        str_ = str_.lower()
        for s in sub_list:
            str_ = re.sub('sar\s', 'SAR ', re.sub('\s+', ' ', re.sub('[\s\,\.]*%s[\s\,\.]*' % s, ' ', str_)))
    else:
        str_ = ''
    return str_.strip()


# df = pd.read_excel(AIS_parameters.files['vessel_types'], na_values='- ')
# list( df['AIS Vessel Code AIS Ship & Cargo Classification'].apply( clean_vessel_type_str ).drop_duplicates() )
##
def clean_vessel_status_str(str_):
    sub_list = ['reserved for future amendment of navigational status for ships.*', 'undefined.*', '\\xa0']
    if isinstance(str_, str):
        str_ = str_.lower()
        for s in sub_list:
            str_ = re.sub('sar\s', 'SAR ', re.sub('\s+', ' ', re.sub('[\s\,\.]*%s[\s\,\.]*' % s, ' ', str_)))
    else:
        str_ = ''
    return str_.strip()

def get_df( date = None, df = None, multicolumn = False, join = False, clean = True, pbar = None ):
    global df_vessel_types
    global df_vessel_status
    if df_vessel_types is None:
        df_vessel_types = pd.read_excel( AIS_parameters.files['vessel_types'], na_values='- ').dropna( subset = ['AIS Vessel Type'])
        df_vessel_types['cleaned'] = df_vessel_types['AIS Vessel Code AIS Ship & Cargo Classification'].apply(clean_vessel_type_str)
    if df_vessel_status is None:
        df_vessel_status = pd.read_excel(AIS_parameters.files['vessel_status'] )
        df_vessel_status['cleaned'] = df_vessel_status['description'].apply(
            clean_vessel_status_str)
    if df is None:
        df, filename = AIS_import.get_df( date )
    ##
    class mydict(dict):
        def __missing__(self, key):
            return str(key)
    dict_vt = mydict()
    #note duplicates for 'AIS Vessel Code AIS Ship & Cargo Classification'
    if clean:
        column = 'cleaned'
    else:
        column = 'AIS Vessel Code AIS Ship & Cargo Classification'
    for tpl in df_vessel_types[['AIS Vessel Type', column]].itertuples():
        if isinstance( tpl[1], int ):
            dict_vt[tpl[1]] = ('%s (%s)'%(tpl[2], str(tpl[1]) )).strip()
    dict_vs = mydict()
    if clean:
        column = 'cleaned'
    else:
        column = 'description'
    for tpl in df_vessel_status[['code', column]].itertuples():
        if isinstance( tpl[1], int ):
            dict_vs[tpl[1]] = ('%s (%s)'%(tpl[2], str(tpl[1]) )).strip()
    ##
    column_VesselType = [col for col in df.columns.to_list() if 'VesselType' in col][0]
    column_Status = [col for col in df.columns.to_list() if 'Status' in col][0]
    df_out = df[[column_VesselType, column_Status]].copy().astype( "category" )
    df_out.columns = ['VesselType', 'Status']
    df_out.VesselType.cat.categories = [dict_vt[c] for c in df_out.VesselType.cat.categories]
    df_out.Status.cat.categories = [dict_vs[c] for c in df_out.Status.cat.categories]
    df_out
    if multicolumn:
        df_out.columns = pd.MultiIndex.from_product([[fname[:-3]], df_out.columns.to_flat_index()])
    if join:
        df_out = pd.concat([df, df_out], axis=1)
    df_out
    ##
    return df_out
##

if __name__ == '__main__':

    ##
    if 'stephan' == AIS_parameters.user:
        df = get_df( date = '2022_01_01', join = True )
        print(df)
