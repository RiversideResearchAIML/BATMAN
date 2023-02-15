import functions_basic_sak as funsak
import pandas as pd
import os
import inspect
import AIS_interp
import AIS_neighbors
import AIS_parameters
import AIS_trajectory
import AIS_geopandas_tagging
import AIS_numpy_tagging
import AIS_loitering
import AIS_description
import re
import numpy as np
import datetime

fname = os.path.split( inspect.getfile(inspect.currentframe()) )[1]

def multicolumns2single( df_ ):
    df_.columns = ['\n'.join(col_) for col_ in df_.columns.to_list()]
    return df_

def get_df( date_str = '2022_01_01', freq = 'H', overwrite =  False, multicolumn = False ):
    df = []
    file = []
    module = []

    def add_(fnc = None, kwargs = None, df_ = None):
        if df_ is None:
            df_, file_ = fnc(**kwargs)
            module.append(inspect.getmodule(fnc).__name__)
            file.append(file_)
        else:
            module.append( None )
            file.append(None)
        df.append(multicolumns2single( df_ ))

    out_file = os.path.join( AIS_parameters.dirs['root'], 'training', datetime.datetime.now().strftime('%m-%d'), 'AIS_%s (%s %s).feather'%(re.sub( '-', '_', date_str ), fname[:-3], freq) )
    if not os.path.isdir(os.path.split(out_file)[0]):
        os.mkdir(os.path.split(out_file)[0])
    if os.path.exists( out_file ) and not overwrite:
        df_master = pd.read_feather( out_file )
    else:
        add_( AIS_interp.interpolator().get_df,  {'date': date_str, 'freq': freq, 'multicolumn':True} )
        add_( df_=AIS_description.get_df( df = df[0][['AIS_interp\nStatus', 'AIS_interp\nVesselType']].rename(
            columns={'AIS_interp\nStatus': 'Status', 'AIS_interp\nVesselType': 'VesselType'}), multicolumn=True ) )
        add_( AIS_neighbors.get_df,  {'date': date_str, 'freq': freq, 'multicolumn':True, 'n_nearest': 3, 'upper_radius_m': 5E3} )
        add_( AIS_trajectory.get_df,  {'date': date_str, 'freq': freq, 'multicolumn':True} )
        add_( AIS_geopandas_tagging.get_df,  {'date': date_str, 'freq': freq, 'multicolumn':True,
             'tagging': { 'NOAA_MPA_inventory': {'tags': 'Fish_Rstr', 'geometry':'geometry'},
              'Oil_Gas_Pipelines': { 'tags': 'PipelineName', 'geometry': 'geometry', 'buffer': 2E3 },
              'world_ports': {'tags': ['portname', 'geonameid'], 'geometry': 'lat_lon', 'buffer': 10E3 }} } )

        add_( AIS_numpy_tagging.get_df, {'date': date_str, 'freq': freq, 'multicolumn':True, 'files': ['shipping_intensity', 'coast_distance_km', 'earth_elevation_m'] } )
        add_( AIS_loitering.get_df, {'date': date_str, 'freq': freq, 'multicolumn':True, 'radii_m' : 5E3 } )

        ##
        assert all( [df_.shape[0] == df[0].shape[0] for df_ in df] ), 'not all dataframes are same height'
    ##
        df_master = pd.concat( [df_ for df_ in df], axis = 1 )
        df_master.to_feather( out_file )
    ##
    if multicolumn:
        df_master.columns = pd.MultiIndex.from_tuples([tuple(col.split('\n')) for col in df_master.columns])
    return df_master, out_file

if __name__ == '__main__':
    #
    # df = pd.read_feather( '/Users/stephankoehler/Dropbox/RiversideResearch_dark_ships/AIS data/MarineCadastre/training/10-29/AIS_2022_01_01 (AIS_harvest H).feather')
    # ##
    # from collections import defaultdict
    # unique_values = defaultdict(set)
    # for c in df:
    #     unique_values[c].update( df[c].unique().tolist() )
    ##
    if 'stephan' == AIS_parameters.user:
        for date_str in pd.date_range('2022-01-01', '2022-02-01').strftime('%Y-%m-%d').to_list()[:-1]:
            get_df( date_str, freq = 'H', overwrite = True )
##