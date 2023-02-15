import pandas as pd
import geopandas as gpd
import os
import numpy as np
pd.set_option('display.max_columns', 10, 'display.width', 1500)
import re
import xarray
from geocube.api.core import make_geocube
import import_AIS
import tqdm
import functions_basic_sak as funsak
import AIS_infer_mmsi
import AIS_interp
import shapely
import itertools
from shapely.geometry import Point

import inspect
import AIS_parameters

fname = os.path.split( inspect.getfile(inspect.currentframe()) )[1]
##
str2list_strs = lambda str_or_list: [str_or_list] if isinstance( str_or_list, str ) else str_or_list
##


def load_raw_data(gdf_filename, pbar = None ):
    if isinstance(pbar, tqdm.std.tqdm):
        pbar.set_postfix_str('%s loading %s' % (pbar.postfix, os.path.split( gdf_filename)[1] ) )
    return gpd.read_file(gdf_filename).reset_index(drop=True)

def get_gdf( gdf_filename = AIS_parameters.files['world_ports'], tagging_ = None, truncated = True, pbar = None ):

    gdf = None
    if not os.path.exists( os.path.splitext(gdf_filename)[0] + ' trunc.feather' ):
        gdf = load_raw_data(gdf_filename, pbar)
        if isinstance(pbar, tqdm.std.tqdm):
            pbar.set_postfix_str('%s saving to %s' %(pbar.postfix, os.path.splitext(gdf_filename)[0] + ' trunc.feather'))
        drop_columns = []
        for col in gdf.columns:
            if any( [ isinstance( gdf.loc[0, col], cls) for cls in [shapely.geometry.polygon.Polygon, shapely.geometry.point.Point, shapely.geometry.linestring.LineString]] ):
                drop_columns.append(col)
        gdf[[ col for col in gdf.columns if not col in drop_columns]].to_feather(os.path.splitext(gdf_filename)[0] + ' trunc.feather')

    if truncated:
        return pd.read_feather(os.path.splitext(gdf_filename)[0] + ' trunc.feather')
    ##
    if gdf is None:
        gdf = load_raw_data(gdf_filename)
    ##
    if tagging_['geometry'].lower() in ['lat/lon', 'lat_lon', 'lon/lat', 'lon_lat', 'latitude/longitude']:
        gds = gpd.GeoSeries([Point(lon, lat) for lon, lat in zip(gdf.longitude, gdf.latitude)], name='geomtry').set_crs(
            4326)  # coordinate system with meters as units.
    else:
        gds = gpd.GeoSeries(gdf[tagging_['geometry']]).set_crs(4326).to_crs(3857)
    if ('buffer' in tagging_) and (not tagging_['buffer'] is None) and (tagging_['buffer']> 0):
        gds = gds.to_crs(3857).buffer(tagging_['buffer']).to_crs(4326)
    gdf['geometry'] = gds
    gdf = gdf.set_geometry( 'geometry' )
    return gdf
    ##
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1)
    gdf.plot(ax=ax)
    plt.show()

##
def geotag( df, key, tagging_, pbar = None ):
    if not 'res' in tagging_:
        tagging_['res'] = .01
    ##
    gdf_full_filename = AIS_parameters.files[key]
    raster_file = os.path.splitext(gdf_full_filename)[0] + \
        ' (%s %s res %g' % ( fname, tagging_['geometry'], tagging_['res'])
    if ( 'buffer' in tagging_ ) and not ( tagging_['buffer'] is None ) and ( tagging_['buffer'] > 0 ):
        raster_file += ' buffer %g).tif'%tagging_['buffer']
    else:
        raster_file += ').tif'
        tagging_['buffer'] = None
    ##
    if not os.path.exists( raster_file ):
        ##
        gdf = gpd.read_file( gdf_full_filename )
        ##
        import warnings
        warnings.filterwarnings("ignore" )
        gdf['index'] = gdf.index.to_flat_index()
        ##
        out_grid = make_geocube(
            vector_data=gdf.to_crs(4326),
            measurements= ['index'],
            resolution=( -tagging_['res'], tagging_['res'] ),
            fill=-1
        )
        warnings.filterwarnings("default" )
        ##
        if out_grid.index.values.max() < 2**15:
            out_grid.index.values = out_grid.index.values.astype(np.int16)
        out_grid.rio.to_raster(raster_file)
        ##
    gdf = get_gdf( gdf_full_filename, tagging_, truncated=True, pbar = pbar )
    for tag in str2list_strs( tagging_['tags'] ):
        assert tag in gdf,  '%s not in \n\t %s'%(tag, ', '.join( gdf.columns) )

        ##
    ds = xarray.open_dataset(raster_file)
    col = lambda c: np.round( np.interp(c, [ds.x.values[0], ds.x.values[-1]], [0, ds.x.values.shape[0]-1]) ).astype(int)
    row = lambda r: np.round( np.interp(r, [ds.y.values[-1], ds.y.values[0]], [ds.y.values.shape[0]-1, 0]) ).astype(int)
    ##
    values = ds.band_data.values[[0] * df.shape[0], row(df['latitude']), col(df['longitude'])]
    tf = np.isfinite(values)
    df_tagged = pd.DataFrame( [], index = df.index )
    for tag in str2list_strs(tagging_['tags']):
        df_tagged[tag] = np.nan
        df_tagged.loc[tf, tag] = gdf.loc[values[tf], tag].values
    df_tagged.columns = pd.MultiIndex.from_tuples( [(os.path.split( gdf_full_filename)[1], tagging_['geometry'], col) for col in list( df_tagged.columns ) ] )
    return df_tagged

def get_df_old(  date = None, df = None,
    tagging= { 'Oil_Gas_Pipelines': { 'tags': ['portname', 'geonameid'], 'geometry': 'geometry', 'buffer': 10E3, 'res': .01 } },
    freq = None, overwrite = False, pbar = None, join = False, multicolumn=True ):
    if isinstance( date, list ):
        pbar = tqdm.tqdm( date, desc = fname )
        dfs = dict()
        out_files = dict()
        for date_ in pbar:
            dfs[date_], out_files[date_] = get_df(date = date_, df = df,
                tagging= tagging, freq = freq,  overwrite = overwrite, pbar = pbar, join = join, multicolumn=multicolumn )
        return dfs, out_files
    else:
        df_out = None
        df = None
        date_str = pd.to_datetime(re.sub( '_', ' ', date)).strftime('AIS_%Y_%m_%d')
        if freq:
            out_file = os.path.join( AIS_parameters.dirs['root'], 'downsample', '%s (%s %s [%s]).feather'%(date_str, fname[:-3], freq, ', '.join( tagging.keys() ) ) )
        else:
            out_file = os.path.join( AIS_parameters.dirs['root'], 'ingested', '%s (%s [%s]).feather'%(date_str, fname[:-3], ', '.join( tagging.keys() ) ) )
        if not os.path.isdir( os.path.split( out_file)[0] ):
            os.mkdir(  os.path.split( out_file)[0] )
        if os.path.exists(out_file) and not overwrite :
            ##
            df_out = pd.read_feather(out_file)
            ##
            desired_columns = []
            for key in tagging:
                for tag in str2list_strs( tagging[key]['tags'] ):
                    desired_columns.append('%s\n%s\n%s'%(os.path.split(AIS_parameters.files[key])[1], tagging[key]['geometry'], tag))
            ##
            if all( [col in df_out.columns for col in desired_columns] ):
                df_out = funsak.AIS_rename( df_out[desired_columns] )
        if df_out is None:
            if df is None:
                if freq:
                    if pbar:
                        pbar.set_postfix({'date': date, 'freq': freq })
                    df, _ = AIS_downsample.get_df(date, freq=freq, pbar = pbar)
                else:
                    df, _ = import_AIS.get_df(date, pbar = pbar)

            dfs_tagged = []
            if pbar:
                pbar_ = tagging.keys()
            else:
                pbar_ = tqdm.tqdm( tagging.keys(), desc = '%s geo-tagging'%fname, total = len( tagging.keys() ) )
                pbar = pbar_
            for key in pbar_:
                dict_ = {'date': date, 'gdf_filename': os.path.split(key)[1] }
                if 'geometry' in tagging[key]:
                    dict_['geometry'] = tagging[key]['geometry']
                else:
                    dict_['geometry'] = 'geometry'
                if 'buffer' in tagging[key]:
                    dict_['buffer'] = tagging[key]['buffer']
                pbar.set_postfix( dict_ )
                dfs_tagged.append( geotag( df, key, tagging[key], pbar = pbar ) )
            df_out = pd.concat( dfs_tagged, axis = 1).astype( "category")
            columns = df_out.columns
            df_out.columns = ['\n'.join(l) for l in columns.to_list()]
            df_out.to_feather(out_file)
            df_out.columns = columns
        if join:
            if df is None:
                df, _ = import_AIS.get_df(date, pbar = pbar)
            df_out = pd.concat( [df, df_out], axis = 1)

        return df_out, out_file

def get_df( df = None,
    tagging= { 'Oil_Gas_Pipelines': { 'tags': ['portname', 'geonameid'], 'geometry': 'geometry', 'buffer': 10E3, 'res': .01 } },
    pbar = None, join = False, multicolumn=True ):
            ##
    desired_columns = []
    for key in tagging:
        for tag in str2list_strs( tagging[key]['tags'] ):
            desired_columns.append('%s\n%s\n%s'%(os.path.split(AIS_parameters.files[key])[1], tagging[key]['geometry'], tag))
        ##

    dfs_tagged = []
    if isinstance(pbar, tqdm.std.tqdm):
        pbar_ = tagging.keys()
        postfix_str = str(pbar.postfix)
    elif isinstance(pbar, type(None)) or (isinstance(pbar, bool) and pbar):
        pbar = tqdm.tqdm(tagging.keys(), desc = '%s geo-tagging'%fname, total = len( tagging.keys() ) )
        pbar_ = pbar
        postfix_str = ''
    for key in pbar_:
        if isinstance(pbar, tqdm.std.tqdm):
            pbar.set_postfix_str(postfix_str + '... %s' % key )
        dfs_tagged.append( geotag( df, key, tagging[key] ) )
    df_out = pd.concat( dfs_tagged, axis = 1).astype( "category")
    if join:
        df_out = pd.concat( [df, df_out], axis = 1)
    if multicolumn == False:
        df_out.columns = ['\n'.join(l) for l in df_out.columns.to_list()]
    return df_out
##

if __name__ == '__main__':
    if 'stephan' == AIS_parameters.user:
        # linestr2polygon(in_filename=AIS_parameters.files['Oil_Gas_Pipelines'], column='geometry', width_m=1E3,
        #                 overwrite=True, rows=None)
        # dates = ['2020_02_01', '2020_02_02']
        DF, df = AIS_interp.get_interp(time=pd.date_range('2022-01-01', '2022-01-02', freq='H')[:-1],
                            return_df=True, aux_columns=None)
        dfs_out = get_df(df = DF, tagging={'World_EEZ': {'tags': 'ISO_SOV1', 'geometry': 'geometry'}}, join=True, multicolumn=True)
        # gdf_filenames =[ 'World_EEZ', 'World_Heritage_Marine']
        # dates = ['2022_01_01']
        #
        # gdf_filenames = ['world_ports']
        # tags = [['portname', 'geonameid']]
        # geometry_columns = [['lon_lat']]
        # buffers = [[1E3]]
        #
        # gdf_filenames = ['Oil_Gas_Pipelines']
        # tags = [['PipelineName']]
        # geometry_columns = [['geometry']]
        # buffers = [[10E3]]
        ##
        # tagging = { 'Oil_Gas_Pipelines': { 'tags': ['portname', 'geonameid'], 'geometry': 'geometry', 'buffer': 10E3 } }
        # ##
        # dfs_out, filenames_out = get_df(date, tagging = {
        #       'WPI_Shapefile2019': {'tags': ['PORT_NAME', 'COUNTRY'], 'geometry': 'geometry', 'buffer': 10E3 }}, overwrite = True, freq = None, join = False )
        #
        # ##
        # dfs_out, filenames_out = get_df(date, tagging = { 'NOAA_MPA_inventory': {'tags': 'Fish_Rstr', 'geometry':'geometry'},
        #       'Oil_Gas_Pipelines': { 'tags': 'PipelineName', 'geometry': 'geometry', 'buffer': 5E3 },
        #       'world_ports': {'tags': ['portname', 'geonameid'], 'geometry': 'lat_lon', 'buffer': 10E3 }}, overwrite = True, freq = None, join = False )
        print(dfs_out.head())


