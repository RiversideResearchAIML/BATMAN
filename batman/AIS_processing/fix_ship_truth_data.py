import pandas as pd
import numpy as np
import AIS_parameters
# import ship_pairing_analysis
import PIL
import tqdm
import functions_basic_sak as fun_sak
import os


##
jpg_lookup = fun_sak.file_lookup(endswith='.jpg')
##
df = pd.read_feather( AIS_parameters.files['ship_truth_data_original'])
##
tf = df.lon1 > df.lon2
df.loc[tf, ['lon1', 'lon2']] = df.loc[tf, ['lon2', 'lon1']].to_numpy()
##
tf = df.lat1 < df.lat2
df.loc[tf, ['lat1', 'lat2']] = df.loc[tf, ['lat2', 'lat1']].to_numpy()
##
df['x'] = df['ship_pixel_loc'].apply( lambda p: p[0])
df['y'] = df['ship_pixel_loc'].apply( lambda p: p[1])
df['diameter'] = np.hypot( df.x2 - df.x1, df.y2 - df.y1 )
df.drop( columns = ['ship_pixel_loc', 'x1', 'x2', 'y1', 'y2', 'image_w', 'image_h', 'meters_per_pixel'], inplace = True)
df['top width m'] = fun_sak.distance_haversine(df.lon1, df.lon2, df.lat2, df.lat2, convert2radians=True)
df['bottom width m'] = fun_sak.distance_haversine(df.lon1, df.lon2, df.lat1, df.lat1, convert2radians=True)
df['height m'] = fun_sak.distance_haversine(df.lon1, df.lon1, df.lat1, df.lat2, convert2radians=True)
for i, (filename_, df_) in enumerate( tqdm.tqdm( df.groupby( 'filename') ) ):
    im = PIL.Image.open(jpg_lookup.get(filename_) )
    df.loc[df_.index.to_flat_index(), 'longitude'] = np.interp(df_.x, [0, im.size[0] - 1], [df_.lon1.values[0], df_.lon2.values[0]])
    assert np.isclose( np.interp(df.loc[df_.index.to_flat_index(), 'longitude'], [df_.lon1.values[0], df_.lon2.values[0]], [0, im.size[0] - 1]),  df.loc[
        df_.index.to_flat_index(), 'x'], atol = .0001 ).all()
        ##
    df.loc[df_.index.to_flat_index(), 'latitude'] = np.interp(df_.y, [0, im.size[1] - 1], [df_.lat2.values[0], df_.lat1.values[0]])
    assert np.isclose( np.interp(df.loc[df_.index.to_flat_index(), 'latitude'], [df_.lat2.values[0], df_.lat1.values[0]], [0, im.size[1] - 1]),  df.loc[
        df_.index.to_flat_index(), 'y'], atol = .0001 ).all()
##
df.to_feather( AIS_parameters.files['ship_truth_data'] )
##

