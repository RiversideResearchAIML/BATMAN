# Here's the interactive version. Edit: Fixed bug producing extra spaces in Matplotlib 3.


def rainbow_text(x,y,ls,lc, version = 'horizontal',**kw):
    """
    Take a list of strings ``ls`` and colors ``lc`` and place them next to each
    other, with text ls[i] being shown in color lc[i].

    This example shows how to do both vertical and horizontal text, and will
    pass all keyword arguments to plt.text, so you can set the font size,
    family, etc.
    """
    t = plt.gca().transData
    fig = plt.gcf()
    plt.show()
    if version.lower().startswith('h'):
        #horizontal version
        for s,c in zip(ls,lc):
            text = plt.text(x,y,s+" ",color=c, transform=t, **kw)
            text.draw(fig.canvas.get_renderer())
            ex = text.get_window_extent()
            t = transforms.offset_copy(text._transform, x=ex.width, units='dots')
    else:
        #vertical version
        for s,c in zip(ls,lc):
            text = plt.text(x,y,s+" ",color=c, transform=t,
                    rotation=90,va='bottom',ha='center',**kw)
            text.draw(fig.canvas.get_renderer())
            ex = text.get_window_extent()
            t = transforms.offset_copy(text._transform, y=ex.height, units='dots')


##
import ship_pairing_analysis
# https://stackoverflow.com/questions/9169052/partial-coloring-of-text-in-matplotlib for coloring characters inside text
import collections
import AIS_description
import functions_basic_sak as fun_sak
import matplotlib.pyplot as plt
from matplotlib import transforms
import pandas as pd
import AIS_parameters
import os
import numpy as np
import tqdm
import functions_basic_sak as funsak
import import_AIS
import AIS_description
import AIS_infer_mmsi
import re
import PIL
from PIL import ImageStat
import functions_basic_sak as fun_sak
import AIS_interp
import ship_pairing_analysis
from scipy import interpolate

import warnings
import matplotlib
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
jpg_lookup = None

def crop_line_into_frame(x=np.array([.5, 1.5]), y=np.array([-.1, .5]), X=np.array([0, 1]), Y=np.array([0, 1]), res=.001):
    ##
    # truncates the line defined by x, y into the frame X, Y
    ##
    try:
        p = np.linspace(0, 1, np.ceil(np.hypot(np.diff(x), np.diff(y)) / res)[0].astype(int))
        xy = np.array([x[0] + (x[1] - x[0]) * p, y[0] + (y[1] - y[0]) * p]).T
        inside, = np.where((xy[:, 0] >= min(X)) & (xy[:, 0] <= max(X)) & (xy[:, 1] >= min(Y)) & (xy[:, 1] <= max(Y)))
        return xy[inside[[0, -1]], 0], xy[inside[[0, -1]], 1]
    except:
        print(3)
    ##



def get_df_combined(files):
    sec_m_perm_eps_val = []
    sec_m_perm_eps_str = ['threshold sec', 'threshold m', 'max # permutations', 'clustering eps m']
    for file in files:
        sec_m_perm_eps_val.append( [ float(v) for i, v in enumerate( os.path.split( file)[1].split()) if i in [0,2,4,6]] )

    sec_m_perm_eps_val = np.array(sec_m_perm_eps_val)
    files_labels = ['']*len(files)
    out_path = ''
    for i, std in enumerate( sec_m_perm_eps_val.std(axis=0) ):
        if std == 0:
            out_path += '%s %g '%(sec_m_perm_eps_str[i], sec_m_perm_eps_val[0,i])
        else:
            out_path += '%s %s ' % (sec_m_perm_eps_str[i], str( np.unique( sec_m_perm_eps_val[:, i]) ) )
            for j, val in enumerate( sec_m_perm_eps_val[:, i] ):
                files_labels[j] += '%s=%.2g'%(sec_m_perm_eps_str[i], val )
    out_path = os.path.join( os.path.split( file)[0], 'tif', out_path[:-1] )
    df_combined = []
    for file_label, file in zip( files_labels, files ):
        ##
        df = pd.read_feather(file)
        df[['VesselType', 'Status']] = AIS_description.get_df(df=df).values
        ##
        df['file_label'] = file_label
        df_combined.append( df )
    df_combined = pd.concat( df_combined, axis = 0).sort_values(['timestamp', 'x', 'y', 'file_label', 'filename']).reset_index(drop = True)
    tmp = AIS_description.get_df(df=df, clean=True)
    df[tmp.columns] = tmp.values
    df_combined['interp distance sec'] = (df_combined['timestamp nearest'] - df_combined['timestamp']).dt.total_seconds()
    return df_combined, out_path
##

def get_out_path(files):
    sec_m_perm_eps_val = []
    sec_m_perm_eps_str = ['threshold sec', 'threshold m', 'max # permutations', 'clustering eps m']
    for file in files:
        sec_m_perm_eps_val.append( [ float(v) for i, v in enumerate( os.path.split( file)[1].split()) if i in [0,2,4,6]] )

    sec_m_perm_eps_val = np.array(sec_m_perm_eps_val)
    files_labels = ['']*len(files)
    out_path = ''
    for i, std in enumerate( sec_m_perm_eps_val.std(axis=0) ):
        if std == 0:
            out_path += '%s %g '%(sec_m_perm_eps_str[i], sec_m_perm_eps_val[0,i])
        else:
            out_path += '%s %s ' % (sec_m_perm_eps_str[i], str( np.unique( sec_m_perm_eps_val[:, i]) ) )
            for j, val in enumerate( sec_m_perm_eps_val[:, i] ):
                files_labels[j] += '%s=%.2g'%(sec_m_perm_eps_str[i], val )
    out_path = os.path.join( os.path.split( file)[0], 'tif', out_path[:-1] )
    return out_path, files_labels

def generate_figs_images(
    files = [None, None],
    overwrite = False ):
    global jpg_lookup
    if jpg_lookup is None:
        jpg_lookup = fun_sak.file_lookup(endswith='.jpg')
    out_path, file_labels = get_out_path(files)
    dict_df = dict()
    for file, file_label in zip( files, file_labels ):
        ##
        df = pd.read_feather(file).sort_values(['timestamp', 'filename', 'x', 'y'])
        df['interp distance sec'] = (df['timestamp nearest'] - df['timestamp']).dt.total_seconds()
        df = df.join( AIS_description.get_df(df = df, clean = True))
        df[['VesselType', 'Status', 'interp VesselName']] = df[['VesselType', 'Status', 'interp VesselName']].applymap( lambda s: re.sub(  '^$', 'NA', re.sub( '\(.*\)', '', s).strip() ) if isinstance(s, str) else 'NA' )
        ##
        dict_df[file_label] = df
    ##
    linestyles = ['solid', 'dashed', 'dashdot'] * 100
    ##
    for timestamp_, df_ in tqdm.tqdm( df.groupby('timestamp') ):
        index_ = df_.index.to_flat_index()
        for i, (filename__, df__) in enumerate( df_.groupby( 'filename') ):
            if len(jpg_lookup.get_all(filename__)) > 1: #this is duplicated jpg filename for different images
                continue
            out_file = os.path.join( out_path, timestamp_.strftime('%Y-%m-%d %H:%M:%S')+' (%i %i) %s.tif'%(  i+1, df_['filename'].drop_duplicates().shape[0], os.path.splitext(filename__)[0]) )
            if not overwrite and os.path.exists( out_file ):
                continue
            lat2pixel = lambda lat_: interpolate.interp1d( df__[['lat2', 'lat1']].mean().to_numpy(float), [0, im.size[1]-1], bounds_error = False, fill_value="extrapolate")(lat_)
            lon2pixel = lambda lon_: interpolate.interp1d( df__[['lon1', 'lon2']].mean().to_numpy(float), [0, im.size[0]-1], bounds_error = False, fill_value="extrapolate")(lon_)

            fig, ax = plt.subplots(1,1, figsize = [16, 7])
            im = PIL.Image.open( jpg_lookup.get(filename__) )
            ax.imshow(im)
            ax.set_xlim( [0, im.size[0]])
            ax.set_ylim([im.size[1], 0])

            ax.set_xticks([0, im.size[0]-1],
                          labels=['%f deg'%df__['lon1'].mean(),
                                  '%f deg\n%.1f km' %( df__['lon2'].mean(), df__['bottom width m'].mean()/1000) ], horizontalalignment='right',
                          verticalalignment = 'top' )
            ax.set_yticks( [im.size[1]-1, 0],
                          labels=['%f deg' %df__['lat2'].mean(),
                                  '%f deg\n%.1f km' %( df__['lat1'].mean(), df__['height m'].mean()/1000) ], horizontalalignment='right',
                           verticalalignment = 'bottom' )

            stat = ImageStat.Stat(im)
            r, g, b = stat.mean
            brightness = np.sqrt(0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))
            title = [ '{}: image {}/{}, contains {}/{} ships ({})'.format( timestamp_, i+1, df_['filename'].drop_duplicates().shape[0],  df__.shape[0], df_.shape[0], filename__) ]
            horizontalalignment = lambda x: 'right' if x < im.size[0]/2 else 'left'
            verticalalignment = lambda x: 'bottom' if x < im.size[1] / 2 else 'top'
            letter_dict = dict()
            number_dict = dict()
            for row, xy in df__[['x', 'y']].reset_index().iterrows():
                letter_dict[xy.x, xy.y] = chr(ord('a') + row)
                ax.text( xy.x, xy.y, letter_dict[xy.x, xy.y], color = 'w', horizontalalignment=horizontalalignment(xy.x), verticalalignment = verticalalignment(xy.y), fontweight = 'bold')
            index__ = df__.index.to_flat_index()
            for j, file_label in enumerate( dict_df):
                df___ = dict_df[file_label].loc[index__]
                title.append( r'$\bf{%s}$ lines for '%linestyles[j] + '{}: {}/{} pairings, mean distance {:.1f}/{:.1f} m'.format( file_label, sum(df___['interp mmsi'] >= 0), sum(dict_df[file_label].loc[index_,'interp mmsi'] >= 0),
                    df___['interp distance m'].mean(), dict_df[file_label].loc[index_, 'interp distance m'].mean() ) )
 #                if title == ['2018-03-21 18:00:00: image 9/9, contains 2/20 ships (29.36_-94.79_2018_03_21.jpg)',
 # '$\\bf{solid}$ lines: 1/16 pairings, mean distance 883.6/446.0 m']:
 #                    print(3)
                ##
                n = sum(df___['interp mmsi'] >= 0)
                if brightness > 50:
                    colors = plt.cm.jet(np.linspace(0, 1, n))  # This returns RGBA; convert:
                else:
                    colors = plt.cm.spring(np.linspace(0, 1, n))  # This returns RGBA; convert:

                for row, ser in df___.dropna( subset = ['interp longitude'] ).reset_index().iterrows():
                    x = lon2pixel(ser[['longitude', 'interp longitude']].to_list())
                    y = lat2pixel(ser[['latitude', 'interp latitude']].to_list())
                    x_, y_ = crop_line_into_frame( x, y, [0, im.size[0]-1], [0, im.size[1]-1] )
                    # if ser['interp VesselName'] == 'PAPA BOB':
                    #     print(3)
                    if not ser['interp mmsi'] in number_dict:
                        number_dict[ser['interp mmsi']] = len(number_dict)
                        ax.text( x_[1], y_[1], number_dict[ser['interp mmsi']], color = 'r', horizontalalignment=horizontalalignment(x_[1]), verticalalignment = verticalalignment(y_[1]), fontweight = 'bold', bbox = {'alpha' : .5, 'facecolor': 'w', 'edgecolor':'none'} )
                    elif any( (x < 0) | (x > im.size[0]) | (y < 0 ) | (y > im.size[1]) ):
                        ax.text(x_[1], y_[1], number_dict[ser['interp mmsi']], color='r',
                                horizontalalignment=horizontalalignment(x_[1]),
                                verticalalignment=verticalalignment(y_[1]), fontweight='bold',
                                bbox={'alpha': .5, 'facecolor': 'w', 'edgecolor': 'none'})

                    ##
                    label = r'$\bf{%s}-\bf{%i}$: '%(letter_dict[(ser.x, ser.y)], number_dict[ser['interp mmsi']])
                    label += '{:.0f}, {}, {}, {}, L={:.0f}m, ∆ = {:.0f}sec, {:.0f}m'.format( *ser[
                        ['interp mmsi', 'interp VesselName', 'VesselType', 'Status', 'interp Length',
                         'interp distance sec', 'interp distance m']])
                    label = re.sub( 'nanm', 'NA', label )
                    ##
                    ax.plot(x_.T, y_.T, linestyle=linestyles[j], color=colors[row], linewidth=2,
                            label=label, alpha=.6)
                    if np.isfinite(ser['interp Length']):
                        circle = plt.Circle( (ser.x, ser.y), radius = ser['interp Length']/ser['height m']*im.size[1]/2, linestyle=linestyles[j], color = 'w', fill = False, clip_on = True )
                        ax.add_patch(circle)

                ##
            bounds = list( ax.get_position().bounds )
            bounds[:2] = [.1, .1]
            if bounds[2] > .45:
                bounds[3] *= .45/bounds[2]
                bounds[2] = .45
            ax.set_position(bounds)
            # ax.set_title('\n'.join(title))
            ax.legend( loc=[1.02, 0], fontsize='small', facecolor='grey', framealpha=1,
                      title='\n'.join(title), handlelength=4)
            # plt.show()
            funsak.create_dir4file(out_file)
            fig.savefig(out_file, dpi = 100 )

if __name__ == '__main__':
    if False:
        ##
        df_stats = ship_pairing_analysis.get_df_stats()
        df_stats
        ##
    elif True:
        file_pairs = [
            [None, None],
            [None, None],
        ]
        for files in file_pairs:
            generate_figs_images( overwrite = False, files = files )
    elif True:
        ##
        from scipy import interpolate
        x, y = np.array([.5, 1.5]), np.array([-.1, .5])
        X, Y = np.array([0, 1]), np.array([0, 1])
        into_frame(np.array([.5, 1.5]), np.array([-.1, .5]), np.array([0, 1]), np.array([0, 1]) )
        ##
    else:
        ##
        im_file = None
        jpg_lookup = fun_sak.file_lookup(endswith='.jpg')
        full_im_file = jpg_lookup.get(None)
        df_truth = pd.read_feather( AIS_parameters.files['ship_truth_data'] )
        df_truth['datetime'] = df_truth['datetime'].dt.tz_convert('UTC').dt.tz_localize(None)
        ##
        radius = df_truth.loc[df_truth.filename == im_file, 'diameter'].max() / 2
        xlim = [df_truth.loc[df_truth.filename == im_file, 'x'].min() - radius, df_truth.loc[df_truth.filename == im_file, 'x'].max() + radius]
        ylim = [df_truth.loc[df_truth.filename == im_file, 'y'].min() - radius,
                df_truth.loc[df_truth.filename == im_file, 'y'].max() + radius]

        ##
        df_interp, df_source = AIS_interp.get_interp(time=df_truth.loc[df_truth.filename == im_file, 'datetime'].mean(), return_df=True,
                                                     aux_columns=['timestamp nearest'])
        df_interp, _ = import_AIS.get_df( df_truth.loc[df_truth.filename == im_file, 'datetime'].mean())
        threshold_sec = 1E3
        good, = np.where( (df_interp.timestamp - df_truth.loc[df_truth.filename == im_file, 'datetime'].mean()).dt.total_seconds().abs() < threshold_sec )
        threshold_m = 1E3
        tmp = fun_sak.distance_haversine( lat1 = df_interp.loc[good, 'latitude'], lon1 = df_interp.loc[good, 'longitude'], \
            lon2 = df_truth.loc[df_truth.filename == im_file, 'longitude'].mean(), lat2 = df_truth.loc[df_truth.filename == im_file, 'latitude'].mean(), convert2radians=True ) < threshold_m
        good = good[tmp]
        im = PIL.Image.open( jpg_lookup.get(None))
        ##
        plt.figure(figsize=[12,10])
        plt.imshow(im)
        ax = plt.gca()
        ax.set_xlim( xlim )
        ax.set_ylim(ylim[::-1])
        target_Length = 12
        (df_interp['interp Length'] - target_Length).abs().sort_values(ascending = True)
        # for index, ser in df_truth.loc[df_truth.filename == im_file].iterrows():
        #     cir = plt.Circle( ( ser.x, ser.y ), radius=ser.diameter / 2, color='w', fill=False)
        #     ax.add_patch(cir)

        pixel_per_m = im.size[1]/fun_sak.distance_haversine( lat1 = df_truth.loc[df_truth.filename == im_file, 'lat1'].mean(),
              lat2 = df_truth.loc[df_truth.filename == im_file, 'lat2'].mean(),
               lon1 = df_truth.loc[df_truth.filename == im_file, 'lon1'].mean(), lon2 = df_truth.loc[df_truth.filename == im_file, 'lon1'].mean(), convert2radians=True )
        Length_pixel = df_interp.loc[good[df_interp.loc[good, 'interp Length']>0], 'interp Length'].drop_duplicates().sort_values()*pixel_per_m
        X = [796, 848, 897.5]
        Y = [257, 251, 235]
        Length_pixel = np.ones(Length_pixel.shape[0])*target_Length*pixel_per_m
        for i, length in enumerate( Length_pixel ):
            # print( X[i%sum(df_truth.filename == im_file)], Y[i%sum(df_truth.filename == im_file)] )
            cir = plt.Circle( ( X[i%sum(df_truth.filename == im_file)], Y[i%sum(df_truth.filename == im_file)] ),
            radius=length / 2, color='r', fill=False )
            ax.add_patch(cir)
        df_interp.timestamp
        plt.show()
##

