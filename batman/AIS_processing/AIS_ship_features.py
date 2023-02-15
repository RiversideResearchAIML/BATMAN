import pandas as pd
import AIS_parameters
import AIS_import
import tqdm
import numpy as np
import AIS_description
import matplotlib.pyplot as plt
import re
import os
import inspect
import seaborn as sns
import functions_basic_sak as funsak

fname = os.path.split( inspect.getfile(inspect.currentframe()) )[1]

dates = ['2022-02-14']
df = pd.concat([AIS_description.get_df(date = date_, clean=True, join = True, multicolumn = True) for date_ in tqdm.tqdm(dates)], axis=0).drop_duplicates('mmsi')
##
df['AspectRatio'] = df.Width/df.Length
df['type'] = df[('AIS_description', 'VesselType')].apply( lambda s: re.sub( '\s*\,.*', '', re.sub('\(.*\)', '', s) ).strip() if isinstance( s, str) and s[-1] == ')' and '(' in s[2:] else np.NaN )
threshold = df.type.dropna().shape[0]*.01
VC = df.type.sort_values().value_counts()
##
width = 2
fig, ax = plt.subplots( ((VC>threshold).sum()+width-1)//width, width, figsize = [10, 10])
ax = ax.flatten()
choice = ('Length', 'Length (m)', np.logspace(0, 3, 31) )
choice = ('AspectRatio', 'Width/Length', np.logspace(-3, 3, 31) )
count = 0
for type, df_ in df.groupby('type'):
    if VC[type]> threshold:
        print(type)
        df_[choice[0]].plot.hist(bins = choice[-1], loglog = True, ax = ax[count] )
        ax[count].set_title(' '+type, x = 0, y = .9, fontsize = 'small', verticalalignment = 'top', horizontalalignment = 'left')
        count += 1
ax[-2].set_xlabel( choice[-2] )
ax[-1].set_xlabel( choice[-2] )
plt.show()
out_file = os.path.join( AIS_parameters.dirs['root'], os.path.splitext(fname)[0], '%s by type.tif'%choice[0])
funsak.create_dir4file(out_file)
fig.savefig(out_file, dpi = 300)
##
sns.histplot( x = df.Length, y = df.Width, bins = 20, pthresh = .01)
sns.scatterplot(x=df.Length, y=df.Width, s = 5, color = ".15", alpha = .2 )
plt.show()
##


