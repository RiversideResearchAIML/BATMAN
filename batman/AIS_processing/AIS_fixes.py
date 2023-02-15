import pandas as pd
import os
import AIS_parameters
import tqdm
import itertools
import numpy as np
from rapidfuzz import process, fuzz
from scipy import stats

path = os.path.join( AIS_parameters.dirs['root'], 'download')
##
columns = dict()
for file in tqdm.tqdm(os.listdir(path)):
    if file.endswith('.zip'):
        columns[file] = pd.read_csv( os.path.join( path, file), engine='c', low_memory=False, nrows=1 ).columns.to_list()
##
unique_columns, column_counts = np.unique(list(itertools.chain(*columns.values())), return_counts=True)
##
target_number_columns = stats.mode([ len(v) for v in columns.values() ]).mode[0]
##
column_counts_dict = dict( zip(unique_columns, column_counts))
metric = []
for uval in unique_columns:
    metric.extend([[uval, p[0], p[1]*column_counts_dict[p[0]]] for p in process.extract(uval, unique_columns, scorer=fuzz.ratio, limit=None)])
##
dict_rename = dict(zip(unique_columns, unique_columns))
for argsort in np.argsort( [-m[-1] for m in metric]):
    if len(np.unique(list(dict_rename.values()))) == target_number_columns:
        break
    m = metric[argsort]
    dict_rename.update( {m[0]:dict_rename[m[1]]})
    dict_rename = {k: dict_rename[v] for k, v in dict_rename.items()}
##
df_rename = pd.DataFrame( columns).T
##
set_ = set( dict_rename.values())
df_rename['ok'] = [set_.issubset(values) for values in columns.values()]
##
