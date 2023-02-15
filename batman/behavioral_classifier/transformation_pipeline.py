from math import sqrt
from zlib import crc32
import tarfile
import urllib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import sklearn.preprocessing
import sklearn.base
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from typing import List, Dict
import pickle
import tensorflow as tf
from glob import glob
from pdb import set_trace
from tqdm import tqdm
from pandas_tfrecords import pd2tf
import json
import re


# todo make this import work
# from dark_ships.AIS_collection.classes import world_ports
import dark_ships.data_utils.api_utils as apis
from dark_ships.behavioral_classifier.utils import ais_encoders, world_data
import dark_ships.behavioral_classifier.utils as utils

download = " "
path = " "
url = " "

# class world_data():

#     def __init__(self, ports_file: str):
#         self.ports = world_ports(path4csv=ports_file)
    
#     def closest_port(self, lat, lon):
#         return self.ports.merge(self.ports.closest(pd.DataFrame(np.array([[lat, lon]]), columns=['latitude', 'longitude'])))


def get_data(url: str, path: str) -> pd.DataFrame:
    os.makedirs(path, exist_ok=True)
    tgz_path = os.path.join(path, ".tgz")
    urllib.request.urlretrieve(url, tgz_path)
    new_tgz = tarfile.open(tgz_path)
    new_tgz.extractall(path=path)
    new_tgz.close()
    csv_path = os.path.join(path, ".csv")
    return pd.read_csv(csv_path)

def split_train_test(dataframe: pd.DataFrame, test_ratio: float) -> pd.DataFrame:
    shuffled_indices = np.random.permutation(len(dataframe))
    test_set_size = int(len(dataframe) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return dataframe.iloc[train_indices], dataframe.iloc[test_indices]

def test_set_check(identifier, test_ratio: float):
    return crc32(np.float64(identifier)) & 0xffffffff < (test_ratio * (2^32))

def split_train_test_by_id(dataframe: pd.DataFrame, test_ratio: float, id_column):
    ids = dataframe[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return dataframe.loc[-in_test_set], dataframe.loc[in_test_set]

def stratified_sampling():
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    stratified_dataframe = get_data(url, path)
    for train_index, test_index in split.split(stratified_dataframe, stratified_dataframe[" "]):
        strat_train_set = stratified_dataframe.loc[train_index]
        strat_test_set = stratified_dataframe.loc[test_index]
    return strat_train_set, strat_test_set

def ordinal(dataframe: pd.DataFrame, label: str):
    category = dataframe[str]
    ordinal_encoder = sklearn.preprocessing.OrdinalEncoder()
    dataframe_encoded = ordinal_encoder.fit_transform(category)
    return dataframe_encoded[0:]


class CombinedAttributes(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    A custom transformer class to combine specific attributes

    Utilizes scikit-learn base classes BaseEstimator and TransformerMixin
    ...

    Attributes
    ----------

    addition : bool
        hyperparameter default set to True

    Methods
    -------
    fit:
        Returns self

    transform:
        Returns X

    """

    def __init__(self, addition=True):
        self.addition = addition

    def fit(self, X, y=None):
        return self

    def transform(self, X):  # will change depending on relevant feature selection
        return X

def determinant(matrix):
    matrix_determinant = ((matrix[0,0] * matrix[1,1]) - (matrix[0,1] * matrix[1,0]))
    return matrix_determinant

def invertible(matrix):
    num_rows, num_cols = matrix.shape
    if (sqrt(np.size(matrix))) != num_rows & num_cols:
        return False
    else:
        if determinant(matrix) != 0:
            return True
        else:
            return False

def mod_inverse(n: int, m: int):
    for x in range(m+1):
        if ((n*x) % m == 1):
            return x
        else:
            Exception("There is no modular multiplicative inverse.")

# prepared = full_pipeline.fit_transform(get_data(url, path))
def get_data(source: str, type: str):
    if type == "Pandas":
        # if GPU > 0:
        #     return cudf.read_feather(source)
        return pd.read_feather(source)

def write_tfrecords(sample: pd.DataFrame, outpath: str, num_samples: int, labels, first_iter: bool, is_classic=True):
    fn = str(str(int(sample['AIS_downsample_timestamp_year'].iloc[0]*3000)) + str(int(sample['AIS_downsample_timestamp_month'].iloc[0]*12)).zfill(2) + str(int(sample['AIS_downsample_timestamp_day'].iloc[0]*365)).zfill(2) + str(int(sample['AIS_downsample_timestamp_hour'].iloc[0]*24)).zfill(2) + str(int(sample['AIS_downsample_timestamp_minute'].iloc[0]*60)).zfill(2) + str(sample['AIS_downsample_mmsi'].iloc[0]) + str(int(sample['AIS_downsample_timestamp_second'].iloc[0]*60)))
    
    
    record_outpath = os.path.join(outpath, fn + '.tfrecord')
    # target = sample[target_cols].iloc[0].to_numpy().astype('float32').reshape(-1)
    # sample = sample.drop(target_cols, axis=1)
    sample = sample.drop(['AIS_downsample_mmsi'], axis=1)
    pd.set_option('display.max_rows', None)
    sample_numpy = sample.to_numpy().astype('float32').reshape(-1)
    if max(sample_numpy > 2) or min(sample_numpy < -2):
        print('pause')
    if len(sample_numpy) < len(sample.columns)*num_samples:
        vec = np.zeros(len(sample.columns)*num_samples, dtype=float)
        vec[:len(sample_numpy)] = sample_numpy
        sample_numpy = vec.astype('float32')
        
    if is_classic:
        label_cols = []
        for val in range(num_samples):
            label_cols.extend([x + '_' + str(val) for x in sample.columns])
        if first_iter:
            json_out = os.path.join(outpath, 'cols.json')
            with open(json_out, 'w') as fp:
                remove = ' \t?%,'
                label_cols = [re.sub(remove, '_', x) for x in label_cols]
                json.dump(label_cols, fp)
            
        
        sample = tf.io.serialize_tensor(sample_numpy).numpy()
        target = tf.io.serialize_tensor(labels).numpy()
        sample_dict={x: tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(y).numpy()])) for x, y in zip(label_cols, sample_numpy)}
        sample_dict["target"] =  tf.train.Feature(bytes_list=tf.train.BytesList(value=[target]))
    
    
    else:
        sample = tf.io.serialize_tensor(sample_numpy).numpy()
        target = tf.io.serialize_tensor(labels).numpy()
        sample_dict = {
                        "example":  tf.train.Feature(bytes_list=tf.train.BytesList(value=[sample])),
                        "target": tf.train.Feature(bytes_list=tf.train.BytesList(value=[target]))
                    }
    sample = tf.train.Example(features=tf.train.Features(feature=sample_dict)).SerializeToString()
    with tf.io.TFRecordWriter(record_outpath) as writer:
        writer.write(sample)
    
def gather_dataset(directories: List[str]):
    files = []
    for directory in directories:
        files.extend(glob(directory + "/*.tfrecord"))
    
    return files

def parse_tfrecord(example):
    feature_description = {
            "example": tf.io.FixedLenFeature([], tf.string),
            'target': tf.io.FixedLenFeature([], tf.string)
        }
    return tf.io.parse_single_example(example, feature_description, )

def parse_tfrecord_rf(example, cols_fp):
    with open(cols_fp, 'r') as fp:
        cols = json.load(fp)
    feature_description = {
            x: tf.io.FixedLenFeature([], tf.string) for x in cols
        }
    feature_description["target"] = tf.io.FixedLenFeature([], tf.string)
    return tf.io.parse_single_example(example, feature_description, )


def parse_sample_RF(sample, cols_fp, ind):
    with open(cols_fp, 'r') as fp:
        cols = json.load(fp)
    output = {     
        x: tf.io.parse_tensor(sample[x], out_type=tf.float32) for x in cols
    }
    label = tf.io.parse_tensor(sample['target'], out_type=tf.int64)
    return output, label[ind]

def parse_sample(sample):
    output = {
        'example': tf.io.parse_tensor(sample['example'], out_type=tf.float32),
        "target": tf.io.parse_tensor(sample['target'], out_type=tf.int64),
    }
    return output['example'], output['target']

def parse_sample_auto(sample):
    output = {
        'example': tf.io.parse_tensor(sample['example'], out_type=tf.float32),
        "target": tf.io.parse_tensor(sample['example'], out_type=tf.int64),
    }
    return output['example'], output['example']

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a floast_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
  array = tf.io.serialize_tensor(array).numpy()
  return array

def process_df(df: pd.DataFrame, DATA_CONF: Dict, encoders: ais_encoders = None):
    # Data preprocessing and management
    ais = utils.create_specials(df, DATA_CONF['SPECIAL_VALUES'])
    ais = utils.is_outside(ais, DATA_CONF['IS_OUTSIDE'])
    ais = utils.clip(ais, DATA_CONF['CLIP'])
    ais.rename(columns=DATA_CONF['MAPPING'], inplace=True)
    ais = utils.type_aware_fillna(ais)
    ais = utils.split_datetime(ais,  DATA_CONF['SPLIT_TIME'])
    ais = utils.normalize_df(ais, DATA_CONF['NORMALIZE'])
    
    if encoders is not None:
        ais = encoders.encode(ais)
    return ais

def process_labels(df: pd.DataFrame, DATA_CONF: Dict):
    labels = {}
    labels.update(utils.is_greater(df, DATA_CONF['IS_GREATER']))
    labels.update(utils.is_less(df, DATA_CONF['IS_LESS']))
    labels.update(utils.is_any(df, DATA_CONF['IS_ANY']))
    labels.update(utils.is_not(df, DATA_CONF['IS_NOT']))
    
    sorted_keys = sorted(labels.keys())
    
    labels = np.array([labels[key] for key in sorted_keys]).astype(int)
    
    return labels


def process_data(conf):
    
    # Get data, this will run different things based on what data you have
    

    if conf['DATA']['LOAD_FROM_OUTPUT'] == False or conf['DATA_ONLY'] == True:

        RESAMPLED_DATA_CONF = conf['DATA']['SAMPLE']
        LEGALITY_DATA_CONF = conf['DATA']['LEGALITY']
        labels_conf = conf['DATA']['LABELS']
        data_fps = RESAMPLED_DATA_CONF['PATHS']
        label_fps = labels_conf['PATHS']
        legality_fps = LEGALITY_DATA_CONF['PATHS']
        
        file_num = 0
        # Create save path
        output = conf['DATA']['OUTPUT']['SAVE_PATH']

        if not os.path.exists(output):
            os.makedirs(output)
            
        print('\n\nSaving sample data at "{}"\n'.format(output))
        first_iter=True
        for idx, (resampled_data, labels_data) in enumerate(zip(data_fps, label_fps)):
           
            # Transofrm resampled data
            resampled = get_data(resampled_data, 'Pandas')
            legality = get_data(legality_fps[idx], 'Pandas')
            legality.rename(columns={x: x.replace('\n', '_') for x in legality.columns}, inplace=True)
            legality.rename(columns={x: x.replace(' ', '_') for x in legality.columns}, inplace=True)
            resampled.rename(columns={x: x.replace('\n', '_') for x in resampled.columns}, inplace=True)
            resampled.rename(columns={x: x.replace(' ', '_') for x in resampled.columns}, inplace=True)
            legality = utils.decimate(legality, LEGALITY_DATA_CONF['DECIMATE'])
            resampled = utils.decimate(resampled,  RESAMPLED_DATA_CONF['DECIMATE'])
            mmsis = resampled['AIS_downsample_mmsi'].unique()
            encoders = ais_encoders(RESAMPLED_DATA_CONF['ENCODERS'], resampled)

            labels_df = get_data(labels_data, 'Pandas')
            labels_df.rename(columns={x: x.replace(' ', '_') for x in labels_df.columns}, inplace=True)
            resampled = resampled.join(legality)

            
            for mmsi in tqdm(mmsis[5:100], desc='sample generation progress for file "{}"'.format(data_fps[file_num])):
                example = resampled[resampled['AIS_downsample_mmsi'] == mmsi][:10]
                example = process_df(example, RESAMPLED_DATA_CONF, encoders)
                example = process_df(example, LEGALITY_DATA_CONF)
                idxs = example.index
                labels = process_labels(labels_df.iloc[idxs],labels_conf)
                try:
                    assert not example.isnull().values.any(), "\nError: nan  found in dataframe: skipping sample"
                    if len(example['AIS_downsample_mmsi']) > 1 and example['AIS_downsample_mmsi'].any() != 0:
                        write_tfrecords(example, output, 10, labels, first_iter)
                    first_iter=False
                except AssertionError as msg:
                    breakpoint()
                    print(msg)
            file_num += 1
            

    DATA_CONF = conf['DATA']['OUTPUT']
    return gather_dataset(DATA_CONF['LOAD_PATHS'])



if  __name__ == "__main__":

    pass
