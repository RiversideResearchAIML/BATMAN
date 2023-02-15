from unicodedata import category
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import pickle
import category_encoders as ce
import os
import sklearn
from typing import List, Dict
from copy import deepcopy
import tensorflow as tf
from tensorflow import keras
import math
import datetime


# from dark_ships.AIS_collection.classes import world_ports

def type_aware_fillna(df: pd.DataFrame):
    for col in df:
        #get dtype for column
        dt = df[col].dtype 
        #check that it is not categorical
        if dt == 'category':
            df[col] = df[col].astype(str)
        #check if it is a number
        if is_numeric_dtype(dt):
            df[col].fillna(-1, inplace=True)
        else:
            df[col].fillna("NoEntry", inplace=True)

    return df

def column_exists(input: pd.DataFrame, col: str):
    try:
        assert col in input.columns, "\nError: Column '{}' not found in dataframe".format(col)
    except AssertionError as msg:
        print(msg)

def decimate(input: pd.DataFrame, removals: List[str]):
    return input.drop(removals, axis=1)

def split_datetime(input: pd.DataFrame, col_names: List[str]) -> pd.DataFrame:

    for col_name in col_names:
        new_cols = [col_name+'_year', col_name+'_month', col_name+'_day', 
        col_name+'_hour', col_name+'_minute', col_name+'_second']
        input_date = pd.to_datetime(input[col_name]).dt.date
        input_time = pd.to_datetime(input[col_name]).dt.time
        input[new_cols[0]] = input_date.map(lambda x: x.year)/3000
        input[new_cols[1]] = input_date.map(lambda x: x.month)/12
        input[new_cols[2]] = input_date.map(lambda x: x.day)/31
        input[new_cols[3]] = input_time.map(lambda x: x.hour)/24
        input[new_cols[4]] = input_time.map(lambda x: x.minute)/60
        input[new_cols[5]] = input_time.map(lambda x: x.second)/60
        input = input.drop(col_name, axis=1)

    return input

def create_specials(ais: pd.DataFrame, specials: Dict):
    for special, val in specials.items():
        ais[val[0]] = ais[special] == val[1]
    return ais

def is_outside(ais: pd.DataFrame, specials: Dict):
    for special, val in specials.items():
        # conditions = [ ais[special] < val[1] or ais[special] > val[2]]
        # choices = [True, False]
        ais[val[0]] = [True if (x < val[1] or x > val[2]) else False for x in ais[special]]
        ais = clip(ais, {special: val[1:]})
        ais[special] = normalize(ais[special], val[1:])
    return ais

def is_any(ais: pd.DataFrame, specials: Dict):
    output = {}
    for special, val in specials.items():
        output[special] = (ais[special] == val).any()
    
    return output

def is_not(ais: pd.DataFrame, specials: Dict):
    output = {}
    for special, val in specials.items():
        output[special] = (ais[special] != val).any()
    
    return output

def is_greater(ais: pd.DataFrame, specials: Dict):
    output = {}
    for special, val in specials.items():
        output[special] = (ais[special] > val).any()
    
    return output

def is_less(ais: pd.DataFrame, specials: Dict):
    output = {}
    for special, val in specials.items():
        output[special] = (ais[special] < val).any()
    
    return output

def clip(ais: pd.DataFrame, cols: Dict):
    for col, val in cols.items():
        if val[0] != 'NA':
            ais[col] = [val[0] if x < val[0] else x for x in ais[col]]
        if val[1] != 'NA':
            ais[col] = [val[1] if x > val[1] else x for x in ais[col]]
    
    return ais


def normalize_df(dataframe: pd.DataFrame, d: Dict):
    for col in d.keys():
        column_exists(dataframe, col)
        dataframe[col] = normalize(dataframe[col], d[col])

    return dataframe

def normalize(series:pd.Series, range: List[float]) -> pd.Series:

    """
    Normalizes values to a range between -1 and +1 between the specified range
    """
    series = series.astype(float)
    mean = (max(range)+min(range))/2
    series = series.map(lambda x: (x-mean)/((max(range)-min(range))/2))
    return series

def contains_value(df: pd.DataFrame, cols: List[str]):
    breakpoint()
    pass


class world_data():

    def __init__(self, ports_file: str):
        self.ports = world_ports(path4csv=ports_file)
    
    def closest_port(self, lat, lon):
        return self.ports.merge(self.ports.closest(pd.DataFrame(np.array([[lat, lon]]), columns=['latitude', 'longitude'])))


class ais_encoders():
    def __init__(self, encode_dict: dict,  ais: pd.DataFrame):
        self.encode_dict = encode_dict
        if not os.path.exists(self.encode_dict['ENCODER_PATH']):
            os.makedirs(self.encode_dict['ENCODER_PATH'])
        self.one_hot_encoders = {}
        self.base2_encoders = {}
        self.one_hot_create(encode_dict['ONE_HOT']['CREATE'], ais)
        self.base2_create(encode_dict['BASE2']['CREATE'], ais)
        self.load_encoders()


    def one_hot_create(self, d: dict, dataframe: pd.DataFrame):
        if d == None:
            return
        for save, label in d.items():
            column_exists(dataframe, label)
            category = dataframe[[label]]
            one_hot_encoder = sklearn.preprocessing.OneHotEncoder(sparse=False)
            encoder = one_hot_encoder.fit(category)
            outpath = os.path.join(self.encode_dict['ENCODER_PATH'], save)
            with open(outpath, 'wb') as f:
                pickle.dump(encoder, f)
            self.one_hot_encoders.update({label: encoder})

    def base2_create(self, d: dict, dataframe: pd.DataFrame):
        if d == None:
            return
        for save, label in d.items():
            column_exists(dataframe, label)
            category = dataframe[[label]].astype(str)
            encoder = ce.BaseNEncoder(base=2)
            encoder = encoder.fit(category)
            outpath = os.path.join(self.encode_dict['ENCODER_PATH'], save)
            with open(outpath, 'wb') as f:
                pickle.dump(encoder, f)
            self.base2_encoders.update({label: encoder})
    
    def load_encoders(self):
        if self.encode_dict['ONE_HOT']['LOAD'] is not None:
            for save, label in self.encode_dict['ONE_HOT']['LOAD'].items():
                pth = os.path.join(self.encode_dict['ENCODER_PATH'], save)
                with open(pth, 'rb') as f:
                    encoder = pickle.load(f)
                self.one_hot_encoders.update({label: encoder})

        if self.encode_dict['BASE2']['LOAD'] is not None:
            for save, label in self.encode_dict['BASE2']['LOAD'].items():
                pth = os.path.join(self.encode_dict['ENCODER_PATH'], save)
                with open(pth, 'rb') as f:
                    encoder = pickle.load(f)
                self.base2_encoders.update({label: encoder})
        
        
    def encode(self, ais):
        removals = []
        for col, encoder in sorted(self.base2_encoders.items()):
            names = encoder.get_feature_names()
            removals.append(col)
            category = ais[[col]].astype(str)
            df = pd.DataFrame(encoder.transform(category), columns=names)
            ais = pd.merge(ais, df, left_index=True, right_index=True)

        for col, encoder in sorted(self.one_hot_encoders.items()):
            names = encoder.get_feature_names([col])
            removals.append(col)
            category = ais[[col]]
            df = pd.DataFrame(encoder.transform(category), columns=names)
            ais = pd.merge(ais, df, left_index=True, right_index=True)

        return decimate(ais, removals)
    
class EvaluationCallback(keras.callbacks.Callback):
    
    def __init__(self, ds, report_save):
        super(EvaluationCallback, self).__init__()
        self.ds = ds
        self.report = report_save
        self.label_names = ['anamolous routing 1E+04 m away from coast for cargo ships at sea, where > 50% waypoints are un-traveled', 'mmsi of transshipment within 20 m 1E+04 m away from coast', 'successive loitering < 1E+04 m near coast', 'successive loitering > 1E+04 m away from coast', 'successive loitering at sea and fishing vessel and Fishing Prohibited', 'successive loitering at sea and fishing vessel and Fishing Prohibited and dark', 'successive loitering at sea and fishing vessel dark', 'successive loitering at sea within 5E+03 m', 'successive loitering within 10E3 m of port', 'successive loitering within 2E3 m of pipelines by dredging or diving vessel']
        self.labels = []
        for x in ds.__iter__():
            for y in x[1]:
                self.labels.append(y)
                
        self.report = report(self.label_names, self.report)

            
        


    def on_epoch_end(self, epoch, logs=None):
        
        preds = self.model.predict(self.ds)
        p = (preds>0.5).astype(int)
        self.report.compute(self.labels, p, epoch)
        # report = sklearn.metrics.multilabel_confusion_matrix(self.labels, p, samplewise=True)
        print('\n')
        print(self.report)
        print('\n')
        self.report.reset()
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))


def lr_time_based_decay(epoch, lr):
    return lr * math.pow(.99, epoch)


class report():
    
    
    
    def __init__(self, class_names, report_save):

        self.class_names = class_names
        self.num_labels = len(class_names)
        self.count_classes = True
        self.save_location = report_save.format(datetime.datetime.now())
        self.reset()
        
        
        
    def reset(self):
        self.cm = np.zeros((self.num_labels,2,2,))
        self.p = np.zeros((self.num_labels,))
        self.r = np.zeros((self.num_labels,))
        self.f = np.zeros((self.num_labels,))
        
    def compute(self, labels, preds, epoch):
        if self.count_classes:
            self.class_counts = sum(labels)
            
        for label, pred in zip(labels, preds):
            self.update_cm(label, pred)
        
        with open(self.save_location, 'a+') as f:
            f.write('epoch number: {}\n'.format(epoch))
            f.write(self.__str__())
            f.write('\n')
                
                
            
        for ind in range(self.num_labels):
            self.p[ind], self.r[ind], self.f[ind] = self.precision_recall(self.cm[ind])
    
    
    def update_cm(self, label, pred):
        for idx, pair in enumerate(zip(label,pred)):
            self.cm[idx, self.flip(pair[1]), self.flip(pair[0]), ] +=1
        
    
    def __str__(self):
        s = ''
        s += '\tprecision\trecall\tf1_score'
        for ind, x in enumerate(self.class_names):
            s += "\nclass {}\t{:.2f}\t\t{:.2f}\t{:.2f}".format(ind, self.p[ind], self.r[ind], self.f[ind])
        s += '\n' + str(self.cm)
        return s
    
    def precision_recall(self, cm):
        precision = cm[0][0]/(cm[0][0]+cm[0][1])
        recall = cm[0][0]/(cm[0][0]+cm[1][0])
        f = 2*(precision*recall)/(precision+recall)
        return precision, recall, f
    
    
    @staticmethod
    def flip(x):
        if x==1:
            return 0
        return 1