"""Controller file for the behavioral classifier
"""
from distutils.command.install_egg_info import to_filename


from dark_ships.behavioral_classifier.transformation_pipeline import process_data, parse_sample, parse_sample_auto, parse_tfrecord, parse_sample_RF, parse_tfrecord_rf
from dark_ships.behavioral_classifier.models import build_conv2d_model, build_conv1d_model, build_auto_encoder, build_2D_CAE, build_dense_model, build_random_forest_model
from dark_ships.behavioral_classifier.utils import EvaluationCallback, lr_time_based_decay
import tensorflow as tf
from omegaconf import OmegaConf
import argparse
from copy import deepcopy
import time
import os
from typing import List

    
def split_data(files: List[str], conf, model_type: str, use_RFF: bool=False, index: int=0):
    
    dataset = tf.data.TFRecordDataset(files)
    if model_type in ['RF']:
        cols = os.path.join(conf['DATA']['OUTPUT']['SAVE_PATH'], 'cols.json')
        ds = dataset.map(lambda x: parse_tfrecord_rf(x, cols))
        ds = ds.map(lambda x: parse_sample_RF(x, cols, index))
    
    else:
        ds = dataset.map(parse_tfrecord)
    if model_type in ['CAE', 'AE']:
        ds = ds.map(parse_sample_auto)
    elif model_type in ['2D_Conv', '1D_Conv', 'dense']:
        ds = ds.map(parse_sample)
    
    ds = ds.batch(conf["MODEL"]['TRAIN']['batch_size'])
    return ds

def main(args):

    conf = OmegaConf.load(args.config)
    dataset = process_data(conf)
    
    if conf['DATA_ONLY']:
        return
    
    model_type = conf['MODEL']['MODEL_TYPE']
    use_RFF = conf['MODEL']['RFF']

    
    split = 0.8
    index=None
    if "index" in conf['MODEL']['TRAIN'].keys():
        index = conf['MODEL']['TRAIN']['index']
    
    train_ds = split_data(dataset[:int(len(dataset)*split)], conf, model_type, use_RFF, index)
    test_ds = split_data(dataset[int(len(dataset)*split):], conf, model_type, use_RFF, index)
    
    
    x = next(iter(train_ds))
    breakpoint()
    input_size = len(x[0])
    output_size = len(x[1])

    if model_type == 'CAE':
        model = build_2D_CAE(input_size, 3, 128, use_RFF)
    elif model_type == 'AE':
        model = build_auto_encoder(input_size, use_RFF)
    elif model_type == '2D_Conv':
        model = build_conv2d_model(input_size, 3, 128, output_size, use_RFF)
    elif model_type == '1D_Conv':
        model = build_conv1d_model(input_size, 3, 128, output_size, use_RFF)
    elif model_type == 'dense':
        model = build_dense_model(input_size, output_size, use_RFF)
    elif model_type == 'RF':
        model = build_random_forest_model(input_size)
#    elif model_type == 'OPTICS':
#        model = build_optics_model(input_size)
#    elif model_type == 'KMEANS':
#        model = build_kmeans_model(input_size)

    model_save = os.path.join(conf['MODEL']['SAVE_PATH'], 'checkpoint')
    if not os.path.exists(model_save):
        os.makedirs(model_save)
    
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #     filepath = model_save,
    #     monitor = 'val_accuracy',
    #     save_freq = 'epoch',
    # )
    
    
    # lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_time_based_decay, verbose=True)
    
    # eval_cb = EvaluationCallback(
    #     test_ds,
    #      conf['MODEL']['REPORT']
    # )
    breakpoint()
    history = model.fit(x=train_ds.as_numpy_iterator())

    # preds = model.predict(ds)
    breakpoint()

    print('test')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",'--config', help="Configuration file path")

    args = parser.parse_args()

    main(args)
