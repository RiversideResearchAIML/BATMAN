from dark_ships.behavioral_classifier.transformation_pipeline import process_data, parse_sample, parse_sample_auto, parse_tfrecord
import tensorflow as tf
from glob import glob
from tqdm import tqdm


directory = ''

def count_empty(ds):
    labels = []
    count = 0
    for x in tqdm(ds.__iter__(), desc='total items counted {}'.format(count)):
        if all(x[1]==0):
            count+=1
        labels.append(x[1])
    
    # class_counts = sum(labels)
    print("Final count: {}".format(count))

    
if __name__ == "__main__":
    files = glob(directory + "/*.tfrecord")
    
    dataset = tf.data.TFRecordDataset(files)
    ds = dataset.map(parse_tfrecord)
    ds = ds.map(parse_sample)
    
    count_empty(ds)
