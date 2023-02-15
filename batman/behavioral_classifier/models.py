from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Reshape, Input, Dropout, UpSampling2D, MaxPooling2D, Conv1D
from keras.losses import mse
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import tensorflow as tf
import keras
import tensorflow_decision_forests as tfdf
from sklearn.cluster import OPTICS, cluster_optics_dbscan

RFF_layer = tf.keras.layers.experimental.RandomFourierFeatures
loss_tracker = keras.metrics.Mean(name="loss")
mae_metric = keras.metrics.MeanAbsoluteError(name="mae")
# sig_cross_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=None, logits=None, name=None)



# general params
INPUTS = 40
OUTPUTS = 10

# CNN params
KERNEL_SIZE = 3
IMAGE_SIZE = 28


class custom_loss(tf.keras.losses.Loss):
    
    def __init__(self, outputs=OUTPUTS):
        super().__init__()
        self.outputs = outputs
    
    def call(self, y_true, y_logit):
        '''
        Multi-label cross-entropy
        * Required "Wp", "Wn" as positive & negative class-weights
        y_true: true value
        y_logit: predicted value
        '''
        loss = float(0)

        y_true = tf.cast(y_true, tf.float32)
        for i in range(self.outputs):
            first_term = y_true[i] * K.log(y_logit[i] + K.epsilon())
            second_term = (1 - y_true[i]) * K.log(1 - y_logit[i] + K.epsilon())
            loss -= (first_term + second_term)
        return loss

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def build_2D_CAE(inputs=INPUTS, kernel_size=KERNEL_SIZE, image_size=IMAGE_SIZE, RFF=0):

    #create model
    model = Sequential()
    model.add(Input(shape=(inputs,)))
    if RFF > 0:
        model.add(RFF_layer(inputs*RFF))
    model.add(Dense(image_size*image_size, activation='relu'))
    model.add(Dropout(.2))
    model.add(Reshape((image_size,image_size,1)))

    # Encoder
    model.add(Conv2D(32, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(MaxPooling2D(2, strides=2))
    model.add(Conv2D(64, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(MaxPooling2D(2, strides=2))
    model.add(Conv2D(128, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(MaxPooling2D(2, strides=2))
    

    # Decoder
    model.add(Conv2D(128, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(UpSampling2D(2))
    model.add(Conv2D(16, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(UpSampling2D(2))
    model.add(Conv2D(1, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(UpSampling2D(2))
    model.add(Flatten())
    model.add(Dense(inputs, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss="mean_squared_error")
    model.summary()

    return model

def build_conv2d_model(inputs=INPUTS, kernel_size=KERNEL_SIZE, image_size=IMAGE_SIZE, outputs=OUTPUTS, RFF=0):

    
    # model = Sequential()
    # model.add(Input(shape=(inputs,)))
    # if RFF > 0:
    #     model.add(RFF_layer(inputs*RFF))
    
#     model.add(Dense(image_size*image_size, activation='relu'))
#     model.add(Dropout(.2))
#     model.add(Reshape((image_size,image_size,1)))
#     model.add(Conv2D(64, kernel_size=kernel_size, padding='same', activation='relu'))
#     model.add(MaxPooling2D(2, strides=2))
#     model.add(Conv2D(32, kernel_size=kernel_size, padding='same', activation='relu'))
#     model.add(MaxPooling2D(2, strides=2))
#     model.add(Flatten())
#     # model.add(Dense(image_size*image_size, activation='relu'))
#     model.add(Dense(int(inputs*.5), activation='relu'))
#     model.add(Dense(outputs, activation='sigmoid'))

#     model.compile(optimizer=Adam(), metrics = [tf.keras.metrics.Accuracy()], loss="categorical_crossentropy")
#     model.summary()
    #create model
    model = Sequential()
    model.add(Input(shape=(inputs,)))
    if RFF > 0:
        model.add(RFF_layer(inputs*RFF))
    
    model.add(Dense(image_size*image_size, activation='relu'))
    model.add(Dropout(.2))
    model.add(Reshape((image_size,image_size,1)))
    model.add(Conv2D(16, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(MaxPooling2D(2, strides=2))
    model.add(Conv2D(16, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(MaxPooling2D(2, strides=2))
    model.add(Conv2D(16, kernel_size=kernel_size, padding='same', activation='relu'))
    model.add(MaxPooling2D(2, strides=2))
    model.add(Flatten())
    model.add(Dense(image_size, activation='relu'))
    model.add(Dense(outputs, activation='sigmoid', name='output'))

    # model.compile(optimizer=Adam(), metrics = [tf.keras.metrics.Accuracy()], loss="categorical_crossentropy")
    l = custom_loss(outputs)
    model.compile(optimizer=Adam(), metrics = [tf.keras.metrics.Accuracy()], loss=l)
    model.summary()

    return model

def build_dense_model(inputs=INPUTS, outputs=OUTPUTS, RFF=0):
    dim0 = int(inputs)
    dim1 = int(inputs*.8)
    dim2 = int(inputs*.5)
    
    model = Sequential()
    model.add(Input(shape=(inputs,)))
    if RFF > 0:
        model.add(RFF_layer(inputs*RFF))
        
    model.add(Dense(dim0, activation='relu'))
    model.add(Dense(dim1, activation='relu'))
    model.add(Dense(dim2, activation='relu'))
    
    model.add(Dense(outputs, activation='sigmoid'))
    l = custom_loss(outputs)

    model.compile(optimizer=Adam(), metrics = [tf.keras.metrics.Accuracy()], loss=l)
    model.summary()

    return model
    

def build_conv1d_model(inputs=INPUTS, kernel_size=KERNEL_SIZE, image_size=IMAGE_SIZE, outputs=OUTPUTS, RFF=0):

    #create model
    model = Sequential()
    model.add(Input(shape=(inputs,)))
    if RFF > 0:
        model.add(RFF_layer(inputs*RFF))
    
    model.add(Dense(image_size*image_size, activation='relu'))
    model.add(Dropout(.2))
    model.add(Reshape((image_size,image_size,1)))
    model.add(Conv1D(64, kernel_size=kernel_size, activation='relu'))
    model.add(Conv1D(32, kernel_size=kernel_size, activation='relu'))
    model.add(Flatten())
    model.add(Dense(outputs, activation='sigmoid'))

    model.compile(optimizer=Adam(), loss="mean_squared_error")
    model.summary()

    return model



def build_auto_encoder(inputs=INPUTS, RFF=0):
    dim0 = int(inputs)
    dim1 = int(inputs*.8)
    dim2 = int(inputs*.5)

    model = Sequential()
    model.add(Input(shape=(inputs,)))
    if RFF > 0:
        model.add(RFF_layer(inputs*RFF))

    model.add(Dense(dim0, activation='relu'))
    model.add(Dense(dim1, activation='relu'))
    model.add(Dense(dim2, activation='relu'))
    model.add(Dense(dim1, activation='relu'))
    model.add(Dense(inputs, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss="mean_squared_error")
    print(model.summary())

    return model

def build_random_forest_model(inputs=INPUTS, outputs=OUTPUTS, RFF=0):
    tuner = tfdf.tuner.RandomSearch(num_trials=20)
    model = tfdf.keras.RandomForestModel(
        verbose=2,
        num_trees=100,
        tuner=tuner,
        split_axis="SPARSE_OBLIQUE",
        random_seed=1000)
    model.compile(metrics=[tf.keras.metrics.Accuracy(),
                           tf.keras.metrics.Precision(), 
                           tf.keras.metrics.Recall()])
    model.build(input_shape=[inputs])
    print(model.summary())

    return model

def build_gradient_boosted_trees_model(inputs=INPUTS, outputs=OUTPUTS, RFF=0):
    tuner = tfdf.tuner.RandomSearch(num_trials=20)
    model = tfdf.keras.GradientBoostedTreesModel(
        verbose=2,
        num_trees=149,
        tuner=tuner,
        split_axis="SPARSE_OBLIQUE",
        random_seed=1000)
    model.compile(metrics=[tf.keras.metrics.Accuracy(),
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall()])
    model.build(input_shape=[inputs])
    print(model.summary())
    
    return model

def build_cart_model(inputs=INPUTS, outputs=OUTPUTS, RFF=0):
    model = tfdf.keras.CartModel(verbose=2)
    model.compile(metrics=[tf.keras.metrics.Accuracy(), 
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall()])
    model.build(input_shape=[inputs])
    print(model.summary())
    
    return model
