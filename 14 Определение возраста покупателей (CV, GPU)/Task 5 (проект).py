import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_train(path):

    labels = pd.read_csv(path+'/labels.csv')
    
    datagen = ImageDataGenerator(
                validation_split=0.25, 
                horizontal_flip=True,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                rescale=1. / 255
    )
    train_datagen_flow = datagen.flow_from_dataframe(
        labels,
        path+'/final_files',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        seed=12345,
        subset='training')

    return train_datagen_flow
    
def load_test(path):

    labels = pd.read_csv(path+'/labels.csv')
    
    datagen = ImageDataGenerator(
                validation_split=0.25, 
                rescale=1. / 255
    )
    train_datagen_flow = datagen.flow_from_dataframe(
        labels,
        path+'/final_files',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        seed=12345,
        subset='validation')

    return train_datagen_flow


def create_model(input_shape):
    backbone = ResNet50(input_shape=(input_shape),
                 include_top=False,
                 weights='/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='relu'))
    optimizer = Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
    return model


def train_model(model, train_data, test_data, batch_size=None, epochs=9,
                steps_per_epoch=None, validation_steps=None):
    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)
    model.fit(train_data,
              validation_data=test_data,
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2)
    return model