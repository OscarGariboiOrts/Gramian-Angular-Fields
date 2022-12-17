import time

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import sequence as seqq
from livelossplot import PlotLossesKeras

from keras.applications import ResNet50, MobileNet
from keras.models import Model, load_model

from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField

import numpy as np
import sys
import os
from shutil import copyfile, move

#import pywt

import pandas as pd

def model_keras(name):
    '''
    Function to load the pretrained models
    '''
    base_model = available_models[name](
        include_top=False,
        weights='imagenet',
    )

    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(1), activation='linear')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    base_model.trainable = True

    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])


source_dir = 'regression_images_gadf/'

images = [x for x in os.listdir(source_dir) if x.endswith('jpg')]

indexes = [x.split('.')[0].split('_')[-1] for x in images]
indexes = [int(x) for x in indexes]

ref1 = 'ref1.txt' 

with open(ref1, 'r') as f:
    ref = [float(x.split(';')[1].strip()) for x in f.readlines()[:]]

exponents = [ref[x] for x in indexes]

df = pd.DataFrame()

df['id'] = images
df['exponents'] = exponents

train_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.2, height_shift_range=0.2,)

test_datagen = ImageDataGenerator(rescale=1./255)

target_size = (95, 63)
img_width = target_size[0]
img_height = target_size[1]

train_generator = train_datagen.flow_from_dataframe(dataframe=df, directory=source_dir, x_col="id", y_col="exponent", subset='training', batch_size=32, shuffle=True, class_mode="other", target_size=(img_width, img_height))

valid_generator = train_datagen.flow_from_dataframe(dataframe=df, directory=source_dir, x_col="id", y_col="exponent", subset="validation", batch_size=32, shuffle=True, class_mode="other", target_size=(img_width, img_height))

steps_per_epoch_train = train_generator.n // train_generator.batch_size
steps_per_epoch_valid = valid_generator.n // valid_generator.batch_size

available_models = {'mobilenet': MobileNet, 'resnet': ResNet50}

model = model_keras('mobilenet')

path_to_model = 'models/'

if not os.path.isdir(path_to_model):
    os.makedirs(path_to_model, exist_ok=True)

es = EarlyStopping(monitor='val_mean_absolute_error', mode='min', verbose=1, patience=20)
mc = ModelCheckpoint(path_to_model + 'regression_gadf_mobilenet-{epoch:04d}-{val_mean_absolute_error:.4f}.h5', monitor='val_mean_absolute_error', mode='min', verbose=0, save_best_only=True)

history = model.fit_generator(generator=train_generator, steps_per_epoch=steps_per_epoch_train, validation_data=valid_generator, validation_steps=steps_per_epoch_valid, epochs=200, verbose=1, callbacks=[es, mc])

print("Program finished!")
