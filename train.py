from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D 
from keras.layers import UpSampling2D, Convolution2D, ZeroPadding2D
from keras.optimizers import Adam

import numpy as np
from numpy import genfromtxt
from numpy.testing import assert_allclose
from scipy import misc
import matplotlib.pyplot as plt

import os
from skimage.transform import resize
from skimage.io import imsave
from PIL import Image

from google.colab import drive
drive.mount('/content/drive')

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def U_net(image_rows, image_cols, img_channels):

    inputs = Input((image_rows, image_cols, img_channels))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
        
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


K.set_image_data_format('channels_last')

NB_EPOCH = 40
BATCH_SIZE = 32
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIMIZER = Adam()
IMG_ROWS, IMG_COLS, IMG_CHANNELS = 320, 240, 1
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)

Train = 1314
Valid = 145

x_train = []
y_train = []

x_valid = []
y_valid = []


path = os.getcwd()+'people\\'

print('__________LOAD TRAIN DATA__________')
for i in range(1315):
  x_train.append((np.array(Image.open('/content/drive/My Drive/Colab Notebooks/people/train/'+str(i)+'.jpg'))).dot([0.299, 0.587, 0.114]))
  y_train.append(np.array(Image.open('/content/drive/My Drive/Colab Notebooks/people/train_mask/'+str(i)+'.png'))) 
  if i%130==0: print(i//13,' %')
print('__________END LOAD TRAIN DATA__________') 

print('__________LOAD VALID DATA__________')
for i in range(145):
  x_valid.append((np.array(Image.open('/content/drive/My Drive/Colab Notebooks/people/valid/'+str(1315+i)+'.jpg'))).dot([0.299, 0.587, 0.114]))
  y_valid.append(np.array(Image.open('/content/drive/My Drive/Colab Notebooks/people/valid_mask/'+str(1315+i)+'.png')))
print('\n__________END LOAD VALID DATA__________')

print('__________STAR PREPROCCECING__________')
x_train = np.array(x_train)
x_train = x_train.astype('float32')
x_train /= 255
x_train = x_train[:, :, :, np.newaxis]


y_train = np.array(y_train)
y_train = y_train.astype('float32')
y_train /= 255
y_train = y_train[:, :, :, np.newaxis]


x_valid = np.array(x_valid)
x_valid = x_valid.astype('float32')
x_valid /= 255
x_valid = x_valid[:, :, :, np.newaxis]


y_valid = np.array(y_valid)
y_valid = y_valid.astype('float32')
y_valid /= 255
y_valid = y_valid[:, :, :, np.newaxis]
print('__________END PREPROCCECING__________')


history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, 
                    shuffle=True, validation_split=VALIDATION_SPLIT, callbacks=[model_checkpoint])

score = model.evaluate(x_valid, y_valid, batch_size=BATCH_SIZE, verbose=VERBOSE)

plt.plot(history.history['dice_coef'])
plt.plot(history.history['val_dice_coef'])
plt.title('model accuracy')
plt.ylabel('dice_coef')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc = 'upper left')
plt.show()
