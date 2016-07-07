from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import numpy as np
from sklearn.cross_validation import train_test_split
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from keras.callbacks import TensorBoard
import h5py

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses_batch = []
        self.val_losses_epoch = []
    def on_batch_end(self, batch, logs={}):
        self.losses_batch.append(logs.get('loss'))
    def on_epoch_end(self, batch, logs={}):
        self.val_losses_epoch.append(logs.get('val_loss'))


#Loading the dataset
with h5py.File('data.h5','r') as hf:
    print('List of arrays in this file: \n', hf.keys())
    data = hf.get('X')
    X = np.array(data, dtype=np.float32).transpose((0, 3, 1, 2))/255.0
    data = hf.get('y')
    y = np.array(data, dtype=np.uint8)

y = np.expand_dims(y, axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)


input_img = Input(shape=(3, 32, 32))

x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init='glorot_normal')(input_img)

x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='glorot_normal')(x)

x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init='glorot_normal')(x)

x = Convolution2D(8, 3, 3, activation='relu', border_mode='same', init='glorot_normal')(x)

x = Convolution2D(4, 3, 3, activation='relu', border_mode='same', init='glorot_normal')(x)

x = Convolution2D(2, 3, 3, activation='relu', border_mode='same', init='glorot_normal')(x)

decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same', init='glorot_normal')(x)

model = Model(input_img, decoded)
model.compile(optimizer='adadelta', loss='binary_crossentropy')
print (model.summary())
checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)
history = LossHistory()
model.fit(X_train, y_train,
                nb_epoch=100,
                batch_size=255,
                shuffle=True,
                validation_data=(X_test, y_test), callbacks=[history, checkpointer, history])

val_losses = history.val_losses_epoch
val_axis = np.arange(len(val_losses))
losses_batch = history.losses_batch
loss_batch_axis = np.arange(len(losses_batch))

plt.subplot(1, 2, 1)
plt.plot(val_axis, val_losses, color="blue", label="validation_loss")
plt.subplot(1, 2, 2)
plt.plot(loss_batch_axis, losses_batch, color="red", label="training_loss")
plt.savefig('losses.png')


prediction = model.predict(X_test)
import random

l = random.sample(np.arange(X_test.shape[0]), 10)

for num, index in enumerate(l):
    fig = plt.figure()
    fig.suptitle("Results", fontsize=16)
    mask = (prediction[index,:,:,:]>0.5)*255
    ax = plt.subplot(1,3,1)
    ax.set_title("Prediction")
    plt.imshow(mask.reshape((32,32)), cmap="Greys_r")

    ax = plt.subplot(1,3,2)
    ax.set_title("Input Image")
    plt.imshow(X_test[index,:,:,:].transpose((1, 2, 0)))

    ax = plt.subplot(1,3,3)
    ax.set_title("Target Mask")
    plt.imshow((y_test[index]*255).reshape((32,32)), cmap="Greys_r")
    plt.savefig('results%d.png'%num)
