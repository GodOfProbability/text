from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, np
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
import time
from matplotlib import pyplot as plt

batch_size = 32
nb_classes = 10
nb_epoch = 200
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
model.summary()

model.load_weights('weights.hdf5')

# Let us just apply the method to a single image and see what are the results


img = plt.imread('trial_image.jpeg').transpose((2, 0, 1))
channels, rows, cols = img.shape

enlarged = np.lib.pad(img, ((0, 0), (16, 16), (16, 16)), mode='mean')
# print (enlarged.shape)
heat_map = np.zeros((rows, cols))
t1 = time.clock()
for r in np.arange(rows - 32):
    for c in np.arange(cols - 32):
        temp = np.expand_dims(enlarged[:, r:r + 32, c:c + 32], axis=0)
        heat_map[r, c] = model.predict(temp)[0]
print("Time taken for one image: ", time.clock() - t1)
