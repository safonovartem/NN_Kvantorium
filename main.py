import keras
import tensorflow
import numpy

#import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
# % matplotlib inline

from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(x_train.shape)
print(y_train.shape)

plt.imshow(x_train[12], cmap='binary')
plt.axis('off')
print(y_train[12])

X_train /= 255
X_test  /= 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test  = keras.utils.to_categorical(y_test,  10)

model = Sequential()

model.add(Dense(32, activation='relu', input_shape=(X_train[0].shape)))

model.add(Dense(64,  activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))

model.add(Flatten())