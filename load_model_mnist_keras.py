
import numpy as np
import time
import matplotlib.pyplot as plt
import keras.models as km
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout, Activation
from keras.utils import np_utils

predictions = ['T-shirt/top', 'trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

(X_train, y_train), (X_test, y_test) = mnist.load_data()
num_pixels = X_train.shape[1] * X_train.shape[2]
num_classes = 10

t1 = time.time()
X_train = X_train.reshape(60000, 784) / 255
X_test = X_test.reshape(10000, 784) / 255
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# let's print the shape before we reshape and normalize
print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)
t2 = time.time()
print("Preprocessing took %.2f sec." % (t2 - t1))

print('Loading model and weights.')
json_file = open('mnist.json', 'r')
loaded_nnet = json_file.read()
json_file.close()

model = km.model_from_json(loaded_nnet)
model.load_weights('mnist.h5')

print("Training network ...")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Making predictions...")
t3 = time.time()
for i in range(9):
	img = np.array(X_test[i][np.newaxis,:])
	preds = model.predict_classes(img)
	print("Image[", i, "] - Me thinks me saw a : ", predictions[int(preds[0])] )
t4 = time.time()
print("Predictions took %.2f sec." % (t4 - t3))

print('Total running time %.2f sec' % (t4 - t1))



