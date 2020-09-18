
import numpy as np
import time
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout, Activation
from keras.utils import np_utils
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()
num_pixels = X_train.shape[1] * X_train.shape[2]
num_classes = 10

fig = plt.figure()
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.tight_layout()
  plt.imshow(X_test[i], cmap='gray', interpolation='none')
  plt.title("Digit: %d" % (y_test[i]))
  plt.xticks([])
  plt.yticks([])
fig.show()
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

t1 = time.time()
# building a linear stack of layers with the sequential model
model = Sequential()
model.add(Dense(512, input_shape = (784,)))
model.add(Activation('relu'))                            
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))


print("Training network ...")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Fitting data")
model.fit(X_train, Y_train, batch_size=128, epochs=15, 
          verbose=1, validation_data=(X_test, Y_test))
t2 = time.time()
print("Training took %.2f sec." % (t2 - t1))

print("Making predictions...")
t1 = time.time()
for i in range(9):
	img = np.array(X_test[i][np.newaxis,:])
	preds = model.predict_classes(img)
	print("Image[", i, "] - Me thinks me saw a : ", int(preds[0]) )
t2 = time.time()
print("Predictions took %.2f sec." % (t2 - t1))



