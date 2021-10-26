#based on https://developers.google.com/codelabs/tensorflow-1-helloworld

import sys
print("Python version:",sys.version)

import tensorflow as tf
import numpy as np
from tensorflow import keras
print("TensorFlow version:", tf.__version__)

#create model
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

#compile model
model.compile(optimizer='sgd', loss='mean_squared_error')

#function for creating data
f = lambda x: x * 3 + 1 

#learn data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = f(xs)

#test data
x_test = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
y_test = f(x_test) 

#fit 1
print("FIT")
model.fit(xs, ys, epochs=20)

#evaluate after fit 1
print("EVALUATE")
model.evaluate(x_test,  y_test, verbose=2)

#fit 2
print("FIT2")
model.fit(xs, ys, epochs=100)

#evaluate after fit 2
print("EVALUATE2")
model.evaluate(x_test,  y_test, verbose=2)

#predict
print("PREDICT")
print(model.predict(np.array(range(10))))