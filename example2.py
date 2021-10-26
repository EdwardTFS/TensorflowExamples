#based on https://developers.google.com/codelabs/tensorflow-1-helloworld and example 1

#work in progress

import tensorflow as tf
import numpy as np
from tensorflow import keras

#create model
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.summary()
#compile model
model.compile(optimizer='sgd', loss='mean_squared_error')

#function for creating data
f = lambda x: x * 3 + 1 

#learn data
xs = np.arange(0.0,10.0,0.1)
ys = f(xs)

#test data
x_test = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
y_test = f(x_test) 

#fit
print("FIT")
model.fit(xs, ys, epochs=50,verbose=2)

#evaluate after fit
print("EVALUATE")
model.evaluate(x_test,  y_test, verbose=2)

#predict
print("PREDICT")
print(model.predict(np.array(range(10))))