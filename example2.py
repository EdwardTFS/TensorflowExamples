#based on https://developers.google.com/codelabs/tensorflow-1-helloworld and example 1

#effectively the same as example1.py

import tensorflow as tf
import numpy as np
from tensorflow import keras

#function for creating data
f = lambda x: x * 3 + 1 

#learn data
xs = np.arange(0.0,1.0,.001)
ys = f(xs)

#test data
x_test = np.array([0.0, 0.1, 0.2, 0.4, 0.8,1.0], dtype=float)
y_test = f(x_test) 

#create model

model = tf.keras.Sequential([keras.layers.Dense(units=1, activation='linear',input_shape=(None,1))])
model.summary()
#compile model
#opt='sgd'
#opt = tf.keras.optimizers.SGD(learning_rate=0.2)
#opt = tf.keras.optimizers.RMSprop(learning_rate=0.3, momentum=0.0)
opt = tf.keras.optimizers.Adam(learning_rate=0.2)
model.compile(optimizer=opt, loss='mse',metrics=['mae','mse'])


#fit
print("FIT")
model.fit(xs, ys, epochs=10,verbose=0)

weights,bias = model.layers[0].get_weights()
print(weights,bias)


#evaluate after fit
print("EVALUATE")
model.evaluate(x_test,  y_test, verbose=1)

weights,bias = model.layers[0].get_weights()
print(weights,bias)

#predict
print("PREDICT")
print(model.predict([3]),f(3))