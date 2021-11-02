from matplotlib import pyplot
import numpy as np
from tensorflow import keras

def generate_data():
    X = np.arange(-30, 30, 1)
    y = 9*X**3 + 5*X**2 + np.random.randn(60)*1000
    return X, y
trX, trY = generate_data()
trX = trX/max(trX)
trY = trY/max(trY)

model_linear = keras.Sequential([keras.layers.Dense(units=1, input_shape=(1,))])

model_linear.compile(
optimizer='sgd', 
loss='mse')

model_linear.fit(trX, trY, epochs=500)

#plot the data
pyplot.scatter(trX, trY)
pyplot.plot(trX, model_linear.predict(trX), color="red")
pyplot.show()