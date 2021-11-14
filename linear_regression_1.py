from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras

a = 1.0
b = 0.0
c = -1.0
d = 0.0
f = lambda x: a*x**3 + b*x**2 + c*x + d 

def generate_data(num, err = 0):
    min,max = -2,2
    X = np.linspace(min, max, num)
    Y = f(X) 
    Y = Y * (1.0 + err * np.random.randn(num)) # add some error
    return X, Y
trX, trY = generate_data(100,0.1)
testX, testY = generate_data(10)
scaleX, scaleY = max(trX)-min(trX), max(trY)-min(trY)
trX /= scaleX
trY /= scaleY
testX /= scaleX
testY /= scaleY

model_linear = keras.Sequential([keras.layers.Dense(units=1, input_shape=(1,))])

model_linear.compile(
optimizer='sgd', 
loss='mse')

model_linear.fit(trX, trY, epochs=500,verbose=0)


#evaluate after fit
print("EVALUATE")
model_linear.evaluate(testX,  testY, verbose=1)

weights,bias = model_linear.layers[0].get_weights()
print(weights,bias)


#plot the data
plt.plot(trX,trY)
plt.plot(testX, model_linear.predict(testX), color="red")
plt.show()