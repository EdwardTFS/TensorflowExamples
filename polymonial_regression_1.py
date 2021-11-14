from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import PolynomialFeatures

#wip 
#sklearn - not needed
#learning rate vs x scaling

n = 3

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
scaleX, scaleY = max(trX)-min(trX), max(trY)-min(trY)
#trX /= scaleX
trY /= scaleY

trX_expanded = np.expand_dims(trX, axis=1)
poly = PolynomialFeatures(3)
trX_poly = poly.fit_transform(trX_expanded)



model_poly = keras.Sequential([keras.layers.Dense(units=1, input_shape=(n+1,))])
opt = keras.optimizers.SGD()
#opt = keras.optimizers.SGD(learning_rate=0.1)

model_poly.compile(
optimizer=opt, 
loss='mse')

model_poly.fit(trX_poly, trY, epochs=1000,verbose=0)


#evaluate after fit
print("EVALUATE")
#model_linear.evaluate(testX,  testY, verbose=1)

weights,bias = model_poly.layers[0].get_weights()
print(weights,bias)


#plot the data
plt.plot(trX,trY)
plt.plot(trX, model_poly.predict(trX_poly), color="red")
plt.show()