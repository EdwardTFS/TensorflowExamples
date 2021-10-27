#mnist data
#https://matplotlib.org/stable/users/installing.html  - python -m pip install -U matplotlib
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

digit = train_images[0]

import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
print(train_labels[0])