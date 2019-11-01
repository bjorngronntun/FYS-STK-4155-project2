import os
os.sys.path.append(os.path.dirname(os.path.abspath('.')))
print(os.sys.path)
from models.neural import *

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6)

nn = NeuralNetwork([
    {
        'neurons': 50,
        'activation': 'tanh'
    },
    {
        'neurons': 50,
        'activation': 'tanh'
    },
    {
        'neurons': 3,
        'activation': 'softmax'
    }
])

nn.fit(X_train, y_train, iterations=5000, batch_size=10, validation=True, stopping_accuracy=0.93)

final_predictions = nn.predict(X_test)
print('---------')
print('Test accuracy', accuracy_score(y_test, final_predictions))
