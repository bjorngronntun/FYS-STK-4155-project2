import os
os.sys.path.append(os.path.dirname(os.path.abspath('.')))
print(os.sys.path)
from models.neural import *

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, LabelBinarizer().fit_transform(iris.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6)

nn = NeuralNetwork([
    {
        'neurons': 4,
        'activation': 'sigmoid'
    },
    {
        'neurons': 3,
        'activation': 'softmax'
    }
])

nn.fit(X_train, y_train, iterations=5000)

predictions = nn.predict(X_test)
target = np.argmax(y_test, axis=1)
print(predictions.shape)
print(target.shape)
print('Target values:', np.argmax(y_test, axis=1))
print('Test accuracy', accuracy_score(np.argmax(y_test, axis=1), predictions))
