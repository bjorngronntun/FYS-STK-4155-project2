import os
os.sys.path.append(os.path.dirname(os.path.abspath('.')))
print(os.sys.path)
from models.neural import *
from preprocessing.preprocessing import *

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = get_design_matrix(), LabelBinarizer().fit_transform(get_target_values())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6)

nn = NeuralNetwork([
    {
        'neurons': 10,
        'activation': 'sigmoid'
    },
    {
        'neurons': 10,
        'activation': 'sigmoid'
    },
    {
        'neurons': 2,
        'activation': 'softmax'
    }
])

nn.fit(X_train, y_train, iterations=50000)

predictions = nn.predict(X_test)
target = np.argmax(y_test, axis=1)
print(predictions.shape)
print(target.shape)
print('Target values:', np.argmax(y_test, axis=1))
print('Test accuracy', accuracy_score(np.argmax(y_test, axis=1), predictions))
