import os
os.sys.path.append(os.path.dirname(os.path.abspath('.')))

from models.neural import *
from preprocessing.preprocessing import *

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = get_design_matrix(), get_target_values()
print(np.std(X, axis=0))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6)

nn = NeuralNetwork([

    {
        'neurons': 64,
        'activation': 'tanh'
    },
    {

        'neurons': 64,
        'activation': 'tanh'
    },

    {
        'neurons': 2,
        'activation': 'softmax'
    }
])

nn.fit(X_train, y_train, iterations=500000, batch_size=64, validation=True, validation_size=0.05, stopping_accuracy=0.80)

final_predictions = nn.predict(X_test)
print('---------')
print('Test accuracy', accuracy_score(y_test, final_predictions))
print('Final predictions:', final_predictions)
print('Max final prediction:', np.max(final_predictions))
