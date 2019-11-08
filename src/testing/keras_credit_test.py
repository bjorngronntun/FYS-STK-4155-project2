import os
os.sys.path.append(os.path.dirname(os.path.abspath('.')))

from preprocessing.preprocessing import *

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelBinarizer

X, y = get_design_matrix(), LabelBinarizer().fit_transform(get_target_values())
print(np.std(X, axis=0))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6)

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Train the model, iterating on the data in batches of 32 samples
model.fit(X_train, y_train, epochs=30, batch_size=32)
