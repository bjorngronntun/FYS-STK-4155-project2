import os
os.sys.path.append(os.path.dirname(os.path.abspath('.')))

from models.logistic import *
from preprocessing.preprocessing import *

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = get_design_matrix(), get_target_values()
print(np.std(X, axis=0))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6)

lr = LogReg()
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
print(accuracy_score(y_test, predictions))
