import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
class LogReg:
    def __init__(self, iterations = 10000000, lr=0.0005,  stochastic=True, seed=12):
        self.iterations = iterations
        self.lr = lr
        self.stochastic = stochastic
        self.seed = seed
    def fit(self, X, y, batch_size=128, validation=True, validation_size=0.2):
        # Gradient descent scheme here
        self.beta = np.random.random(X.shape[1]) - 0.5
        # print("Beta:", self.beta)
        # Possible transformations of data

        if validation:
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=validation_size, random_state=self.seed)
        else:
            X_train = X
            y_train = y

        # The actual training
        for epoch in range(1, self.iterations + 1):
            if epoch % 1000 == 0:
                print('Epoch', epoch)
            # Here, mini-batching should be done.
            if batch_size < X_train.shape[0]:
                indices = np.random.choice(X_train.shape[0], batch_size)
                X_batch = X_train[indices, :]
                y_batch = y_train[indices]
            else:
                X_batch = X_train
                y_batch = y_train

            sigmoid_input = np.dot(X_batch, self.beta)
            sigmoid_output = 1/(1 + np.exp(-sigmoid_input))
            gradients = (1/X_batch.shape[0])*np.dot(X_batch.transpose(), (sigmoid_output - y_batch))
            self.beta = self.beta - self.lr*gradients

            if validation:
                predictions = self.predict(X_valid)

                acc = accuracy_score(y_valid, predictions)
                if epoch % 1000 == 0:
                    print('Accuracy:', acc)
                if acc > 0.81:
                    return self
        return self

    def predict(self, X):
        sigmoid_input = np.dot(X, self.beta)
        sigmoid_output = 1/(1 + np.exp(-sigmoid_input))
        # print("Sigmoid output:", sigmoid_output)
        predictions = np.where(sigmoid_output < 0.5, 0, 1)
        return predictions
