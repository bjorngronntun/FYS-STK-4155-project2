import numpy as np
import unittest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, mean_squared_error

class NeuralNetwork:
    def __init__(self, layers, seed=3):
        # print('Initializing')
        self.layers = layers
        self.W = []
        self.biases = []
        self.activations = []
        self.errors = []
        self.seed = seed
        self.lb = LabelBinarizer()
        self.lb_fitted = False

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def tanh(self, x):
        return 2/(1 + np.exp(-2*x)) - 1

    def ReLU(self, x):
        return np.where(x < 0, 0.01*x, x)

    def softmax(self, x):
        exponentials = np.exp(x)
        denominator = np.sum(exponentials, axis=1)
        denominator_matrix = np.tile(denominator, (exponentials.shape[1], 1)).transpose()
        return exponentials/denominator_matrix

    def linear(self, x):
        return x

    def forward_pass(self, X):
        # print('Forward pass')
        current_activation = X
        for layer_no in range(len(self.layers)):
            current_activation = np.dot(current_activation, self.W[layer_no]) + self.biases[layer_no]

            # Apply activation function
            if self.layers[layer_no]['activation'] == 'sigmoid':
                activation_function = self.sigmoid
            elif self.layers[layer_no]['activation'] == 'softmax':
                activation_function = self.softmax
            elif self.layers[layer_no]['activation'] == 'tanh':
                activation_function = self.tanh
            elif self.layers[layer_no]['activation'] == 'ReLU':
                activation_function = self.ReLU
            elif self.layers[layer_no]['activation'] == 'linear':
                activation_function = self.linear
            current_activation = activation_function(current_activation)

            self.activations[layer_no] = current_activation

        return self

    def backwards_pass(self, X, y, learning_rate, regularization):
        # Error in output layer
        # print('y.shape[1]', y.shape[1])
        output_error = np.zeros(y.shape)
        if self.layers[-1]['activation'] == 'softmax':
            output_error = (self.activations[-1] - y)*(self.activations[-1]*(-self.activations[-1])+self.activations[-1])/X.shape[0]
        elif self.layers[-1]['activation'] == 'linear':
            # print('activations[-1].shape', self.activations[-1].shape)
            # print('y.shape', y.shape)
            # print('X.shape[0]', X.shape[0])
            output_error = (self.activations[-1] - y)/X.shape[0]
        self.errors[-1] = output_error
        error = output_error
        for layer_no in range(len(self.layers) - 2, -1, -1):
            if self.layers[layer_no]['activation'] == 'sigmoid':
                error = self.activations[layer_no]*(1 - self.activations[layer_no])*np.dot(error, np.transpose(self.W[layer_no + 1]))
            elif self.layers[layer_no]['activation'] == 'tanh':
                error = (1 - (self.activations[layer_no])**2)*np.dot(error, np.transpose(self.W[layer_no + 1]))

            elif self.layers[layer_no]['activation'] == 'ReLU':
                error = (np.where(self.activations[layer_no] < 0, 0.01, 1.0))*np.dot(error, np.transpose(self.W[layer_no + 1]))
            self.errors[layer_no] = error

        # Update output layer weights
        self.W[-1] -= learning_rate*(np.dot(np.transpose(self.activations[-2]), self.errors[-1]) + regularization*self.W[-1])
        self.biases[-1] -= learning_rate*np.sum(self.errors[-1], axis=0)

        # Update hidden layer weights
        for layer_no in range(len(self.layers) - 2, 0, -1):
            self.W[layer_no] -= learning_rate*(np.dot(np.transpose(self.activations[layer_no - 1]), self.errors[layer_no]) + regularization*self.W[layer_no])
            self.biases[layer_no] -= learning_rate*np.sum(self.errors[layer_no], axis=0)

        # Update first layer weights
        self.W[0] -= learning_rate*(np.dot(np.transpose(X), self.errors[0]) + regularization*self.W[0])
        self.biases[0] -= learning_rate*np.sum(self.errors[0], axis=0)

    def fit(    self,
                X,
                y,
                iterations=10000,
                batch_size=16,
                learning_rate=0.01,
                regularization=0.0,
                classification=True,
                validation=False,
                validation_size=0.2,
                stopping_accuracy=0.9):
        np.random.seed(self.seed)

        # Necessary initialization
        self.W.append(np.random.random((X.shape[1], self.layers[0]['neurons'])) - 0.5)
        for layer_no in range(len(self.layers) - 1):
            self.W.append(np.random.random((self.layers[layer_no]['neurons'], self.layers[layer_no + 1]['neurons'])) - 0.5)
        for layer_no in range(len(self.layers)):
            self.biases.append(np.random.random(self.layers[layer_no]['neurons']) - 0.5)

        self.activations = [None for _ in range(len(self.layers))]
        self.errors = [None for _ in range(len(self.layers))]

        # Possible transformations of data
        if classification:
            y = self.lb.fit_transform(y)
            self.lb_fitted = True
        else:
            if y.ndim == 1:
                y = y.reshape((-1, 1))
                # print('y shape:', y.shape)
        if validation:
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=validation_size, random_state=self.seed)
        else:
            X_train = X
            y_train = y

        # The actual training
        for epoch in range(1, iterations + 1):
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
            self.forward_pass(X_batch)
            self.backwards_pass(X_batch, y_batch, learning_rate, regularization)

            if validation:
                predictions = self.predict(X_valid)
                if classification:
                    y_valid_transformed = self.lb.inverse_transform(y_valid)
                    acc = accuracy_score(self.lb.inverse_transform(y_valid), predictions)
                    if epoch % 1000 == 0:
                        print('Accuracy:', acc)
                    if acc > stopping_accuracy:
                        print('Stopped iterating, validation accuracy now is', acc)
                        return self
                else:
                    mse_score = mean_squared_error(y_valid, predictions)
                    if epoch % 1000 == 0:
                        print('MSE:', mse_score)
                    



        return self


    def predict(self, X):
        self.forward_pass(X)
        output = self.activations[-1]
        predictions = output
        if self.lb_fitted:
            predictions = self.lb.inverse_transform(output)
        return predictions

if __name__ == '__main__':
    nn = NeuralNetwork(
        layers = [
            {
                'neurons': 28,
                'activation': 'tanh'
            },
            {

                'neurons': 28,
                'activation': 'tanh'
            }
        ]
    )
    x = np.random.random(10) - 0.5
    print(x)
    print(nn.ReLU(x))
    print(np.where(x < 0, 0, 1))
