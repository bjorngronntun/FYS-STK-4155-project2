import numpy as np
import unittest

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.1, seed=3):
        print('Initializing')
        self.layers = layers
        self.W = []
        self.biases = []
        self.activations = []
        self.errors = []
        self.seed = seed
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def softmax(self, x):
        exponentials = np.exp(x)
        denominator = np.sum(exponentials, axis=1)
        denominator_matrix = np.tile(denominator, (exponentials.shape[1], 1)).transpose()
        return exponentials/denominator_matrix

    def forward_pass(self, X, training=True):
        # print('Forward pass')
        current_activation = X
        for layer_no in range(len(self.layers)):
            current_activation = np.dot(current_activation, self.W[layer_no]) + self.biases[layer_no]

            # Apply activation function
            if self.layers[layer_no]['activation'] == 'sigmoid':
                activation_function = self.sigmoid
            elif self.layers[layer_no]['activation'] == 'softmax':
                activation_function = self.softmax
            current_activation = activation_function(current_activation)

            self.activations[layer_no] = current_activation

        return self.activations[-1]

    def backwards_pass(self, X, y):
        output_error = (self.activations[-1] - y)*(self.activations[-1]*(-self.activations[-1])+self.activations[-1])/X.shape[0]
        self.errors[-1] = output_error
        error = output_error
        for layer_no in range(len(self.layers) - 2, -1, -1):
            error = self.activations[layer_no]*(1 - self.activations[layer_no])*np.dot(error, np.transpose(self.W[layer_no + 1]))
            self.errors[layer_no] = error

        # Update output layer weights
        self.W[-1] -= self.learning_rate*np.dot(np.transpose(self.activations[-2]), self.errors[-1])
        self.biases[-1] -= self.learning_rate*np.sum(self.errors[-1], axis=0)

        # Update hidden layer weights
        for layer_no in range(len(self.layers) - 2, 0, -1):
            self.W[layer_no] -= self.learning_rate*np.dot(np.transpose(self.activations[layer_no - 1]), self.errors[layer_no])
            self.biases[layer_no] -= self.learning_rate*np.sum(self.errors[layer_no], axis=0)

        # Update first layer weights
        self.W[0] -= self.learning_rate*np.dot(np.transpose(X), self.errors[0])
        self.biases[0] -= self.learning_rate*np.sum(self.errors[0], axis=0)

    def fit(self, X, y, iterations=1000, batch_size=16):
        np.random.seed(self.seed)
        self.W.append(np.random.random((X.shape[1], self.layers[0]['neurons'])) - 0.5)
        for layer_no in range(len(self.layers) - 1):
            self.W.append(np.random.random((self.layers[layer_no]['neurons'], self.layers[layer_no + 1]['neurons'])) - 0.5)
        for layer_no in range(len(self.layers)):
            self.biases.append(np.random.random(self.layers[layer_no]['neurons']) - 0.5)

        self.activations = [None for _ in range(len(self.layers))]
        self.errors = [None for _ in range(len(self.layers))]

        for epoch in range(iterations):
            print('Epoch', epoch + 1)
            # Here, mini-batching should be done.
            if batch_size < X.shape[0]:
                indices = np.random.choice(X.shape[0], batch_size)
                X_batch = X[indices, :]
                y_batch = y[indices]
            else:
                X_batch = X
                y_batch = y
            self.forward_pass(X_batch, training=True)
            self.backwards_pass(X_batch, y_batch)

    def predict(self, X):
        print('Predicting with X shape', X.shape)
        output = self.forward_pass(X, training=False)
        predictions = np.argmax(output, axis=1)
        # Some tresholding here...
        # print(output)
        print('Returning predictions with shape', predictions.shape)
        return predictions

class LogRegTest(unittest.TestCase):
    def test_fit(self):
        X = np.array([  [1, 2, 3],
                        [1, 4, 5],
                        [1, 7, 6],
                        [1, 9, 9]])
        y = np.array([1, 0, 1, 0])
        lr = LogReg()
        lr.fit(X, y)

if __name__ == '__main__':
    # unittest.main()
    nn = NeuralNetwork([
        {
            'neurons': 5,
            'activation': 'sigmoid'
        },
        {
            'neurons': 2,
            'activation': 'softmax'
        }
    ])
    X = np.array([  [3, 2, 3],
                    [1, 4, 5],
                    [2, 7, 6],
                    [5, 9, 9]])
    y = np.array([  [1, 0],
                    [0, 1],
                    [0, 1],
                    [1, 0]])
    nn.fit(X, y)

    print('And now, for prediction')
    prediction = nn.predict(X)
    print('Target:', y)
    print('Predictions:', prediction)
