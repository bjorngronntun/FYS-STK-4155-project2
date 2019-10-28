import numpy as np
import unittest

def sigmoid(x):
    return 1/(1 + np.exp(-x))
def softmax(x):
    print('SOFTMAX')
    exponentials = np.exp(x)
    denominator = np.sum(exponentials, axis=1)
    return exponentials/np.c_[denominator, denominator]

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


    def forward_pass(self, X, training=True):
        # print('Forward pass')
        current_activation = X
        for layer_no in range(len(self.layers)):
            current_activation = np.dot(current_activation, self.W[layer_no]) + self.biases[layer_no]
            # Apply activation function
            current_activation = self.layers[layer_no]['activation'](current_activation)
            if training:
                self.activations[layer_no] = current_activation
            # print('CURRENT:', current.shape)
        print('Number of activation vectors:', len(self.activations))
        return self.activations[-1]

    def backwards_pass(self, X, y):
        # print('Backwards pass')
        # Find error in output layer:
        # Should be possible to do this with softmax, too.
        # print('Target:', y)
        # print('Activations:', self.activations[-1])
        # output_error = (self.activations[-1] - y)*self.activations[-1]*(1 - self.activations[-1])
        output_error = (self.activations[-1] - y)*(self.activations[-1]*(-self.activations[-1])+self.activations[-1])/X.shape[0]
        print('OUTPUT ERROR', output_error)
        # self.W[:, -1] += self.learning_rate*(np.dot())
        # print('Error:', output_error)
        self.errors[-1] = output_error
        error = output_error
        # print('Number of layers:', len(self.layers))
        for layer_no in range(len(self.layers) - 2, -1, -1):
            # print('Now in layer number', layer_no)
            # Ok, now it is getting harder.
            # print('Activations:', self.activations[-2])
            error = self.activations[layer_no]*(1 - self.activations[layer_no])*np.dot(error, np.transpose(self.W[layer_no + 1]))
            self.errors[layer_no] = error

        # Update output layer weights
        self.W[-1] -= self.learning_rate*np.dot(np.transpose(self.activations[-2]), self.errors[-1])
        self.biases[-1] -= self.learning_rate*np.sum(self.errors[-1], axis=0)
        # Update hidden layer weights
        for layer_no in range(len(self.layers) - 2, 0, -1):
            print('Layer no', layer_no)

            self.W[layer_no] -= self.learning_rate*np.dot(np.transpose(self.activations[layer_no - 1]), self.errors[layer_no])
            self.biases[layer_no] -= self.learning_rate*np.sum(self.errors[layer_no], axis=0)
            print('For loop went fine!')
        # Update first layer weights
        self.W[0] -= self.learning_rate*np.dot(np.transpose(X), self.errors[0])
        self.biases[0] -= self.learning_rate*np.sum(self.errors[0], axis=0)

    def fit(self, X, y):
        np.random.seed(self.seed)
        self.W.append(np.random.random((X.shape[1], self.layers[0]['neurons'])) - 0.5)
        for layer_no in range(len(self.layers) - 1):
            self.W.append(np.random.random((self.layers[layer_no]['neurons'], self.layers[layer_no + 1]['neurons'])) - 0.5)
        for layer_no in range(len(self.layers)):
            self.biases.append(np.random.random(self.layers[layer_no]['neurons']) - 0.5)

        self.activations = [None for _ in range(len(self.layers))]
        self.errors = [None for _ in range(len(self.layers))]
        print('Number of weight matrices:', len(self.W))
        print('Number of bias vectors:', len(self.biases))
        # for layer_no in range(len(self.layers)):
            # print('w shape:', self.W[layer_no].shape)
            # print('bias shape:', self.biases[layer_no].shape)

        for epoch in range(6000):
            # Here, mini-batching should be done.
            self.forward_pass(X, training=True)
            self.backwards_pass(X, y)
            print('Predictions: ', self.predict(X))
        # for i, a in enumerate(self.activations):
            # print('Activation', i, a.shape)
        # for i, e in enumerate(self.errors):
            # print('Error', i, e.shape)
    def predict(self, X):
        output = self.forward_pass(X, training=False)
        # Some tresholding here...
        # print(output)
        return output

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
            'activation': sigmoid
        },
        {
            'neurons': 2,
            'activation': softmax
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
