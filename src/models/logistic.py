class LogReg:
    def __init__(self, max_iter = 1000, lr=0.01, tol=0.00001, stochastic=False):
        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self.stochastic = stochastic
    def fit(self, X, y):
        # Gradient descent scheme here
        self.beta = np.random.random(X.shape[1]) - 0.5
        # print("Beta:", self.beta)

        for iter in range(self.max_iter):
            if self.stochastic:
                batch_size = 64 # User defined?
                batch_indices = np.random.choice(X.shape[0], batch_size)
                print(batch_indices)
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
            else:
                X_batch = X
                y_batch = y
            sigmoid_input = np.dot(X_batch, self.beta)
            sigmoid_output = 1/(1 + np.exp(-sigmoid_input))
            gradients = (1/X_batch.shape[0])*np.dot(X_batch.transpose(), (sigmoid_output - y_batch))
            max_abs_gradient = np.max(np.abs(gradients))
            print("Max absolute gradient component:", max_abs_gradient)
            if max_abs_gradient < self.tol:
                print('BREAK')
                break
            self.beta = self.beta - self.lr*gradients

            # predictions = self.predict(X)

            # print("Gradients:", gradients)
            # print("Beta:", self.beta)
            # print("predictions:", predictions)
            # print("Max prediction:", np.max(predictions))

    def predict(self, X):
        sigmoid_input = np.dot(X, self.beta)
        sigmoid_output = 1/(1 + np.exp(-sigmoid_input))
        # print("Sigmoid output:", sigmoid_output)
        predictions = np.where(sigmoid_output < 0.5, 0, 1)
        return predictions
