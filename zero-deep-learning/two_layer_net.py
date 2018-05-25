import sys
from functions import *
from gradient import numerical_gradient

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        # print('in predict')
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # print('out predict')
        return y

    def loss(self, x, t):
        # print('in loss')
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        acc = np.sum(y == t) / float(x.shape[0])
        # print('out loss')
        return acc

    def numerical_gradient(self, x, t):
        print('in numerical gradient')
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        print('W1 at numerical gradient')
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        print('b1 at numerical gradient')
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        print('W2 at numerical gradient')
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        print('b2 at numerical gradient')
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        print('out numerical gradient')
        return grads


