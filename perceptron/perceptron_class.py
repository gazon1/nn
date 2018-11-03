import numpy as np
import logging
from os import path, remove


class Perceptron:
    # однослойная нейросеть: n входов с весами, суммирование с активацией -
    # порожек бинарный
    # т.к. сетка однслойная, то каждый нейрон является одновременно и выходом сети

    def __init__(self, num_inputs_nodes, num_neurons):

        self.logger = logging.basicConfig(filename='logs.log')

        self.num_inputs_nodes = num_inputs_nodes
        # + 1 - means adding bias
        weights_no_bias = np.random.randn(num_inputs_nodes, num_neurons)
        bias = np.ones((1, num_neurons))
        self.weights = np.concatenate((weights_no_bias, bias), axis=0)
        self.num_neurons = num_neurons

    def check_x(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError
        return len(x.shape) == 2 and x.shape[1] == self.num_inputs_nodes

    def check_y(self, y):
        if not isinstance(y, np.ndarray):
            raise TypeError
        return len(y.shape) == 2 and y.shape[1] == self.num_neurons

    def train(self, x_train, y_train, eta, num_epoch):
        if not (self.check_x(x_train) or self.check_y(y_train)):
            raise ValueError

        # bias = np.ones((x_train.shape[0], 1))
        # x_train = np.concatenate((x_train, bias), axis=1)

        i = 0
        # stochastic gradient descent applied to perceptron criterion
        for epoch in range(num_epoch):
            for x_train_batch, y_train_batch in zip(x_train, y_train):
                x_train_batch = x_train_batch[np.newaxis, :]
                y_train_batch = y_train_batch[np.newaxis, :]

                preds = self.predict(x_train_batch)

                bias = np.ones((x_train_batch.shape[0], 1))
                x_train_batch = np.concatenate((x_train_batch, bias), axis=1)

                self.weights -= eta * np.dot(x_train_batch.T, preds - y_train_batch)

            print("Epoch {}. Weights: {}".format(epoch, str(self.weights)))

    def predict(self, x_test):
        bias = np.ones((x_test.shape[0], 1))
        x_test = np.concatenate((x_test, bias), axis=1)

        activations = np.dot(x_test, self.weights)
        return np.where(activations > 0, 1, 0)
