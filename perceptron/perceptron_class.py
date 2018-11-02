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

    def train(self, x_train, y_train, eta):
        if not (self.check_x(x_train) or self.check_y(y_train)):
            raise ValueError

        # activations = np.dot(x_train, self.weights)
        preds = self.predict(x_train)
        # print(preds.shape, y_train.shape)

        bias = np.ones((x_train.shape[0], 1))
        x_train_tmp = np.concatenate((x_train, bias), axis=1)

        # stochastic gradient descent applied to perceptron criterion
        self.weights -= eta * np.dot(x_train_tmp.T, preds - y_train)
        print("{}", str(self.weights))

        # for over_samples in range(x_train.shape[0]):
        #     for over_neurons in range(self.num_neurons):
        #         for over_inupts in range(self.num_inputs_nodes):
        #             self.weights[over_inupts, over_neurons] -= \
        #                 eta * (y_train[over_neurons] - preds[over_neurons]) * \
        #                 x_train[over_samples, over_inupts]

            # print("{}\{} iteration: {}".format(over_samples, x_train.shape[0],
            #                                              str(self.weights)))
            # self.logger.info("{}\{} iteration: {}".format(over_samples, x_train.shape[0],
            #                                              str(self.weights)))

    def predict(self, x_test):
        bias = np.ones((x_test.shape[0], 1))
        x_test = np.concatenate((x_test, bias), axis=1)

        activations = np.dot(x_test, self.weights)
        return np.where(activations > 0, 1, 0)
