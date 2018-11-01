import numpy as np


class Perceptron:
    # однослойная нейросеть: n входов с весами, суммирование с активацией -
    # порожек бинарный
    # т.к. сетка однслойная, то каждый нейрон является одновременно и выходом сети

    def __init__(self, num_inputs_nodes, num_neurons):
        self.num_inputs_nodes = num_inputs_nodes
        # + 1 - means adding bias
        weights_no_bias = np.random.randn(num_inputs_nodes + 1, num_neurons)
        bias = np.ones((num_inputs_nodes))
        self.num_neurons = num_neurons

    def check_x(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError
        return len(x.shape) == 2 and x.shape[1] == self.num_inputs_nodes

    def check_y(self, y):
        if not isinstance(y, np.ndarray):
            raise TypeError
        return len(y.shape) == 2 and y.shape[1] == self.num_neurons

    def train(self, x_train, y_train):
        if not (self.check_x(x_train) or self.check_y(y_train)):
            raise ValueError

        activations = np.dot(x_train, self.weights)


    def predict(self, x_test):
        activations = np.dot(x_test, self.weights)
        return np.where(activations > 0, 1, 0)
