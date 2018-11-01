import unittest
import sklearn
from perceptron_class import Perceptron
import numpy as np

class TestPerceptron(unittest.TestCase):

    def setUp(self):
        self.num_inputs_nodes = 10
        self.num_neurons = 10
        self.pnn = Perceptron(self.num_inputs_nodes, self.num_neurons)

    def test__init__(self):
        num_of_inputs_plus_bias = self.num_inputs_nodes + 1
        self.assertTrue(self.pnn.weights.shape[0], num_of_inputs_plus_bias)
        self.assertTrue(self.pnn.weights.shape[1], self.num_neurons)
        self.assertEqual(self.pnn.weights.size, num_of_inputs_plus_bias * self.num_neurons)

    def test_train(self):
        # XOR problem
        self.num_inputs_nodes = 2
        self.num_neurons = 1
        self.pnn = Perceptron(self.num_inputs_nodes, self.num_neurons)

        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 1])

        weights_init = self.pnn.weights
        self.pnn.train(x, y)
        weights_final = self.pnn.weights

        self.assertFalse(np.array_equal(weights_init, weights_final))

        x_test = x
        preds = self.pnn.predict(x_test)
        self.assertTrue(np.array_equal(preds, y))


if __name__ == '__main__':
    unittest.main()