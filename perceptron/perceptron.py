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
        self.num_inputs_nodes = 2
        self.num_neurons = 1
        self.pnn = Perceptron(self.num_inputs_nodes, self.num_neurons)

        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        # XORtargets = np.array([[0], [1], [1], [0]])
        targets = {'And data': np.array([[0], [0], [0], [1]]),
                   'Or data': np.array([[0], [1], [1], [1]])}
        self.assertTrue(targets['And data'].shape == self.pnn.predict(x).shape)

        for key, y in targets.items():
            self.pnn.train(x, y, 0.2, 200)
            preds = self.pnn.predict(x)
            self.assertTrue(np.array_equal(preds, y), msg="given {} not equals to the right answer: {}".format(
                preds, y
            ))


if __name__ == '__main__':
    unittest.main()