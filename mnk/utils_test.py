import unittest
import numpy as np
from utils import get_x, get_squared_error, generate_train_set


class UtilsTest(unittest.TestCase):
    def test_get_x(self):
        num_of_features = 5
        x = get_x(num_of_features)

        self.assertTrue(isinstance(x, np.ndarray))
        self.assertTrue(x.shape == (1, num_of_features))

    def test_get_squared_error(self):
        with self.assertRaises(TypeError):
            get_squared_error('some str', np.array([1]))

        with self.assertRaises(ValueError):
            get_squared_error(np.array([]), np.array([1]))

    def test_generate_train_set(self):
        def init_kwd():
            kwd = {'sigma': 0, 'sample_size': 2, 'num_of_features': 2}
            kwd['w'] = np.random.randn(kwd['num_of_features'])
            return kwd

        with self.assertRaises(AssertionError):
            kwd = init_kwd()
            kwd['sigma'] = -1
            generate_train_set(**kwd)

        with self.assertRaises(AssertionError):
            kwd = init_kwd()
            kwd['sample_size'] = 0
            generate_train_set(**kwd)

        with self.assertRaises(AssertionError):
            kwd = init_kwd()
            kwd['num_of_features'] = -1
            generate_train_set(**kwd)

        with self.assertRaises(AssertionError):
            kwd = init_kwd()
            kwd['w'] = 'some str'
            generate_train_set(**kwd)

        with self.assertRaises(AssertionError):
            kwd = init_kwd()
            kwd['w'] = np.zeros(kwd['num_of_features'] + 1)
            generate_train_set(**kwd)

        kwd = init_kwd()
        x, y = generate_train_set(**kwd)
        self.assertTrue(isinstance(x, np.ndarray) and isinstance(y, np.ndarray))
        self.assertTrue(x.shape == (kwd['sample_size'], kwd['num_of_features']) and
                        y.shape == (kwd['sample_size'], 1))


if __name__ == '__main__':
    unittest.main()