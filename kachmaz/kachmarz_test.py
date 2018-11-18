import unittest
from kachmarz import kaczmarz_alg
import numpy as np


class KachmarzTest(unittest.TestCase):

    def test_arguments(self):
        with self.assertRaises(TypeError):
            list(kaczmarz_alg('wrong arg', np.array([])))
        with self.assertRaises(ValueError):
            list(kaczmarz_alg(np.array([1, 2]), np.array([1, 2, 3])))

    def test_squared_error_decreasing(self):
        w = np.random.rand(1)  # это будет истинным значением памяти модели y = w * x + epsilon

        y = np.array([])
        x = np.array([])

        sigma = 0.1  # дисперсия ошибки(она гаусова)
        n = 100  # размер выборки
        for i in range(n):
            x = np.append(x, np.random.rand(1))
            y = np.append(y, x[i] * w + np.random.randn(1) * sigma)

        gen = kaczmarz_alg(x, y)
        w_estiomation_prev_step = next(gen)
        squared_error_prev_step = (w_estiomation_prev_step - w) ** 2
        for w_estimation in gen:
            squared_error_current_step = (w_estimation - w) ** 2
            print("w = {}; w_estimation_current_step = {}; error current step = {}; error prev step = {}".format(
                w, w_estimation, squared_error_current_step, squared_error_prev_step))
            self.assertTrue(squared_error_current_step <= squared_error_prev_step)

            w_estiomation_prev_step = w_estimation
            squared_error_prev_step = squared_error_current_step


if __name__ == '__main__':
    unittest.main()