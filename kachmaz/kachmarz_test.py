import unittest
from kachmarz import kaczmarz_alg
import numpy as np


class KachmarzTest(unittest.TestCase):

    # def test_arguments(self):
        # with self.assertRaises(TypeError):
        #     list(kaczmarz_alg('wrong arg', np.array([]), 1))
        # with self.assertRaises(ValueError):
        #     list(kaczmarz_alg(np.array([1, 2]), np.array([1, 2, 3]), 1))

    def test_squared_error_decreasing(self):


        sigma = 0.1  # дисперсия ошибки(она гаусова)
        n = 100  # размер выборки
        N = 2 # кол-во признаков

        w = np.random.rand(N)  # это будет истинным значением памяти модели y = w * x + epsilon
        y = np.array([])
        x = np.random.rand(N)
        for i in range(n):
            y = np.append(y, np.dot(x[i], w)) # + np.random.randn(1) * sigma)
            x = np.vstack((x, np.random.rand(N)))

        y.reshape((-1, 1))

        gen = kaczmarz_alg(x, y, N)
        print(x.shape, y.shape)
        w_estiomation_prev_step = next(gen)

        get_squared_error = lambda x, y: np.dot(x - y, x - y)
        squared_error_prev_step = get_squared_error(w_estiomation_prev_step, w)
        for w_estimation in gen:
            squared_error_current_step = get_squared_error(w_estimation, w)
            print("w = {}; w_estimation_current_step = {}; error current step = {}; error prev step = {}".format(
                w, w_estimation, squared_error_current_step, squared_error_prev_step))
            self.assertTrue(squared_error_current_step <= squared_error_prev_step)

            w_estiomation_prev_step = w_estimation
            squared_error_prev_step = squared_error_current_step


if __name__ == '__main__':
    unittest.main()