import unittest
from kachmarz import kaczmarz_alg
import numpy as np
from utils import get_squared_error, generate_train_set


class KachmarzTest(unittest.TestCase):

    # def test_arguments(self):
        # with self.assertRaises(TypeError):
        #     list(kaczmarz_alg('wrong arg', np.array([]), 1))
        # with self.assertRaises(ValueError):
        #     list(kaczmarz_alg(np.array([1, 2]), np.array([1, 2, 3]), 1))

    def test_shapes(self):
        # TODO: написать тест для проверки размера выходных и входных даннныъ
        pass

    def test_squared_error_decreasing(self):
        sample_size = 100  # размер выборки
        num_of_features = 2 # кол-во признаков

        w = np.random.rand(num_of_features)  # это будет истинным значением памяти модели y = w * x + epsilon
        x, y = generate_train_set(w, 0, sample_size, num_of_features)

        gen = kaczmarz_alg(x, y, 0.6)
        print(x.shape, y.shape)
        w_estiomation_prev_step = next(gen)


        squared_error_prev_step = get_squared_error(w_estiomation_prev_step, w)
        for w_estimation in gen:
            squared_error_current_step = get_squared_error(w_estimation, w)
            print("w = {}; w_estimation_current_step = {}; w_est_prev_step={}; error current step = {}; error prev step = {}".format(
                w, w_estimation, w_estiomation_prev_step, squared_error_current_step, squared_error_prev_step))
            self.assertTrue(squared_error_current_step <= squared_error_prev_step)

            w_estiomation_prev_step = w_estimation
            squared_error_prev_step = squared_error_current_step


if __name__ == '__main__':
    unittest.main()
