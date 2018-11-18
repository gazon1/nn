import numpy as np


def kaczmarz_alg(x, y, N):
    """

    :param x:
    :param y:
    :param N: num of features
    :return: yield updated weights
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError
    assert(x.shape[1] == N and x.shape[0] == y.shape[0])
    # if x.shape != y.shape:
    #     raise ValueError

    w = np.random.randn(N)
    for x_sample, y_sample in zip(x, y):
        w = w + (y_sample - np.dot(x_sample, w)) / np.dot(x_sample, x_sample) * x_sample
        yield w