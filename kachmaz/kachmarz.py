import numpy as np


def kaczmarz_alg(x, y, gamma=1.):
    """
    kaczmarz_alg make a step of kaczmaz algorithm for updating weights(w) of linear model: y = x.T * w
    num of iteration of kaczmarz algorithm equals to the size of thaining set

    :param x: numpy array of shape (sample_size, ...), where sample size is
        num of examples in training set
    :param y: numpy array of shape (sample_size, 1)
    :param gamma: multiplier for slower of quicker convergence
    :return: yield updated weights
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError
    assert (x.shape[0] == y.shape[0])

    # Выбираем первое приближение для оценки вектора весов w
    w = np.zeros(x.shape[1])

    for x_sample, y_sample in zip(x, y):
        print(x_sample, y_sample)
        w = w + gamma * x_sample * (y_sample - np.dot(x_sample, w)) / np.dot(x_sample, x_sample)
        yield w