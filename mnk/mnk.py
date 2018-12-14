import numpy as np


def mnk_alg(x, y):
    """
    mnk_alg make a step of kaczmaz algorithm for updating weights(w) of linear model: y = x.T * w
    num of iteration of mnk algorithm equals to the size of thaining set

    :param x: numpy array of shape (sample_size, ...), where sample size is
        num of examples in training set
    :param y: numpy array of shape (sample_size, 1)
    :return: yield updated weights
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError
    
    if not (x.shape[0] == y.shape[0]):
        raise ValueError("x.shape[0] = {}; y.shape[0] = {}".format(x.shape[0], y.shape[0]))

    # Выбираем первое приближение для оценки вектора весов w
    n = x.shape[1]
    w = np.zeros(n).reshape((4, 1))
    # ... для матрицы K ...
    K = np.eye(n)
    
    for x_sample, y_sample in zip(x, y):
        # вектор-столбцы
        x_sample = np.matrix(x_sample).T
        y_sample = np.matrix(y_sample).T
        
        K = K - (K * x_sample * x_sample.T * K)/(1. + x_sample.T * K * x_sample)
        K = np.matrix(K)
        
        A = y_sample - np.dot(w.T, x_sample)
        w = w + K * x_sample * A
        
        yield w.T
