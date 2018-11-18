import numpy as np


def kaczmarz_alg(x, y):
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError
    if x.shape != y.shape:
        raise ValueError

    w = np.random.randn(1)
    for x_sample, y_sample in zip(x, y):
        w += w + (y_sample - x_sample * w) / (x_sample * x_sample) * x_sample
        yield w