import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})


def get_x(num_of_features):
    """
    Генерация вектора x - он используется в линейной модели y = x * w -
    в генерации обучающей выборки

    :return: numpy array of shape (1, num_features)
    """
    return 20. * np.random.rand(1, num_of_features) - 10.


def get_squared_error(x, y):
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("x, y must be of numpy array type! Given {}, {} types".format(
            type(x), type(y)
        ))
    if not (x.shape[0] == x.size) or not (y.shape[0] == y.size) or (x.shape[0] != y.shape[0]):
        raise ValueError('x, y must have the same dimensions! Given x: {}; y: {}'.format(
            x.shape[0], y.shape[0]
        ))

    return np.dot(x - y, x - y)


def generate_train_set(w, sigma, sample_size, num_of_features):
    """
    Генерация обучающей выборки

    :param w: numpy array(shape (num_of_features,)) of true weights of linear model y = x * w
    :param sigma: дисперсия ошибки(она накладывается на y)
    :param sample_size: размер выборки
    :param num_of_features: это число равно кол-ву столбцов в матрице x и размеру вектора w
    :return: feature matrix x of shape (sample_size, num_of_features) and targets y of shape (sample_size,1)
    """

    assert(1 <= num_of_features)
    assert(0 <= sigma)
    assert(1 <= sample_size)
    assert(isinstance(w, np.ndarray))
    assert(w.shape == (num_of_features,))


    y = np.array([])
    x = np.array([])

    for i in range(sample_size):
        if i == 0:
            x = get_x(num_of_features)
        else:
            x = np.vstack((x, get_x(num_of_features)))

        y_with_no_error = np.dot(x[i], w)
        error = sigma * (np.random.rand(1) - 0.5)
        y = np.append(y, y_with_no_error + error)

        #print("y_with_no_error = {}; error = {}".format(y_with_no_error, error))

    y = y.reshape((-1, 1))

    return x, y



def plot_error_and_convergence(errors, weights):
    fig = plt.figure(figsize=(20,20))
    '''
    ax1 = fig.add_subplot(211)
    ax1.plot(errors);
    plt.xlabel('Номер итерации алгоритма Качмажа')
    plt.ylabel('Квадратичная ошибка')
    plt.title('Квадратичная ошибка: (w - w_estimation)^T * (w - w_estimation)')
    '''
    ax2 = fig.add_subplot(212)
    ax2.plot(weights.T)
    plt.title('Как сходятся компоненты вектора w_estimation к w')
    plt.xlabel('Номер итерации')
    plt.ylabel('Численное значение компоненты вектора весов w')

def plot_true_weights(weights, true_weights):
    # Указываем истинные веса на графике: компоненты вектора w 
    for index, w_component in enumerate(np.asarray(true_weights).reshape((-1))):
        plt.axhline(w_component, color='r', linestyle='--', label='{}: {}-ая компонента вектора w'.format(
            w_component, index), alpha=0.5)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
