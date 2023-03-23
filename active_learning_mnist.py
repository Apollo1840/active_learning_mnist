"""

This code is highly inspired by gtoubassi from github: https://github.com/gtoubassi/active-learning-mnist/blob/master/Active_Learning_with_MNIST.ipynb

"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import mlp_classifier
import matplotlib.pyplot as plt


def eval_prioritization_strategy(data, model, prioritizer, verbose=True):
    """

    :param model:
    :param prioritizer: lamdba: indices, predictions: sorted indices
    :return:
    """
    query_steps = 40
    query_size = 500

    x_train, y_train, x_test, y_test = data

    train_indices = range(60000)

    # First subset of data
    selected_indices = train_indices[0:query_size]
    x_train_subset = x_train[selected_indices, ...]
    y_train_subset = y_train[selected_indices, ...]

    test_accuracies = []
    for i in range(query_steps):
        selected_indices = train_indices[0:query_size]

        if i > 0:
            x_train_subset = np.concatenate((x_train_subset, x_train[selected_indices, ...]))
            y_train_subset = np.concatenate((y_train_subset, y_train[selected_indices, ...]))

        model.fit(x_train_subset, y_train_subset, epochs=5, verbose=0)

        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        test_accuracies.append(accuracy)

        if verbose:
            print('Training data size of %d => accuracy %f' % (x_train_subset.shape[0], accuracy))

        # prepare for the next step of the loop
        rest_indices = train_indices[query_size:]
        predictions = model.predict(x_train[rest_indices, ...])

        train_indices = prioritizer(rest_indices, predictions)

    print("- finished -")
    return test_accuracies


trivial_strategy = lambda indices, pred: indices


def max_entropy_strategy(indices, predictions):
    p = predictions * np.log(predictions)
    p = -p.sum(axis=1)
    p = list(zip(indices, p))
    p.sort(reverse=True, key=lambda x: x[1])  # sort in descending order
    return list(list(zip(*p))[0])


def least_margin_strategy(indices, predictions):
    # breaking tie
    p = -np.sort(-predictions)  # sort in descending order
    p = p[:, 0] - p[:, 1]
    p = list(zip(indices, p))
    p.sort(key=lambda x: x[1])  # sort in ascending order
    return list(list(zip(*p))[0])


def least_confidence_strategy(indices, predictions):
    # variation ratio
    max_logit = list(zip(indices, np.amax(predictions, axis=1)))
    max_logit.sort(key=lambda x: x[1])  # sort in ascending order
    return list(list(zip(*max_logit))[0])


if __name__ == '__main__':
    # prepare data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    data = x_train, y_train, x_test, y_test

    # get results of different AL method
    acc_baseline = eval_prioritization_strategy(data, mlp_classifier(), trivial_strategy)
    acc_entropy = eval_prioritization_strategy(data, mlp_classifier(), max_entropy_strategy)
    acc_bt = eval_prioritization_strategy(data, mlp_classifier(), least_margin_strategy)
    acc_vr = eval_prioritization_strategy(data, mlp_classifier(), least_confidence_strategy)

    # visualize the performance difference
    plt.plot(acc_baseline, 'k', label='baseline')
    plt.plot(acc_vr, 'b', label='least confidence')
    plt.plot(acc_entropy, 'g', label='highest entropy')
    plt.plot(acc_bt, 'r', label='least margin')
    plt.legend()
