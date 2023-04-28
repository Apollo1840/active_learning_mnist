"""

The predictions are now: Shape: (N, K, C), probabilities of data points (N) with samples (K) and classes (C).


"""

import numpy as np
import tensorflow as tf

from source.dl_model.mlp import mlp_classifier
import matplotlib.pyplot as plt

from .BALD import get_bald_batch
from .batchBALD import get_batchbald_batch


def dropout_weights(weights, ratio):
    dropout_mask = [np.random.binomial(1, 1 - ratio, w.shape) for w in weights]
    return [w * mask for w, mask in zip(weights, dropout_mask)]


def eval_prioritization_strategy(data, model, prioritizer, verbose=True):
    """

    :param model:
    :param prioritizer: lamdba: indices, predictions: sorted indices, ie. query strategy
    :return:
    """

    query_steps = 20
    query_size = 16

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

        # copy the models 5 times, all weights of the model is dropout with a ratio of 0.2:
        predictions = []
        for i in range(5):
            mc_model = tf.keras.models.clone_model(model)
            dropped_out_weights = dropout_weights(mc_model.get_weights(), 0.2)
            mc_model.set_weights(dropped_out_weights)
            predictions.append(mc_model.predict(x_train[rest_indices, ...]))

        predictions = np.transpose(predictions, (1, 0, 2))  # return (N_samples, K_models, C_classes)
        train_indices = prioritizer(rest_indices, predictions, query_size)

    print("- finished -")
    return test_accuracies


trivial_strategy = lambda indices, pred, batch_size: indices


def bald_strategy(indices, predictions, batch_size):
    scores, indices2 = get_bald_batch(predictions, batch_size)
    return [indices[i] for i in indices2]


def batchbald_strategy(indices, predictions, batch_size):
    scores, indices2 = get_batchbald_batch(predictions, batch_size)
    return [indices[i] for i in indices2]


if __name__ == '__main__':
    # prepare data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    data = x_train, y_train, x_test, y_test

    # get results of different AL method
    acc_baseline = eval_prioritization_strategy(data, mlp_classifier(), trivial_strategy)
    acc_bald = eval_prioritization_strategy(data, mlp_classifier(), bald_strategy)
    acc_batchbald = eval_prioritization_strategy(data, mlp_classifier(), batchbald_strategy)

    # visualize the performance difference
    plt.plot(acc_baseline, 'k', label='baseline')
    plt.plot(acc_bald, 'b', label='bald')
    plt.plot(acc_batchbald, 'g', label='batchbald')
    plt.legend()
