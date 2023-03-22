"""

This code is highly inspired by gtoubassi from github: https://github.com/gtoubassi/active-learning-mnist/blob/master/Active_Learning_with_MNIST.ipynb

"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from .model import mlp_classifier
from .active_learning_mnist import (trivial_strategy,
                                    max_entropy_strategy,
                                    least_margin_strategy,
                                    least_confidence_strategy)


def plot_prioritization_strategy(data, model, prioritizer):
    """

    :param model:
    :param prioritizer: lamdba: indices, predictions: sorted indices
    :return:
    """
    query_steps = 20
    query_size = 500

    x_train, y_train, x_test, y_test = data

    train_indices = range(10000)

    # First subset of data
    selected_indices = train_indices[0:query_size]
    x_train_subset = x_train[selected_indices, ...]
    y_train_subset = y_train[selected_indices, ...]

    for i in range(query_steps):
        selected_indices = train_indices[0:query_size]

        if i > 0:
            x_train_subset = np.concatenate((x_train_subset, x_train[selected_indices, ...]))
            y_train_subset = np.concatenate((y_train_subset, y_train[selected_indices, ...]))

        plot_train_set(x_train, x_train_subset)

        model.fit(x_train_subset, y_train_subset, epochs=5, verbose=0)

        # prepare for the next step of the loop
        rest_indices = train_indices[query_size:]
        predictions = model.predict(x_train[rest_indices, ...])

        train_indices = prioritizer(rest_indices, predictions)

    print("- finished -")


def plot_train_set(x_train, x_train_subset, x_train_dim2=None):
    # Apply t-SNE to x_train only
    tsne = TSNE(n_components=2, random_state=42)

    x_concat = np.concatenate((x_train, x_train_subset), axis=0)

    x_train_tsne = tsne.fit_transform(x_train)
    x_concat_tsne = tsne.fit_transform(x_concat)
    # Apply the trained t-SNE model to both datasets
    x_train_subset_tsne = tsne.transform(x_train_subset)

    # Create a scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x_train_tsne[:, 0], x_train_tsne[:, 1], c='b', s=20)
    ax.scatter(x_train_subset_tsne[:, 0], x_train_subset_tsne[:, 1], c='orange', s=5)

    # Set axis labels and title
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('t-SNE Plot of x_train and x_train_subset')
    plt.show()



if __name__ == '__main__':
    # prepare data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    data = x_train, y_train, x_test, y_test

    # get results of different AL method
    acc_baseline = plot_prioritization_strategy(data, mlp_classifier(), trivial_strategy)
    acc_entropy = plot_prioritization_strategy(data, mlp_classifier(), max_entropy_strategy)
    acc_bt = plot_prioritization_strategy(data, mlp_classifier(), least_margin_strategy)
    acc_vr = plot_prioritization_strategy(data, mlp_classifier(), least_confidence_strategy)

    # visualize the performance difference
    plt.plot(acc_baseline, 'k', label='baseline')
    plt.plot(acc_entropy, 'b', label='least confidence')
    plt.plot(acc_bt, 'g', label='highest entropy')
    plt.plot(acc_vr, 'r', label='least margin')
    plt.legend()
