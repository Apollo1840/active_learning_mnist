import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from model import mlp_classifier
from active_learning_mnist import (trivial_strategy,
                                   max_entropy_strategy,
                                   least_margin_strategy,
                                   least_confidence_strategy)


def plot_prioritization_strategy(data, model, prioritizer):
    """

    :param model:
    :param prioritizer: lamdba: indices, predictions: sorted indices
    :return:
    """
    query_steps = 100
    query_size = 100

    x_train, y_train, x_test, y_test = data

    train_indices = range(query_steps * query_size)

    tsne = TSNE(n_components=2, random_state=42)
    print("operating t-SNE ...")
    x_tsne = tsne.fit_transform(x_train[train_indices, ...].reshape((len(train_indices), -1)))
    print("t-SNE done")

    # First subset of data
    subset_indices = []
    for i in range(query_steps // 2):
        selected_indices = train_indices[0:query_size]
        subset_indices += selected_indices

        x_train_subset = x_train[subset_indices, ...]
        y_train_subset = y_train[subset_indices, ...]

        if i % 5 == 0:
            print("> selected {} data ({:.2f}%)".format(len(subset_indices),
                                                        len(subset_indices) / (query_size * query_steps)))
            plot_train_set(x_tsne, subset_indices)

        model.fit(x_train_subset, y_train_subset, epochs=5, verbose=0)

        # prepare for the next step of the loop
        rest_indices = train_indices[query_size:]
        predictions = model.predict(x_train[rest_indices, ...])

        train_indices = prioritizer(rest_indices, predictions)

    print("- finished -")


def plot_train_set(x_tsne, selected_indices):
    # Separate the transformed arrays back into their original parts
    x_subset_tsne = x_tsne[selected_indices, :]

    # Create a scatter plot using plt.plot
    plt.figure(figsize=(8, 8))
    plt.plot(x_tsne[:, 0], x_tsne[:, 1], 'bo', markersize=10, label='x_train', markeredgecolor="none")
    plt.plot(x_subset_tsne[:, 0], x_subset_tsne[:, 1], 'o', color='orange', markersize=2,
             label='x_train_subset')

    # Set axis labels and title
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE Plot of x_train and x_train_subset')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # prepare data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    data = x_train, y_train, x_test, y_test

    # get results of different AL method
    plot_prioritization_strategy(data, mlp_classifier(), trivial_strategy)
    plot_prioritization_strategy(data, mlp_classifier(), max_entropy_strategy)
    plot_prioritization_strategy(data, mlp_classifier(), least_margin_strategy)
    plot_prioritization_strategy(data, mlp_classifier(), least_confidence_strategy)
