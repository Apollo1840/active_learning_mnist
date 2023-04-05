from .active_learning_mnist import *

from ecgAI.constants import *
from ecgAI import ECGStream
from ecgAI.ml import LabelEncoder
from ecgAI.ml.algrm import ESDigitizer
from ecgAI.ml.models._bank.keras_model.ecgclf.cnn import KerasRobustCNN3


class UnlabeledDataLoader:
    def __init__(self, data_loader, indices, annotated_indices=None):
        self.data_loader = data_loader
        self.indices = indices
        self.annotated_indices = set(annotated_indices) if annotated_indices is not None else set()
        self.unlabeled_indices = [i for i in self.indices if i not in self.annotated_indices]
        self.position = 0

    def __iter__(self):
        self.position = 0
        return self

    def __next__(self):
        if self.position >= len(self.indices):
            raise StopIteration

        current_indices = []
        while len(current_indices) < self.data_loader.batch_size and self.position < len(self.indices):

            # sorted
            index = self.indices[self.position]

            # & unlabeled
            if index not in self.annotated_indices:
                current_indices.append(index)

            self.position += 1

        if not current_indices:
            raise StopIteration

        # Sort the train_gen by provided indices
        ml_ecgbatches_sorted_unlabeled = self.data_loader.ml_ecgbatchs.sub_es(current_indices)
        curr_data_loader = self.data_loader.remake_on(ml_ecgbatches_sorted_unlabeled)

        return curr_data_loader[0]

    def __len__(self):
        return len(self.indices)


def eval_prioritization_strategy(data, model, prioritizer, verbose=True):
    """
    :param model:
    :param prioritizer: lambda: indices, predictions: sorted indices
    :return:
    """
    query_steps = 40
    query_size = 16

    train_gen, x_test, y_test = data

    # Get the initial training data
    train_indices = list(range(len(train_gen) * train_gen.batch_size))

    # Select the first query_size samples as the initial training set
    annotated_indices = train_indices[:query_size]
    train_gen_selected = train_gen.remake_on(train_gen.ml_ecgbatchs.sub_es(annotated_indices))

    test_accuracies = []
    for i in range(query_steps):

        # Train the model on the selected data
        model.fit(train_gen_selected, epochs=5, verbose=0)

        # Evaluate the model on the test set
        eval = model.evaluate(x_test, y_test, verbose=0)
        test_accuracies.append(eval[1])

        if verbose:
            print('Training data size of %d => accuracy %f' % (
                train_gen_selected.batch_size * train_gen_selected.n_batch, eval[1]))

        # Prepare for the next step of the loop

        # Create an UnlabeledDataLoader with the remaining indices
        unlabeled_data_loader = UnlabeledDataLoader(train_gen, train_indices, annotated_indices)

        # Make predictions on the unlabeled data
        predictions = model.predict(unlabeled_data_loader)

        # Apply the prioritizer to select the next batch of samples to annotate
        selected_indices = prioritizer(unlabeled_data_loader.unlabeled_indices, predictions)[:query_size]

        # Update the annotated_indices with the new selected_indices
        annotated_indices.extend(selected_indices)

        # Update the train_gen_selected with the updated annotated_indices
        train_gen_selected = train_gen.remake_on(train_gen.ml_ecgbatchs.sub_es(annotated_indices))

    print("- finished -")
    return test_accuracies


if __name__ == '__main__':
    # prepare data
    es = ECGStream.from_(CSN2022)
    es = es.shuffle()
    num_max = (len(es))
    es1000 = es.sub_es(range(min(1000, num_max)))
    es = es.shuffle()
    es100 = es.sub_es(range(min(100, num_max)))

    label_encoder = LabelEncoder(CSN2022_LABELS)
    func_ecg2x = lambda ecg: ecg.data.T[:, 0].reshape((-1, 1))
    func_ecg2y = lambda ecg: label_encoder.label2y_single(ecg.label, onehot=True) if ecg.label else np.zeros(len(CSN2022_LABELS))
    digitizer = ESDigitizer(16, func_ecg2x, func_ecg2y)
    train_gen = digitizer.digitize_es(es1000)
    test_gen = digitizer.digitize_es(es100)
    x_test, y_test = test_gen.to_xy()
    data = train_gen, x_test, y_test

    m = es1000[0].data.shape[1]
    n = len(CSN2022_LABELS)

    # get results of different AL method
    acc_baseline = eval_prioritization_strategy(data, KerasRobustCNN3(m, 1, n), trivial_strategy)
    acc_entropy = eval_prioritization_strategy(data, KerasRobustCNN3(m, 1, n), max_entropy_strategy)
    acc_bt = eval_prioritization_strategy(data, KerasRobustCNN3(m, 1, n), least_margin_strategy)
    acc_vr = eval_prioritization_strategy(data, KerasRobustCNN3(m, 1, n), least_confidence_strategy)

    # visualize the performance difference
    plt.plot(acc_baseline, 'k', label='baseline')
    plt.plot(acc_vr, 'b', label='least confidence')
    plt.plot(acc_entropy, 'g', label='highest entropy')
    plt.plot(acc_bt, 'r', label='least margin')
    plt.legend()
