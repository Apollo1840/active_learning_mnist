import random
import numpy as np
import threading

from annotation_tool_ecg import AnnotationTool
from active_learning.active_learning_mnist import (least_margin_strategy)

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


class ActiveAnnotationTool(AnnotationTool):
    def __init__(self, train_gen, x_test, y_test, labels, query_strategy, query_factor=2):
        super().__init__(train_gen, x_test, y_test, labels, query_factor)
        self.threads = [None, None]

        self.query_strategy = query_strategy

        # unlabeled indices and annotated indices is global, meaning according to self.train_gen
        self.unlabeled_indices = list(range(len(train_gen) * self.train_gen.batch_size))
        self.annotated_indices = []
        random.shuffle(self.unlabeled_indices)

        self.curr_query_batch = self.gen_iterator
        self.curr_query_batch_indices = list(range(self.query_size))
        self.next_query_batch = None
        self.query_index = 0

        self.is_preparing = False

    def prepare_next_batch(self):
        # prepare next query batch
        self.is_preparing = True

        train_gen_unl = UnlabeledDataLoader(self.train_gen, self.unlabeled_indices, self.annotated_indices)
        indices_unl = train_gen_unl.unlabeled_indices

        print("preparing the next_batch, observed {} data in the train_gen, unlabeled: {}, labeled: {}".format(
            len(self.train_gen) * self.train_gen.batch_size, len(self.unlabeled_indices), len(self.annotated_indices)
        ))

        predictions = []
        for batch in train_gen_unl:
            x_batch = batch[0].reshape((len(batch[0]), batch[0].shape[-1], 1))
            batch_predictions = self.model.predict(x_batch)
            predictions.extend(batch_predictions)
        predictions = np.array(predictions)

        next_query_batch_indices = self.query_strategy(indices_unl, predictions)[:self.query_size]

        self.next_query_batch = UnlabeledDataLoader(self.train_gen, next_query_batch_indices)

        self.is_preparing = False

    def next_image(self):
        label = self.labels.index(self.label_var.get())
        print(f"Image {self.index}: Label {label}")
        print("curr_batch: {}, next_batch: {}".format(len(self.curr_query_batch_indices) - self.query_index,
                                                      len(self.next_query_batch.indices) if self.next_query_batch else 0))

        self.annotated_images.append(self.current_image)
        self.annotated_labels.append(label)

        # Add the annotated index to the annotated_indices list
        self.annotated_indices.append(self.curr_query_batch_indices[self.query_index])

        ind = len(self.annotated_images) % self.query_size
        n_ann = len(self.annotated_images)
        self.display_information.set(
            self.display_tmp.format(ind=ind, q_s=self.query_size, n_ann=n_ann, acc=self.acc_curr))

        self.index += 1
        self.batch_index += 1
        self.query_index += 1

        if self.query_index == self.query_size:
            self.curr_query_batch = iter(self.next_query_batch)
            self.curr_query_batch_indices = self.next_query_batch.indices
            self.next_query_batch = None
            self.query_index = 0

        # If we have finished annotating the current batch, load the next batch
        if self.batch_index >= len(self.current_batch[0]):
            try:
                self.current_batch = next(self.curr_query_batch)
                self.batch_index = 0
            except StopIteration:
                self.destroy()
                return

        if self.index < len(self.train_gen) * self.train_gen.batch_size:
            self.label_var.set(str(self.current_label))
            self.show_image()
        else:
            self.destroy()

        if len(self.annotated_images) % self.query_size == 0 and not self.is_training:
            t = threading.Thread(target=self.train_model)
            t.start()
            self.threads[0] = t

        # prepare the next_query_batch if it is not prepared and model is not training or predicting.
        if self.next_query_batch is None and not self.is_training and not self.is_preparing:
            t = threading.Thread(target=self.prepare_next_batch)
            t.start()
            self.threads[1] = t


if __name__ == '__main__':
    es = ECGStream.from_(CSN2022)
    es = es.shuffle()
    num_max = (len(es))
    es1000 = es.sub_es(range(min(1000, num_max)))
    es = es.shuffle()
    es100 = es.sub_es(range(min(100, num_max)))

    label_encoder = LabelEncoder(CSN2022_LABELS)
    func_ecg2x = lambda ecg: ecg.data.T[:, 0]
    func_ecg2y = lambda ecg: label_encoder.label2y_single(ecg.label, onehot=False) if ecg.label else 0
    digitizer = ESDigitizer(16, func_ecg2x, func_ecg2y)
    train_gen = digitizer.digitize_es(es1000)
    test_gen = digitizer.digitize_es(es100)
    x_test, y_test_num = test_gen.to_xy()

    x_test = x_test.reshape((len(x_test), x_test.shape[-1], 1))
    y_test = np.zeros((len(y_test_num), len(CSN2022_LABELS)))
    for i in range(len(y_test)):
        y_test[i, int(y_test_num[i])] = 1

    annotation_tool = ActiveAnnotationTool(train_gen, x_test, y_test, CSN2022_LABELS, query_strategy=least_margin_strategy)
    annotation_tool.connect(KerasRobustCNN3(es1000[0].data.shape[1], 1, n_classes=len(CSN2022_LABELS)))
    annotation_tool.mainloop()

    annotation_tool.monitor()
