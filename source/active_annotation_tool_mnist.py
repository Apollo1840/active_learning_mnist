import tensorflow as tf
import threading
import random
from annotation_tool_mnist import AnnotationTool, cnn
from active_learning.active_learning_mnist import (least_margin_strategy)


class ActiveAnnotationTool(AnnotationTool):
    display_tmp = "Current batch: {ind}/{q_s}, total annotations: {n_ann}\ncurrent model accuracy: {acc:.2f}"

    def __init__(self, data, query_strategy, query_size=16):
        super().__init__(data, query_size)
        self.threads = [None, None]
        self.query_strategy = query_strategy

        self.is_preparing = False

        self.unlabeled_indices = list(range(len(self.images)))
        self.annotated_indices = []
        random.shuffle(self.unlabeled_indices)
        self.curr_batch = self.unlabeled_indices[:self.query_size]
        self.next_batch = []

        self.index = self.curr_batch.pop(0)

    def next_image(self):
        label = int(self.label_var.get())
        print(f"Image {self.index}: Label {label}")
        print("curr_batch: {}, next_batch: {}".format(len(self.curr_batch), len(self.next_batch)))

        self.annotated_indices.append(self.index)
        self.annotated_images.append(self.images[self.index])
        self.annotated_labels.append(label)

        ind = len(self.annotated_images) % self.query_size
        n_ann = len(self.annotated_images)
        self.model_accuracy.set(self.display_tmp.format(ind=ind, q_s=self.query_size, n_ann=n_ann, acc=self.acc_curr))

        if len(self.curr_batch) != 0:
            self.index = self.curr_batch.pop(0)
        elif len(self.next_batch) != 0:
            self.curr_batch = self.next_batch
            self.next_batch = []
            self.index = self.curr_batch.pop(0)
        else:
            self.index = None

        if self.index is not None:
            self.label_var.set(str(self.labels[self.index]))
            self.show_image()
        else:
            self.destroy()

        if len(self.annotated_images) % self.query_size == 0 and not self.is_training:
            t = threading.Thread(target=self.train_model)
            t.start()
            self.threads[0] = t

        if len(self.next_batch) == 0 and not self.is_training and not self.is_preparing:
            t = threading.Thread(target=self.prepare_next_batch)
            t.start()
            self.threads[1] = t

    def prepare_next_batch(self):
        self.is_preparing = True

        remaining_indices = [i for i in self.unlabeled_indices if i not in self.annotated_indices]
        predictions = self.model.predict(self.images[remaining_indices])
        next_batch_indices = self.query_strategy(remaining_indices, predictions)[:self.query_size]

        # Update next_batch with the new batch
        self.next_batch = next_batch_indices

        self.is_preparing = False


if __name__ == "__main__":
    data = tf.keras.datasets.mnist.load_data()
    (x_train, y_train), (x_test, y_test) = data
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
    data = x_train, y_train, x_test, y_test

    #annotation_tool = ActiveAnnotationTool(data, trivial_strategy)
    annotation_tool = ActiveAnnotationTool(data, least_margin_strategy)
    annotation_tool.connect(cnn())
    annotation_tool.mainloop()

    annotation_tool.monitor(name="test.png")
