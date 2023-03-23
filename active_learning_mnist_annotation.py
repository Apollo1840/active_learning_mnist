import tensorflow as tf
import threading
import random
from mnist_annotation import AnnotationTool


class ActiveAnnotationTool(AnnotationTool):
    display_tmp = "Current batch: {ind}/{s_q}, total annotations: {n_ann}\ncurrent model accuracy: {acc:.2f}"

    def __init__(self, data, query_strategy, query_size=16):
        super().__init__(data, query_size)
        self.query_strategy = query_strategy
        
        self.unlabeled_indices = list(range(len(self.images)))
        random.shuffle(self.unlabeled_indices)
        self.curr_batch = self.unlabeled_indices[:self.query_size]
        self.next_batch = []

        self.index = self.curr_batch.pop(0)

    def next_image(self):
        label = int(self.label_var.get())
        print(f"Image {self.index}: Label {label}")

        self.annotated_images.append(self.images[self.index])
        self.annotated_labels.append(label)

        ind = len(self.annotated_images) % self.query_size
        n_ann = len(self.annotated_images)
        self.model_accuracy.set(self.display_tmp.format(ind=ind, q_s=self.query_size, n_ann=n_ann, acc=self.acc_curr))

        if self.curr_batch:
            self.index = self.curr_batch.pop(0)
        elif not self.is_training and self.next_batch:
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
            threading.Thread(target=self.train_model).start()

        if not self.next_batch and not self.is_training:
            threading.Thread(target=self.prepare_next_batch).start()

    def prepare_next_batch(self):
        remaining_indices = [i for i in self.unlabeled_indices if i not in self.annotated_images]
        predictions = self.model.predict(self.images[remaining_indices])
        next_batch_indices = self.query_strategy(remaining_indices, predictions)[:self.query_size]

        # Update next_batch with the new batch
        self.next_batch = next_batch_indices


def cnn():
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    data = tf.keras.datasets.mnist.load_data()
    (x_train, y_train), (x_test, y_test) = data
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
    data = x_train, y_train, x_test, y_test

    trivial_strategy = lambda indices, pred: indices
    annotation_tool = AnnotationTool(data, trivial_strategy)
    annotation_tool.connect(cnn())
    annotation_tool.mainloop()
