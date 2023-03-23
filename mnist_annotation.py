import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tensorflow as tf
import numpy as np
import threading


class AnnotationTool(tk.Tk):
    display_tmp = "Current batch: {ind}/{q_s}, total annotations: {n_ann}\nCurrent model accuracy: {acc:.2f}"

    def __init__(self, data, query_size=16):
        super().__init__()

        self.images, self.labels, self.x_test, self.y_test = data
        self.annotated_images = []
        self.annotated_labels = []

        self.query_size = query_size

        self.model = None  # wait to be assigned
        self.acc_curr = 0
        self.index = 0
        self.is_training = False

        # construction the GUI
        self.title("MNIST Annotation Tool")

        self.figure = Figure(figsize=(5, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        self.label_var = tk.StringVar()
        self.label_var.set(str(self.labels[self.index]))

        self.label_entry = ttk.Entry(self, textvariable=self.label_var)
        self.label_entry.pack()

        self.ok_button = ttk.Button(self, text="OK", command=self.next_image)
        self.ok_button.pack()

        self.model_accuracy = tk.StringVar()
        self.model_accuracy.set(self.display_tmp.format(ind=0, q_s=self.query_size, n_ann=0, acc=self.acc_curr))
        self.accuracy_label = ttk.Label(self, textvariable=self.model_accuracy)
        self.accuracy_label.pack(anchor="ne")

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.show_image()

    def connect(self, model):
        self.model = model

    def show_image(self):
        img = self.images[self.index]
        img = img.reshape((28, 28))
        self.ax.imshow(img, cmap="gray")
        self.ax.set_title(f"Image {self.index}")
        self.canvas.draw()

    def next_image(self):
        label = int(self.label_var.get())
        print(f"Image {self.index}: Label {label}")

        self.annotated_images.append(self.images[self.index])
        self.annotated_labels.append(label)

        ind = len(self.annotated_images) % self.query_size
        n_ann = len(self.annotated_images)
        self.model_accuracy.set(self.display_tmp.format(ind=ind, q_s=self.query_size, n_ann=n_ann, acc=self.acc_curr))

        self.index += 1

        if self.index < len(self.images):
            self.label_var.set(str(self.labels[self.index]))
            self.show_image()
        else:
            self.destroy()

        if len(self.annotated_images) % self.query_size == 0 and not self.is_training:
            threading.Thread(target=self.train_model).start()

    def train_model(self):
        self.is_training = True

        self.model_accuracy.set("Model is fitting, please wait...")
        self.update()

        x_train = np.array(self.annotated_images)
        y_train = np.array(self.annotated_labels)

        self.model.fit(x_train, y_train, epochs=5, verbose=0)
        _, self.acc_curr = self.model.evaluate(self.x_test, self.y_test, verbose=0)

        ind = len(self.annotated_images) % self.query_size
        n_ann = len(self.annotated_images)
        self.model_accuracy.set(self.display_tmp.format(ind=ind, q_s=self.query_size, n_ann=n_ann, acc=self.acc_curr))

        self.is_training = False

    def on_close(self):
        print("Annotation tool closed.")
        self.destroy()


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

    annotation_tool = AnnotationTool(data)
    annotation_tool.connect(cnn())
    annotation_tool.mainloop()
