import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import threading
import matplotlib.pyplot as plt

from ecgAI.constants import *
from ecgAI import ECGStream
from ecgAI.ml import LabelEncoder
from ecgAI.ml.algrm import ESDigitizer
from ecgAI.ml.models._bank.keras_model.ecgclf.cnn import KerasRobustCNN3


class AnnotationTool(tk.Tk):
    TITLE = "Annotation Tool"
    display_tmp = "Current batch: {ind}/{q_s}, total annotations: {n_ann}\nCurrent model accuracy: {acc:.2f}"

    def __init__(self, train_gen, x_test, y_test, labels, query_factor=1):
        super().__init__()

        self.train_gen = train_gen
        self.x_test, self.y_test = x_test, y_test
        self.labels = labels
        self.annotated_images = []
        self.annotated_labels = []

        self.query_size = query_factor * train_gen.batch_size

        self.model = None  # wait to be assigned
        self.acc_curr = 0
        self.accs = []
        self.index = 0
        self.is_training = False
        self.threads = [None]

        # Initialize the generator
        self.gen_iterator = iter(train_gen)
        self.batch_index = 0
        self.current_batch = next(self.gen_iterator)

        # construction the GUI
        self.title(self.TITLE)
        self.figure = Figure(figsize=(20, 5), dpi=200)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        self.label_var = tk.StringVar()
        self.label_var.set(str(self.current_label))

        self.label_entry = ttk.Combobox(self, textvariable=self.label_var)
        self.label_entry['values'] = list(labels)
        self.label_entry.pack()

        self.ok_button = ttk.Button(self, text="OK", command=self.next_image)
        self.ok_button.pack()

        self.display_information = tk.StringVar()
        self.display_information.set(self.display_tmp.format(ind=0, q_s=self.query_size, n_ann=0, acc=self.acc_curr))
        self.accuracy_label = ttk.Label(self, textvariable=self.display_information)
        self.accuracy_label.pack(anchor="ne")

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.show_image()

    @property
    def current_image(self):
        return self.current_batch[0][self.batch_index]

    @property
    def current_label(self):
        return self.labels[int(self.current_batch[1][self.batch_index])]

    def connect(self, model):
        self.model = model

    def show_image(self):
        img = self.current_image
        self.ax.clear()
        self.ax.plot(img)
        self.ax.set_title(f"Image {self.index}")
        self.canvas.draw()

    def next_image(self):
        label = self.labels.index(self.label_var.get())
        print(f"Image {self.index}: Label {label}")

        self.annotated_images.append(self.current_image)
        self.annotated_labels.append(label)

        ind = len(self.annotated_images) % self.query_size
        n_ann = len(self.annotated_images)
        self.display_information.set(
            self.display_tmp.format(ind=ind, q_s=self.query_size, n_ann=n_ann, acc=self.acc_curr))

        self.index += 1  # record number of annotated images
        self.batch_index += 1

        # If we have finished annotating the current batch, load the next batch
        if self.batch_index >= len(self.current_batch[0]):
            try:
                self.current_batch = next(self.gen_iterator)
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

    def train_model(self):
        self.is_training = True

        self.display_information.set("Model is fitting, please wait...")
        self.update()

        x_train = np.array(self.annotated_images)
        x_train = x_train.reshape((len(x_train), x_train.shape[-1], 1))
        y_train = np.zeros((len(x_train), len(self.labels)))
        for i in range(len(self.annotated_labels)):
            y_train[i, self.annotated_labels[i]] = 1

        self.model.fit(x_train, y_train, epochs=5, verbose=0)
        evaluation = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        self.acc_curr = evaluation[1]  # Assuming the accuracy is the second value in the list
        self.accs.append(self.acc_curr)

        ind = len(self.annotated_images) % self.query_size
        n_ann = len(self.annotated_images)
        self.display_information.set(
            self.display_tmp.format(ind=ind, q_s=self.query_size, n_ann=n_ann, acc=self.acc_curr))

        self.is_training = False

    def on_close(self):
        print("Annotation tool closed.")
        self.save()
        self.destroy()

    def save(self):
        # save the data and save the model
        pass

    def monitor(self, name="accs.png"):
        for t in self.threads:
            t.join()
        plt.plot(range(0, len(self.accs) * self.query_size, self.query_size), self.accs)
        plt.show()
        plt.savefig(name)


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

    annotation_tool = AnnotationTool(train_gen, x_test, y_test, CSN2022_LABELS)
    annotation_tool.connect(KerasRobustCNN3(es1000[0].data.shape[1], 1, n_classes=len(CSN2022_LABELS)))
    annotation_tool.mainloop()

    annotation_tool.monitor()
