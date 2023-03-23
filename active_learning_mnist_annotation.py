import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tensorflow as tf


class AnnotationTool(tk.Tk):
    def __init__(self, images, labels):
        super().__init__()

        self.images, self.labels = images, labels

        self.title("MNIST Annotation Tool")
        self.index = 0

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

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.show_image()

    def show_image(self):
        img = self.images[self.index]
        self.ax.imshow(img, cmap="gray")
        self.ax.set_title(f"Image {self.index}")
        self.canvas.draw()

    def next_image(self):
        label = int(self.label_var.get())
        print(f"Image {self.index}: Label {label}")
        self.index += 1
        if self.index < len(self.images):
            # Update the default value based on the label of the next image
            self.label_var.set(str(self.labels[self.index]))
            self.show_image()
            self.show_image()
        else:
            self.destroy()

    def on_close(self):
        print("Annotation tool closed.")
        self.destroy()


if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    annotation_tool = AnnotationTool(x_train, y_train)
    annotation_tool.mainloop()
