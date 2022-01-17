import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import PIL
import sys
from PyQt6 import QtCore, QtGui, QtWidgets, uic

model = tf.keras.models.load_model('modelCIFAR.h5')
classes = {0: 'airplane',
           1: 'automobile',
           2: 'bird',
           3: 'cat',
           4: 'deer',
           5: 'dog',
           6: 'frog',
           7: 'horse',
           8: 'ship',
           9: 'truck'}

# Функция для предикта НС
def predic(path):
    imag = np.array(img_to_array(load_img(path)).astype('uint8'))
    image = PIL.Image.fromarray(imag, 'RGB')
    image = image.resize((32, 32))
    image = np.array(image)
    image = np.array([image])
    return classes[np.argmax(model.predict(image / 255))]

# Переменная для нажатия на кнопку без картинки
files = None

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("app3.ui", self)
        self.addFunc()

        # Виджет для вывода распознанного класса
        self.label_4 = QtWidgets.QLabel(self)
        self.label_4.setGeometry(QtCore.QRect(0, 200, 300, 50))
        self.label_4.setStyleSheet("font: 10pt \"Arial\";")
        self.label_4.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_4.setObjectName("label4")


    # Функция получения пути картинки
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    # Функция
    def dropEvent(self, event):
        global files
        files = [u.toLocalFile() for u in event.mimeData().urls()]

        # Виджет для вывода картинки на экран
        label_3 = QtWidgets.QLabel(self.centralwidget)
        label_3.setGeometry(QtCore.QRect(0, 0, 300, 200))
        label_3.setText("")
        label_3.setPixmap(QtGui.QPixmap(files[0]).scaled(300, 200))
        label_3.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        label_3.setObjectName("label_3")
        label_3.show()

    # Функция для нажатия на кнопку
    def addFunc(self):
        self.pushButton.clicked.connect(self.press)

    # Функция для выполнения действия при нажатии на кнопку
    def press(self):
        # Проверка для нажатия на кнопку без картинки
        if files != None:
            self.label_4.setText(f'Распознан класс {predic(files[0])}')


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
