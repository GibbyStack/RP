import numpy as np

class CAP():

    def __init__(self):
        self.m = []
        self.mc = []

    # Método para entrenar el clasificador
    def fit(self, x_train, y_train):
        self.m = np.mean(x_train, axis=0)
        x_train = x_train - self.m
        n = len(x_train[0])
        self.mc = [[0 for _ in range(n)] for _ in range(2)]
        for i in range(len(x_train)):
            x = x_train[i]
            if y_train[i] == 0:
                self.mc[0] += x * 1
            else:
                self.mc[1] += x * 1

    # Método para obtener predicciones
    def predict(self, x_test):
        x_test = x_test - self.m
        predictions = []
        for x in x_test:
            x_mc = self.mc @ x
            prediction = np.argmax(x_mc)
            predictions.append(prediction)
        predictions = np.array(predictions)
        return predictions

from sklearn import datasets
from sklearn.model_selection import train_test_split
# from sklearn.datasets import fetch_openml

dataset = datasets.load_iris()
data = dataset.data # Datos del dataset
classes = dataset.target # Clases
targets = dataset.target_names # Etiqueta de clase
labels = dataset.feature_names # Etiquetas de los atributos