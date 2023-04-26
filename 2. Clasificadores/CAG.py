# from sklearn.model_selection import KFold
# from Performance import *
import numpy as np
# from functools import reduce

def train_test_by_class(x_train, x_test, y_train, y_test):
    classes = list(set(y_train))
    index_train = np.where(y_train == classes[0])[0]
    index_test = np.where(y_test == classes[0])[0]
    for i in range(1, len(classes)):
        idx_train = np.where(y_train == classes[i])[0]
        idx_test = np.where(y_test == classes[i])[0]
        index_train = np.concatenate((index_train, idx_train))
        index_test = np.concatenate((index_test, idx_test))
    x_train, y_train = x_train[index_train], y_train[index_train]
    x_test, y_test = x_test[index_test], y_test[index_test]
    return x_train, x_test, y_train, y_test

class CAG():
    def __init__(self):
        self.m = []
        self.mc = []

    # Método para generar el vector medio
    def mean(self, x_train):
        self.m = np.mean(x_train, axis=0)

    # Método para normalizar los datos
    def standar(self, x):
        return x - self.m

    # Método para entrenar el clasificador
    def fit(self, x_train, y_train, calculate=True):
        if calculate: 
            self.mean(x_train)
            x_train = self.standar(x_train)
        n = len(x_train[0])
        self.mc.append([[0 for _ in range(n)] for _ in range(2)])
        for i in range(len(x_train)):
            x = x_train[i]
            if y_train[i] == 0:
                self.mc[-1][0] += x * 1
                self.mc[-1][1] += x * 0
            else:
                self.mc[-1][0] += x * 0
                self.mc[-1][1] += x * 1

    # Método para obtener predicciones
    def predict(self, x_test, pos=0, calculate=True):
        if calculate:
            x_test = self.standar(x_test)
        predictions = []
        for x in x_test:
            x_mc = self.mc[pos] @ x
            prediction = np.where(x_mc == np.max(x_mc))[0][0]
            predictions.append(prediction)
        predictions = np.array(predictions)
        return predictions

class MULTI_CAG(CAG):
    def __init__(self):
        super().__init__()

    # Método para entrenar el clasificador
    def fit(self, x_train, y_train):
        super().mean(x_train)
        x_train = super().standar(x_train)
        data_class = np.unique(y_train, return_counts=True)[1]
        min, max = 0, sum(data_class)
        for i in range(len(data_class)):
            # print(f' {i} '.center(25, '='))
            s1, s2 = sum(data_class[:i]), sum(data_class[:i+1])
            new_x_train = np.concatenate((x_train[s1: s2], np.concatenate((x_train[min:s1], x_train[s2:max]))))
            new_y_train = np.concatenate((np.zeros(s2-s1, dtype=int), np.ones(s1-min+max-s2, dtype=int)))
            super().fit(new_x_train, new_y_train, calculate=False)
    
    # Método para obtener prediccionees
    def predict(self, x_test):
        predictions = []
        x_test = super().standar(x_test)
        for i in range(len(self.mc)):
            p = super().predict(x_test, pos=i, calculate=False)
            predictions.append(p)
        predictions = np.array(predictions)
        return predictions
        


from sklearn import datasets
from sklearn.model_selection import train_test_split

dataset = datasets.load_iris()
data = dataset.data # Datos del dataset
classes = dataset.target # Clases
targets = dataset.target_names # Etiqueta de clase
labels = dataset.feature_names # Etiquetas de los atributos


# classes = np.concatenate((np.zeros(50, dtype=int), np.ones(100, dtype=int)))
# data = np.concatenate((data[100:], data[0:50], data[50:100]))
X_train, X_test, Y_train, Y_test = train_test_split(data, classes, train_size=0.5, shuffle=True)
X_train, X_test, Y_train, Y_test = train_test_by_class(X_train, X_test, Y_train, Y_test)
# classes = np.concatenate((np.zeros(50, dtype=int), np.ones(50, dtype=int)))
# cag = CAG()
# cag.fit(X_train, Y_train)
# predictions = cag.predict(X_test)
# predictions
# Y_test


cag = MULTI_CAG()
cag.fit(X_train, Y_train)
Y_predicted = cag.predict(X_test)
Y_test
Y_predicted

data_class = np.unique(Y_test, return_counts=True)
data_class

# dbc = data_by_class(classes)
# min, max = 0, 0
# Y_predicted = np.array([], dtype=int)
# for i in range(len(dbc) - 1):
#     max = np.sum(dbc[:i+2])
#     new_classes = classes[:max].copy()
#     new_data = data[:max].copy()
#     if i > 0:
#         new_classes[new_classes != i+1] = 0
#     X_train, X_test, Y_train, Y_test = train_test_split(new_data, new_classes, train_size=0.5, shuffle=True)
#     X_train, X_test, Y_train, Y_test = train_test_by_class(X_train, X_test, Y_train, Y_test)
#     cag = CAG()
#     cag.fit(X_train, Y_train)
#     predictions = cag.predict(X_test)
#     if i != 0:
#         predictions[predictions == 1] = i+1
#     Y_predicted = np.concatenate((Y_predicted, predictions[min:]))
#     min = len(Y_predicted)
# Y_predicted