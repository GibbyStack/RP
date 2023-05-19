# from sklearn.model_selection import KFold
# from Performance import *
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def train_test_by_class(x_train, x_test, y_train, y_test):
    classes = list(set(y_train))
    index_train, index_test = np.array([], dtype=int), np.array([], dtype=int)
    for c in classes:
        idx_train = np.where(y_train == c)[0]
        idx_test = np.where(y_test == c)[0]
        index_train = np.concatenate((index_train, idx_train))
        index_test = np.concatenate((index_test, idx_test))
    x_train, y_train = x_train[index_train], y_train[index_train]
    x_test, y_test = x_test[index_test], y_test[index_test]
    return x_train, x_test, y_train, y_test

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

# import matplotlib.pyplot as plt
# plt.scatter(data[:, 0], data[:, 3], c=classes, cmap='Accent')
# classes = np.concatenate((np.zeros(50, dtype=int), np.ones(100, dtype=int)))
X_train, X_test, Y_train, Y_test = train_test_split(data, classes, train_size=0.5, shuffle=True)

def search_index(classes, l_classes):
    index = []
    for i in range(len(classes)):
        if classes[i] in l_classes:
            index.append(i)
    return np.array(index)
    

Y_predicted = np.full(len(Y_test), None)

Y_train1 = Y_train.copy()
index_p = np.where(Y_train == 0)[0]
index_n = np.where(Y_train != 0)[0]
Y_train1[index_p] = 0
Y_train1[index_n] = 1
cap = CAP()
cap.fit(X_train, Y_train1)
predictions = cap.predict(X_test)
index_y1 = np.where(predictions == 0)[0]
Y_predicted[index_y1] = 0



l_classes = [1, 2]
index_l = search_index(Y_train, l_classes)
X_train2 = X_train[index_l].copy()
Y_train2 = Y_train[index_l].copy()
index_p = np.where(Y_train2 == 1)[0]
index_n = np.where(Y_train2 == 2)[0]
Y_train2[index_p] = 0
Y_train2[index_n] = 1
cap = CAP()
cap.fit(X_train2, Y_train2)
predictions = cap.predict(X_test)
index_y2 = np.where(predictions == 0)[0]



d1 = set(index_y1).difference(set(index_y2))
d2 = set(index_y2).difference(set(index_y1))
new_index = np.array(list(d1.union(d2)))
Y_predicted[new_index] = 1

 

index_y3 = np.where(Y_predicted == None)[0]
Y_predicted[index_y3] = 2
Y_predicted.astype(int)
Y_predicted
Y_test



correct = 0
for i in range(len(Y_test)):
    if Y_test[i] == Y_predicted[i]:
        correct += 1

acc = correct / len(Y_test)
acc