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

class MULTI_CAP():

    def __init__(self):
        self.vm = []
        self.mc = []

    # Método para entrenar el clasificador
    def fit(self, x_train, y_train):
        n = len(x_train[0])
        cls_t = list(set(y_train))
        for i in range(len(cls_t)):
            index_p = np.where(y_train == cls_t[i])[0]
            print(f' {i} '.center(50, '='))
            print(index_p)
            x_p = np.mean(x_train[index_p], axis=0)
            x_n = []
            for j in range(len(cls_t)):
                if i != j:
                    index_n = np.where(y_train == cls_t[j])[0]
                    # print(index_n)
                    x_n.append(np.mean(x_train[index_n], axis=0))
            x_n = np.mean(x_n, axis=0)
            self.vm.append((x_p+x_n)/2)
            
            data_train = x_train - self.vm[i]
            self.mc.append([[0 for _ in range(n)] for _ in range(2)])
            for j in range(len(data_train)):
                x = data_train[j]
                if j in index_p:
                    # print(f' j = {j}, True')
                    self.mc[i][0] += x * 1
                else:
                    # print(f' j = {j}, False')
                    self.mc[i][1] += x * 1
    
    # Método para obtener prediccionees
    def predict(self, x_test):
        predictions = []
        for x in x_test:
            prediction = []
            for i in range(len(self.mc)):
                x_vm = x - self.vm[i]
                x_mc = self.mc[i] @ x_vm
                prediction.append(np.argmin(x_mc))
            predictions.append(prediction)
        predictions = np.array(predictions)
        return predictions
        # for i in range(len(self.mc)):
        #     predictions = []
        #     data_test = x_test - self.vm[i]
        #     for x in data_test:
        #         data_mc = self.mc[i] @ x
        #         prediction = np.argmin(data_mc)
        #         predictions.append(prediction)
        #     Y_predicted.append(predictions)
        # Y_predicted = np.array(Y_predicted)
        # return Y_predicted
        
from sklearn import datasets
from sklearn.model_selection import train_test_split
# from sklearn.datasets import fetch_openml

dataset = datasets.load_iris()
data = dataset.data # Datos del dataset
classes = dataset.target # Clases
targets = dataset.target_names # Etiqueta de clase
labels = dataset.feature_names # Etiquetas de los atributos

# dataset = fetch_openml(name='glass')
# data = np.array(dataset.data)
# targets = np.array(list(set(dataset.target)))
# classes = dataset.target
# for i in range(len(targets)):
#     classes = classes.replace({targets[i]: int(i)})
# classes = np.array(classes, dtype=int)
# labels = list(dataset.feature_names)

data = StandardScaler().fit_transform(data)
X_train, X_test, Y_train, Y_test = train_test_split(data, classes, train_size=0.8, shuffle=True)

# cap = CAP()
# cap.fit(X_train, Y_train)
# predictions = cap.predict(X_test)
# Y_test


mcap = MULTI_CAP()
mcap.fit(X_train, Y_train)
predictions = mcap.predict(X_test)
Y_test
predictions

# for i in range(len(predictions)):
#     prediction = predictions[i]
#     if np.count_nonzero(prediction) > 1:
#         idx = np.where(prediction == 1)[0]
#         print(idx)



m = len(set(classes))
lernmatrix = np.zeros((m, m), dtype=int)
for i in range(m):
    lernmatrix[i][i] = 1

corr = 0
for i in range(len(Y_test)):
    print(f' {i} '.center(25, '='))
    index = Y_test[i]
    print(f'{lernmatrix[index]} = {predictions[i]}')
    # if (lernmatrix[index] == predictions[i]).all():
    if predictions[i][index] == 1:
        print(True)
        corr += 1
    else:
        print(False)

corr / len(Y_test)



# classes = np.concatenate((np.zeros(50, dtype=int), np.ones(50, dtype=int)))
# data_copy = np.concatenate((data[100:], data[50:100]))
# cag = CAG()
# cag.fit(data_copy, classes)
# predictions = cag.predict(data_copy)
# classes
# predictions

# X_train, X_test, Y_train, Y_test = train_test_by_class(X_train, X_test, Y_train, Y_test)