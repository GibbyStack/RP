from sklearn.model_selection import KFold
from Dataclass import one_hot_encoding
from Performance import *
import numpy as np
import pandas as pd

sigmoid = lambda x: 1 / (1 + np.e**-x)
der_sigmoid = lambda x: x * (1 - x)

tanh = lambda x: (np.e**x - np.e**-x) / (np.e**x + np.e**-x)
der_tanh = lambda x: 1 - tanh(x)**2

relu = lambda x: x if x >= 0 else 0
der_relu = lambda x: 1 if x >= 0 else 0

class NeuralNetwork():

    def __init__(self, layers, f_activation=sigmoid, der_f_activation=der_sigmoid):
        self.n_layers = len(layers) - 1
        self.f_act = f_activation
        self.der_f_act = der_f_activation
        self.bias = []
        self.weights = []

        for i in range(self.n_layers):
            self.bias.append(np.random.rand(1, layers[i+1]) * 2 - 1)
            # self.bias.append(np.zeros((1, layers[i+1])) + 0.5)
            self.weights.append(np.zeros((layers[i], layers[i+1])) + 0.5)

    # Método para obtener predicciones
    def predict(self, X, train=False, PROBABILITY=False):
        outputs = [X]
        for i in range(self.n_layers):
            z = outputs[i] @ self.weights[i] +  self.bias[i]
            a = self.f_act(z)
            outputs.append(a)
        if train:
            return outputs
        outputs = outputs[-1]
        if PROBABILITY:
            return 1-outputs
        predictions = np.argmax(outputs, axis=1)
        return predictions

    # Método para entrenar el clasificador
    def fit(self, X, Y, lr=0.05, epochs=1000):
        for _ in range(epochs):
            outputs = self.predict(X, train=True)
            a = outputs[-1]
            e = Y - a
            d = self.der_f_act(a) * e
            deltas = [d]
            for l in reversed(range(self.n_layers - 1)):
                a = outputs[l+1]
                e = deltas[0] @ self.weights[l + 1].T
                d = self.der_f_act(a) * e
                deltas.insert(0, d)
            for l in range(self.n_layers):
                DW = (outputs[l].T @ deltas[l]) * lr
                self.weights[l] += DW
                self.bias[l] -= np.mean(deltas[l], axis=0, keepdims=True) * lr

# Método de validación cruzada para Backpropagation
def kfold_NN(data, classes, layers, lr, epochs, multiclass=True, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True)
    m = len(set(classes))
    MC = np.zeros((m, m))
    statics = []
    for (train_index, test_index) in (kf.split(data, classes)):
        X_train, X_test, Y_train, Y_test = data[train_index], data[test_index], classes[train_index], classes[test_index]
        Y_train = one_hot_encoding(Y_train)
        nn = NeuralNetwork(layers=layers)
        nn.fit(X_train, Y_train, lr=lr, epochs=epochs)
        Y_predicted = nn.predict(X_test)
        mc = confusion_matrix(Y_predicted, Y_test, m)
        MC += mc
        ACCr, PPVa, TPRa, TNRa = get_statistics_mc(mc, multiclass=multiclass)
        statics.append([ACCr, PPVa, TPRa, TNRa])
    statics = np.array(statics)
    return statics, MC

# Método para validar mínimo distancia con kfold un determinado número de experimentos
def n_exps_kfold_NN(data, classes, layers, lr=0.05, epochs=1000, multiclass=True, n_splits=5, n_experiments=10):
    experiments = []
    m = len(set(classes))
    c_matrix = np.zeros((m, m))
    for _ in range(n_experiments):
        statics, mc = kfold_NN(data, classes, layers=layers, lr=lr, epochs=epochs, multiclass=multiclass, n_splits=n_splits)
        statics_mean = np.mean(statics, 0)
        experiments.append(statics_mean)
        c_matrix += mc
    c_matrix = np.round(c_matrix/n_experiments)
    print(' Matriz de confución '.center(50, '='))
    print(c_matrix)
    df = pd.DataFrame(experiments, columns=['ACC', 'PPV', 'TPR', 'TNR'])
    print(df.describe())


# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.datasets import fetch_openml
# from Distance import *

# dataset = datasets.load_iris()
# data = dataset.data # Datos del dataset
# classes = dataset.target # Clases

# dataset = fetch_openml(name='segment')
# data = np.array(dataset.data)
# targets = np.array(list(set(dataset.target)))
# classes = dataset.target
# for i in range(len(targets)):
#     classes = classes.replace({targets[i]: int(i)})
# classes = np.array(classes, dtype=int)
# labels = list(dataset.feature_names)

# data = StandardScaler().fit_transform(data)
# Y = one_hot_encoding(classes)

# len(classes[0])
# nn = NeuralNetwork([len(data[0]), 2, len(Y[0])])
# nn.fit(data, Y, lr=0.05, epochs=1000)
# pred = nn.predict(data)
# classes
# pred
# classes[:100]
# pred[:100]

# n_exps_kfold_dmin(data, classes, layers=[len(data[0]), 4, 7], lr=0.005, epochs=1000, n_splits=5, n_experiments=1)