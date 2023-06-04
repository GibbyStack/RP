from sklearn.model_selection import StratifiedKFold, KFold
from Performance import *
import numpy as np
import pandas as pd

def convert_to_biclass(classes, l_classes):
    index_p = np.where(classes == l_classes[0])[0]
    classes[index_p] = 0
    for cls in l_classes[1:]:
        index_n = np.where(classes == cls)[0]
        classes[index_n] = 1

def search_index(classes, l_classes):
    index = []
    for i in range(len(classes)):
        if classes[i] in l_classes:
            index.append(i)
    return np.array(index)
    
class CAP():

    def __init__(self):
        self.vm = []
        self.mc = []

    # Método para entrenar el clasificador
    def fit(self, x_train, y_train, uniclass):
        index_p = np.where(y_train == uniclass[0])[0]
        vmp = np.mean(x_train[index_p], axis=0)
        vmn = []
        for unclass in uniclass[1:]:
            index_n = np.where(y_train == unclass)[0]
            vmn.append(np.mean(x_train[index_n], axis=0))
        vmn = np.array(vmn)
        vmn = np.mean(vmn, axis=0)
        self.vm = (vmp + vmn) / 2
        x_train = x_train - self.vm
        convert_to_biclass(y_train, uniclass)
        n = len(x_train[0])
        self.mc = [[0 for _ in range(n)] for _ in range(2)]
        for i in range(len(x_train)):
            x = x_train[i]
            if y_train[i] == 0:
                self.mc[0] += x * 1
            else:
                self.mc[1] += x * 1

    # Método para obtener predicciones
    def predict(self, x_test, PROBABILITY=False):
        x_test = x_test - self.vm
        predictions = []
        for x in x_test:
            x_mc = self.mc @ x
            if not PROBABILITY:
                prediction = np.argmax(x_mc)
            if PROBABILITY:
                prediction = x_mc
            predictions.append(prediction)
        predictions = np.array(predictions)
        return predictions

def multiCAP(unique_classes, x_train, y_train, x_test):
    aux_index = np.array([], dtype=int)
    predictions = np.full(len(x_test), None)
    for i in range(len(unique_classes)):
        uniclass = unique_classes[i:]
        if i != len(unique_classes) - 1:
            if i != 0:
                aux_index = np.concatenate((index_p, aux_index))

            index = search_index(y_train, uniclass)
            n_x_train = x_train[index].copy()
            n_y_train = y_train[index].copy()
            cap = CAP()
            cap.fit(n_x_train, n_y_train, uniclass)
            y_predicted = cap.predict(x_test)
            index_p = np.where(y_predicted == 0)[0]

            if i != 0:
                d1 = set(aux_index).difference(set(index_p))
                d2 = set(index_p).difference(set(aux_index))
                index_p = np.array(list(d1.union(d2)))
        else:
            index_p = np.where(predictions == None)[0]
        predictions[index_p.astype(int)] = unique_classes[i] 
    return predictions.astype(int)

# Método de validación cruzada para CAP
def kfold_CAP(unique_classes, data, classes, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True)
    m = len(set(classes))
    MC = np.zeros((m, m))
    statics = []
    for (train_index, test_index) in (kf.split(data, classes)):
        X_train, X_test, Y_train, Y_test = data[train_index], data[test_index], classes[train_index], classes[test_index]
        Y_predicted = multiCAP(unique_classes, X_train, Y_train, X_test)
        mc = confusion_matrix(Y_predicted, Y_test, m)
        MC += mc
        ACCr, PPVa, TPRa, TNRa = get_statistics_mc(mc, multiclass=True)
        statics.append([ACCr, PPVa, TPRa, TNRa])
    statics = np.array(statics)
    return statics, MC

# Método para validar multicap con kfold un determinado número de experimentos
def n_exps_kfold_cap(unique_classes, data, classes, n_splits=5, n_experiments=10):
    experiments = []
    m = len(set(classes))
    c_matrix = np.zeros((m, m))
    for _ in range(n_experiments):
        statics, mc = kfold_CAP(unique_classes, data, classes, n_splits)
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

# dataset = datasets.load_iris()
# data = dataset.data # Datos del dataset
# classes = dataset.target # Clases
# targets = dataset.target_names # Etiqueta de clase
# labels = dataset.feature_names # Etiquetas de los atributos

# X_train, X_test, Y_train, Y_test = train_test_split(data, classes, train_size=0.5, shuffle=True)

# unique_classes = [0, 1, 2] # Iris
# unique_classes = [0, 2, 1] # Wine
# unique_classes = [1, 5, 3, 4, 0, 2] # Glass
# unique_classes = [0, 4, 6, 3, 5, 2, 1] # Segment

# n_exps_kfold_cap([0, 1, 2], data, classes)