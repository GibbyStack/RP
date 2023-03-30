from sklearn.model_selection import StratifiedKFold
from Performance import *
from StatiticsPlots import *
import numpy as np

# Método para obtner la clase predicción
def get_predicted(k_neighbors):
    # Contar los datos por clase
    predicted = k_neighbors.index(max(k_neighbors))
    return predicted

# Método para obtener la probabilidad de pertenencia a una clase
def get_probabilities(k_neighbors):
    # Crear lista con indices de las clases
    odds = [0] * len(k_neighbors)
    # Convertir la cantidad en probabilidades
    for i in range(len(odds)):
        probability = k_neighbors[i] / sum(k_neighbors)
        odds[i] = probability
        # print(odds)
    return odds

# Método kNN estandar
def kNN(x_train, y_train, x_test, f_distance, k=1, PROBABILITY=False):
    y_predicted = []
    nk = len(set(y_train))
    # y_train = np.expand_dims(y_train, axis=0).T
    for x in x_test:
        # Sacar las distancias con respecto a los puntos de entrenamiento
        distances = []
        for i in range(len(x_train)):
            d = f_distance(x, x_train[i])
            distances.append((d, y_train[i]))
        distances = np.array(distances)
        # Ordenar las distancias
        distances = distances[distances[:, 0].argsort()]
        # Contar los vecinos mas cercanos
        k_neighbors = [0] * nk
        for j in range(k):
            k_neighbors[int(distances[j][1])] += 1
        if PROBABILITY:
            odds = get_probabilities(k_neighbors)
            y_predicted.append(odds)
        if not PROBABILITY:
            predicted = get_predicted(k_neighbors)
            y_predicted.append(predicted)
    if PROBABILITY: y_predicted = np.array(y_predicted)
    if not PROBABILITY: y_predicted = np.array(y_predicted).astype(int)
    return y_predicted

# Método KNN Dr. Raul
def kNNR(x_train, y_train, x_test, f_distance, k=1, PROBABILITY=False):
    y_predicted = []
    for x in x_test:
        # Sacar las distancias con respecto a los puntos de entrenamiento
        distances = []
        for i in range(len(x_train)):
            d = f_distance(x, x_train[i])
            distances.append((d, y_train[i]))
        distances = np.array(distances)
        # Ordenar las distancias
        distances = distances[distances[:, 0].argsort()]
        # Contar hasta encontrar los k vecinos de una clase
        k_neighbors = [0] * len(set(y_train))
        j = 0
        while max(k_neighbors) != k:
            k_neighbors[int(distances[j][1])] += 1
            j += 1
        if PROBABILITY:
            odds = get_probabilities(k_neighbors)
            y_predicted.append(odds)
        if not PROBABILITY:
            predicted = get_predicted(k_neighbors)
            y_predicted.append(predicted)
    if PROBABILITY: y_predicted = np.array(y_predicted)
    if not PROBABILITY: y_predicted = np.array(y_predicted).astype(int)
    return y_predicted

# Método de validación cruzada para k-NN
def kfold_kNN(data, classes, f_distance, type_KNN='Estandar', k=1, multiclass=True, n_splits=5):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True) # Generar kfold
    m = len(set(classes)) # Numero de clases
    MC = np.zeros((m, m))
    statics = []
    for (train_index, test_index) in (kf.split(data, classes)):
        X_train, X_test, Y_train, Y_test  = data[train_index], data[test_index], classes[train_index], classes[test_index]
        if type_KNN == 'Estandar':
            Y_predicted = kNN(X_train, Y_train, X_test, f_distance, k) # Valores predichos por knn
        if type_KNN == 'Raul':
            Y_predicted = kNNR(X_train, Y_train, X_test, f_distance, k) # Valores predichos por knn
        mc = confusion_matrix(Y_predicted, Y_test) # Matriz de confución
        MC += mc # Sumar la matriz de confución
        ACCr, PPVa, TPRa, TNRa = get_statistics_mc(mc, multiclass) # Obtener estadisticos de la matriz de confución
        statics.append([ACCr, PPVa, TPRa, TNRa])
    statics = np.array(statics)
    return statics, MC

# Método para validar el k-NN con kfold un determinado numero de experimentos 
def n_exps_kfold_knn(data, classes, f_distances, f_label, type_KNN='Estandar', k=1, multiclass=True, n_splits=5, n_experiments=10):
    n_distances = len(f_distances)
    experiments = []
    m = len(set(classes))
    c_matrix = np.zeros((m, m))
    for i in range(n_distances):
        for exp in range(n_experiments):
            statics, mc = kfold_kNN(data, classes, f_distances[i], type_KNN, k=k, multiclass=multiclass, n_splits=n_splits)
            statics_mean = np.append(np.mean(statics, 0), f_label[i])
            experiments.append(statics_mean)
            c_matrix += mc
    c_matrix = np.round(c_matrix/(len(f_distances)*n_experiments))
    print(' Matriz de confución '.center(50, '='))
    print(c_matrix)
    experiments_statistics(experiments, f_distances, f_label)



# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from Distance import *

# dataset = datasets.load_iris()
# data = dataset.data # Datos del dataset
# classes = dataset.target # Clases
# targets = dataset.target_names # Etiqueta de clase
# labels = dataset.feature_names # Etiquetas de los atributos

# f_distance = euclidean

# X_train, X_test, Y_train, Y_test = train_test_split(data, classes, train_size=0.8, shuffle=True)
# y_p = kNN(X_train, Y_train, X_test, f_distance, k=3, PROBABILITY=True)