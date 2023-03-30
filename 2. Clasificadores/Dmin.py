from sklearn.model_selection import StratifiedKFold
from Performance import *
from StatiticsPlots import *
import numpy as np
import pandas as pd

# Método para obtener los prototipos de clase con Minima Distancia
def train_minimum_distance(x_train, y_train):
    m = len(set(y_train)) # Numero de clases
    prototypes = [] # Prototipos
    data = [[] for x in range(m)] # Datos por clase
    for i in range(len(x_train)):
        data[y_train[i]].append(x_train[i])
    for i in range(m):
        data_class = np.array(data[i])
        prototypes.append(np.mean(data_class, 0)) # Agregar el prototipo de clase
    prototypes = np.array(prototypes)
    return prototypes

# Método para clasificar datos mediante la Minima Distancia
def classify_minimum_distance(x_test, prototypes, f_distance, PROBABILITY=False):
    predicted_class = [] # Clasificaciones
    for x in x_test:
        gs = [] # Evaluaciones de distancia contra prototipos de clase
        # g_max = 0 # Evaluación maxima
        g_min = 0 # Evaluación minima
        for m in prototypes:
            g = f_distance(x, m)
            gs.append(g) # Agregar evaluacion con esa clase
        if PROBABILITY: 
            predicted_class.append(gs)
        if not PROBABILITY:
            # g_max = max(gs) # Obtener la evaluación con el valor maximo
            g_min = min(gs)
            predicted_class.append(gs.index(g_min)) # Obtener el índice de la evaluación maxima
    predicted_class = np.array(predicted_class)
    return predicted_class

# Método de validacion cruzada para minima distancia
def kfold_dmin(data, classes, f_distance, multiclass=True, n_splits=5):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True) # Generador kfold
    m = len(set(classes)) # Numero de clases
    MC = np.zeros((m, m))
    statics = []
    for (train_index, test_index) in (kf.split(data, classes)):
        X_train, X_test, Y_train, Y_test  = data[train_index], data[test_index], classes[train_index], classes[test_index]
        prototypes = train_minimum_distance(X_train, Y_train) # Generar los prototipos de clase
        Y_predicted = classify_minimum_distance(X_test, prototypes, f_distance) # Valores predichos por D-min
        mc = confusion_matrix(Y_predicted, Y_test) # Calcular matriz de confución
        MC += mc # Sumar la matriz de confución
        ACCr, PPVa, TPRa, TNRa = get_statistics_mc(mc, multiclass) # Obtener estadisticos de la matriz de confución
        statics.append([ACCr, PPVa, TPRa, TNRa])
    statics = np.array(statics)
    return statics, MC

# Método para validar minima distancia con kfold un determinado número de experimentos
def n_exps_kfold_dmin(data, classes, f_distances, f_label, multiclass=True, n_splits=5, n_experiments=10):
    n_distances = len(f_distances)
    experiments = []
    m = len(set(classes))
    c_matrix = np.zeros((m, m))
    for i in range(n_distances):
        for _ in range(n_experiments):
            statics, mc = kfold_dmin(data, classes, f_distances[i], multiclass, n_splits=n_splits)
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
# prototypes = train_minimum_distance(X_train, Y_train)
# Y_predicted = classify_minimum_distance(X_test, prototypes, f_distance, PROBABILITY=False)