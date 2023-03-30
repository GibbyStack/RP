import numpy as np

# Método para obtener el tamaño de clase
def data_by_class(classes):
    cl = list(set(list(classes))) # Obtener número de clases
    dbc = [0] * len(cl) # Lista de elementos por clase inicializada en 0's
    for c in classes:
        dbc[c] += 1 # Aumentar el contador de elementos por clase
    return dbc

# Método para separar las clases en conjunto de entrenamiento y prueba
def train_test_split_by_class(data, classes, train_size = 0.75):
    dbc = data_by_class(classes)
    m = len(dbc) # Numero de clases
    x_train, x_test, y_train, y_test = [], [], [], []
    idx = 0
    for i in range(m):
        t_train = sum(dbc[:i]) + round(dbc[i]*train_size) # Tamaño de entrenamiento
        for _ in range(dbc[i]):
            if idx < t_train: # Si el indice no se sale del de entrenamiento
                x_train.append(data[idx])
                y_train.append(classes[idx])
            else: # El indice se salio de entrenamiento
                x_test.append(data[idx])
                y_test.append(classes[idx])
            idx += 1
    x_train, x_test, y_train, y_test = np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test) 
    return x_train, x_test, y_train, y_test

# from sklearn import datasets
# dataset = datasets.load_breast_cancer()
# data = dataset.data # Datos del dataset
# classes = dataset.target # Clases

# X_train, X_test, Y_train, Y_test = train_test_split(data, classes, train_size=0.9)
# print('\n', len(data))
# print(len(X_train), len(X_test))
