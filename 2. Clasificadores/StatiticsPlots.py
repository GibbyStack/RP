from matplotlib import pyplot as plt
from Dataclass import *
import pandas as pd
import math

# Método pra obtener un resumen de las estadisticas
def summary_statistics(data, classes, targets, labels):
    dbc = data_by_class(classes) # Obtener cantidad de datos por clase
    for i in range(len(targets)):
        print(f' {targets[i]} '.center(50, '='))
        class_size = dbc[i] # Tamaño de clase
        d = pd.DataFrame(data[i*class_size:(i+1)*class_size], columns=labels)
        print(d.describe())

# Método para generar diagramas de la distribución de caracteristicas
def feature_distribution(data, classes, targets, labels):
    dbc = data_by_class(classes) # Obtener cantidad de datos por clase
    m = len(targets) # Numero de clases
    n = len(data[0]) # Numero de caracterisaticas
    fig = plt.figure(figsize=(12, 3*math.ceil(n/4)))
    for i in range(n):
        ax = fig.add_subplot(math.ceil(n/4), 4, (i+1))
        for j in range(m):
            tc = dbc[j] # Tamaño de clase
            k = round(1 + math.log2(tc)) # Número de clases para histograma
            ax.hist(data[j*tc:(j+1)*tc, i], k, label=targets[j], alpha=0.5)
        ax.legend()
        ax.grid(True, linewidth=0.3)
        ax.set_title(labels[i])
    fig.tight_layout() # Ajustar el espacio de los subplots

# Método para generar diagramas de caja de las caracteristicas
def feature_boxplot(data, classes, targets, labels):
    dbc = data_by_class(classes) # Obtener cantidad de datos por clase
    m = len(targets) # Numero de clases
    n = len(data[0]) # Numero de caracterisaticas
    fig = plt.figure(figsize=(12, 3*math.ceil(n/4)))
    for i in range(n):
        data_class = [] # Datos por clase
        for j in range(m):
            tc = dbc[j] # Tamaño de clase
            data_class.append(data[j*tc:(j+1)*tc, i]) # Agregar una caracteristica por cada clase
        ax = fig.add_subplot(math.ceil(n/4), 4, (i+1))
        bp = ax.boxplot(data_class, labels=targets)
        k = 0
        for box in bp['boxes']:
            color = 'C'+str(k)
            box.set_color(color)
            k += 1
        ax.grid(True, linewidth=0.3)
        ax.set_title(labels[i])
    fig.tight_layout() # Ajustar el espacio de los subplots

# Método para generar diagramas de combinaciones de caracteristicas por clase
def feature_pairs(data, classes, targets, labels):
    n = len(data[0]) # Numero de caracterisaticas    
    n_plot = 1 # Contador de plots
    fig = plt.figure(figsize=(n*3, n*3))
    for i in range(n):
        for j in range(n):
            ax = fig.add_subplot(n, n, n_plot)
            if i == j:
                ax.text(0.25, 0.45, labels[i]) # Si es plot de la diagonal, agregar etiqueta del atributo
            else:
                X = data[:, j] # Datos en el eje X
                Y = data[:, i] # Datos en el eje Y
                sc = ax.scatter(X, Y, c=classes,  cmap='Accent')
                ax.legend(sc.legend_elements()[0], targets)
                ax.grid(True, linewidth=0.3)
            n_plot += 1
    fig.tight_layout() #

# Método para obtener las estadisticas por clase
def experiments_statistics(experiments, f_distances, f_label):
    df = pd.DataFrame(experiments, columns=['ACC', 'PPV', 'TPR', 'TNR', 'f_distance'])
    for i in range(len(f_distances)):
        print(f' {f_label[i]} '.center(50, '='))
        x = df[df.loc[:, 'f_distance'] == f_label[i]]
        x = x.iloc[:, :-1].astype('float64')
        print(x.describe()) 



# from sklearn import datasets
# from sklearn.preprocessing import StandardScaler
# from Datasets import *

# # ================================ DATASET ====================================
# # =============================================================================
# dataset = datasets.load_iris()
# data = dataset.data # Datos del dataset
# classes = dataset.target # Clases
# targets = dataset.target_names # Etiqueta de clase
# labels = dataset.feature_names # Etiquetas de los atributos
# data = StandardScaler().fit_transform(data) # Estandarizar datos
# dbc = data_by_class(classes) # Obtener cantidad de datos por clase

# feature_boxplot(data, dbc, targets, labels) # Diagramas de caja de las caracteristicas
# feature_pairs(data, classes, targets, labels)
# feature_distribution(data, dbc, targets, labels)