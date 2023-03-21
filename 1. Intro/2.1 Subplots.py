from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np

# Cargar Dataset Iris Plant
iris = datasets.load_iris()

# Variables
data = iris.data[:100] # Datos de setosa y versicolour
classes = iris.target[:100] # Clases
labels = ['Sepal Length (cm)', 'Sepal Width (cm)', 'Petal Length (cm)', 'Petal Width (cm)'] # Etiquetas de los atributos

def minimal_distance(X, Y):
    x1 = X[:50]
    x2 = X[50:]
    y1 = Y[:50]
    y2 = Y[50:]
    # Calcular prototipos de clase
    meanx1 = (np.mean(x1), np.mean(y1))
    meanx2 = (np.mean(x2), np.mean(y2))
    # - 1/2 * (mx^2 + my^2)
    m1 = (meanx1[0]**2 + meanx1[1]**2)/2
    m2 = (meanx2[0]**2 + meanx2[1]**2)/2
    # Calcular los coeficientes y b
    M1 = meanx1[0]-meanx2[0]
    M2 = meanx1[1]-meanx2[1]
    b = m1 - m2
    return lambda x: (-M1*x+b)/M2

# Crear figura para los plots
fig = plt.figure(figsize=(12, 12))
n = len(data[0]) # Numero de atributos
n_plot = 1 # Contador de plots
for i in range(n):
    for j in range(n):
        ax = fig.add_subplot(n,n, n_plot) # Agregar subplot
        if i == j:
            ax.text(0.25, 0.45, labels[i]) # Si es plot de la diagonal, agregar etiqueta del atributo
        else:
            X = data[:, j]
            Y = data[:, i]
            x = np.linspace(min(X), max(X)) # Generar puntos x
            fy = minimal_distance(X, Y) # Calcular frontera de desicion
            y = fy(x) # Obtner los valores de y en cada punto x
            ax.scatter(X, Y, c=classes,  cmap='RdBu') # Graficar puntos (X, Y)
            ax.plot(x, y, color='black') # Graficar frontera de desición
        n_plot += 1