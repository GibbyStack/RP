from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np

iris = datasets.load_iris() # Cargar dataset

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
    return (lambda x, y: (M1*x)+(M2*y)-(b), lambda x: (-M1*x+b)/M2, M1, M2, b)

def plotear_frontier(X, Y, xlabel, ylabel, fy, fx, classes=classes):
    scatter = plt.scatter(X, Y, c=classes, cmap='RdBu') # Graficar puntos (X, Y)
    x = np.linspace(min(X), max(X)) # Generar puntos x
    y = fy(x) # Obtner los valores de y en cada punto x
    plt.plot(x, y, color='black') # Graficar frontera de desición
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(handles=scatter.legend_elements()[0], labels=['Setosa', 'Versicolour'])
    plt.title(f'{fx[0]:.1f}x1{fx[1]:.1f}x2{fx[2]:.1f} = 0')

def confusion_matrix(yp, classes=classes):
    mc = np.zeros((2, 2)) # Crear matriz de ceros
    for i in range(len(yp)): # Recorrer cada uno de los valores predichos
        if yp[i] > 0:
            if classes[i] == 0: # TP
                mc[0][0] += 1
            else:
                mc[0][1] += 1 # FP
        if yp[i] < 0:
            if classes[i] == 1: # TN
                mc[1][1] += 1
            else:
                mc[1][0] += 1 # FN
    return mc

def get_statistics(mc):
    TP = mc[0][0] # True positives
    FP = mc[0][1] # False positives
    FN = mc[1][0] # False negatives
    TN = mc[1][1] # True negatives
    P = TP + FN # Positives
    N = TN + FP # Negatives
    ACC = (TP+TN) / (P+N) # Accuracy
    PPV = TP / (TP+FP) # Precision
    TPR = TP / P # Sensitivity
    TNR = TN / N # specificity
    print(' MC '.center(15, '='))
    print(mc)
    print(''.center(15, '='))
    print(f'ACC = {ACC}')
    print(f'PPV = {PPV}')
    print(f'TPR = {TPR}')
    print(f'TNR = {TNR}')

# Caso
i, j = 2, 3
X = data[:, i] # Sepal Length
Y = data[:, j] # Sepal Width
# Calcular frontera de desición mediante minima distancia
fx, y, M1, M2, b = minimal_distance(X, Y)
# Graficar los puntos con la frontera de desición
plotear_frontier(X, Y, labels[i], labels[j], y, [M1, M2, b])
# Calcular valores mediante la funcion en X, Y
yp = fx(X, Y) 
# Generar matriz de confución
mc = confusion_matrix(yp)
# Obtener metricas
get_statistics(mc)
# print(yp)