import numpy as np
from matplotlib import pyplot as plt

# *y = mx + b* (m = La pendiente)(b = Punto de corte)
def minimos_cuadrados(x, y):
    n = len(x)
    xy = x*y
    xx = x**2
    sumx = x.sum()
    sumy = x.sum()
    sumxy = xy.sum()
    sumxx = xx.sum()
    # m = (Σxy-((Σx*Σy)/n))/(Σx^2-(((Σx)^2)/n))
    m = (sumxy-((sumx*sumy)/n))/(sumxx-((sumx**2)/n))
    # b = (Σy/n)-m*(Σx/n)
    b = (sumy/n)-(m*(sumx/n))
    return m, b

# Conjunto de positivos
positivos = [np.random.uniform(0, 10, 200), np.random.uniform(0, 10, 200)]
x = positivos[0]
y = positivos[1]
m, b = minimos_cuadrados(x, y)
plt.scatter(x, y)
plt.plot(x, m*x+b, 'r')
plt.title('Conjunto de positivos')
plt.show()

# Conjunto de negativos
negativos = [np.random.uniform(-10, 10, 200), np.random.uniform(0, 10, 200)]
x = negativos[0]
y = negativos[1]
m, b = minimos_cuadrados(x, y)
plt.scatter(x, y)
plt.plot(x, m*x+b, 'r')
plt.title('Conjunto de negativos')
plt.show()

# Conjunto de los normales
x = np.random.uniform(0, 10, 200)
mean = x.mean()
std = x.std()
y = np.random.normal(mean, std, 200)
m, b = minimos_cuadrados(x, y)
plt.scatter(x, y)
plt.plot(x, m*x+b, 'r')
plt.title('Conjunto de normales')
plt.show()