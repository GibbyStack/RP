'''
PCA - Principal Component Analysis (Análisis de Componentes Principales)

- Autovectores (eigenvector): son las direcciones en las que la varianza de los datos es mayor, 
  representan la esencia principal de la información contenida en el dataset. 
  (mayor dispersión de los datos = mayor información)

- Autovalores (eigenvalue): el valor de la varianza sobre ese autovector

- Matriz de covarianza: medida de dispersión conjunta entre variables

- Varianza explicada: cuánta varianza se puede atribuir a cada uno de los 
  componentes principales
'''

import numpy as np
from matplotlib import pyplot as plt

# Método para realizar PCA sobre los datos
def PCA (data):
    n = len(data[0]) # Numero de características
    cov_matrix = np.cov(data.T)
    # print('Matriz de covarianza:\n', cov_matrix)
    # Calcular autovalores y autovectores de la matriz
    eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
    # print('\nEigenvectors\n', eig_vecs)
    print('\nEigenvalues\n', eig_vals)
    # Lista de parejas (autovector, autovalor) 
    eigen_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    copy_eigen_pairs = eigen_pairs.copy()
    # Ordenamos estas parejas den orden descendiente con la función sort
    eigen_pairs.sort(key=lambda x: x[0], reverse=True)
    index = [copy_eigen_pairs.index(pos) for pos in eigen_pairs]
    print('\nIndex\n', index)
    # Calculamar la varianza explicada
    tot = sum(eig_vals)
    var_exp = [(eig_val / tot)*100 for eig_val in sorted(eig_vals, reverse=True)]
    # var_exp = [(eig_val / tot)*100 for eig_val in eig_vals]
    cum_var_exp = np.cumsum(var_exp)
    # Diagrama de barras de la varianza explicada por cada autovalor, y la acumulada
    with plt.style.context('seaborn-pastel'):
        plt.figure(figsize=(6, 4))
        plt.bar(range(1, n+1), var_exp, alpha=0.5, align='center',
                label='Varianza individual explicada', color='g')
        plt.step(range(1, n+1), cum_var_exp, where='mid', linestyle='--', label='Varianza explicada acumulada')
        plt.ylabel('Ratio de Varianza Explicada')
        plt.xlabel('Componentes Principales')
        plt.legend(loc='best')
        plt.tight_layout()
    plt.show()
    return eigen_pairs, index

# Método para generar una matriz con los componentes principales
def weight_matrix(eigen_pairs, n_components):
    wm = []
    for i in range(n_components):
        wm.append(eigen_pairs[i][1])
    wm = np.array(wm)
    return wm.T