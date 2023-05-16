from sklearn.model_selection import KFold
from Performance import *
import numpy as np
import pandas as pd

def search_idx(array, item):
    if item in array:
        idx = np.where(array == item)[0][0]
        return True, idx
    return False, 0.000000000001

class NaiveBayes():
    
    def __init__(self):
        ''' p(A|B) = P(B|A)P(A) / P(B) '''
        self.data_by_class = [] # Datos por clase
        self.class_priors = [] # Probabilidad de clase a priori
        self.likelihoods = [] # Probabilidad de caracteristica dada una clase
        self.label_likehoods = [] # Estiqueta de caracteristica dad una clase
        self.marginal = [] # Probabilidad de caracteristica a priori
        self.label_marginal = [] # Etiqueta de caracteristicas

    # Método para entrenar el clasificador
    def fit(self, x_train, y_train):
        self.data_by_class = np.unique(y_train, return_counts=True)
        self.class_priors = [data_class/sum(self.data_by_class[1]) for data_class in self.data_by_class[1]]

        for i in range(x_train.shape[1]): # Recorrer caracteristicas
            label_likehood, likelihood = [], []
            for c in self.data_by_class[0]: # Recorrer clases
                index = np.where(y_train == c)
                data_class = x_train[index]
                feature_class = np.unique(data_class[:, i], return_counts=True) # Contar caracteristicas por clase
                label_likehood.append(list(feature_class[0]))
                probability = [fc/sum(feature_class[1]) for fc in feature_class[1]] # Probabilidad de caracteristica por clase
                likelihood.append(probability)
            self.label_likehoods.append(label_likehood)
            self.likelihoods.append(likelihood)

            features_count = np.unique(x_train[:, i], return_counts=True) # Contar caracteristica
            self.label_marginal.append(list(features_count[0]))
            probability = [fc/sum(features_count[1]) for fc in features_count[1]] # Probabilidad de caracteristica a priori
            self.marginal.append(probability)
    
    # Método para obtener predicciones
    def predict(self, x_test, PROBABILITY=False):
        predictions = []
        for x in x_test: # Recorrer datos de prueba
            probability = []
            for i in range(len(self.data_by_class[0])): # Recorrer las 3 clases
                p_likelihood, p_marginal = 1, 1
                for j in range(len(x)): # Recorrer caracteristicas del dato
                    search_index = search_idx(self.label_likehoods[j][i], x[j])
                    if search_index[0]:
                        p_likelihood *= self.likelihoods[j][i][search_index[1]]
                    else:
                        p_likelihood *= search_index[1]
                    search_index = search_idx(self.label_marginal[j], x[j])
                    if search_index[0]:
                        p_marginal *= self.marginal[j][search_index[1]]
                    else:
                        p_marginal *= search_index[1]
                p = (p_likelihood * self.class_priors[i]) / p_marginal # Calcular probabilidad bayesiana
                probability.append(p)
            if not PROBABILITY:
                predicted = probability.index(max(probability)) # Obtener predicción
                predictions.append(predicted)
            if PROBABILITY:
                predictions.append(probability)
        predictions = np.array(predictions)
        return predictions
    
# Método de validación cruzada para Naive Bayes
def kfold_naive_bayes(data, classes, multiclass=True, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True)
    m = len(set(classes))
    MC = np.zeros((m, m))
    statics = []
    for (train_index, test_index) in (kf.split(data, classes)):
        X_train, X_test, Y_train, Y_test  = data[train_index], data[test_index], classes[train_index], classes[test_index]
        naive_bayes = NaiveBayes()
        naive_bayes.fit(X_train, Y_train)
        Y_predicted = naive_bayes.predict(X_test)
        mc = confusion_matrix(Y_predicted, Y_test, m) # Matriz de confución
        MC += mc # Sumar la matriz de confución
        ACCr, PPVa, TPRa, TNRa = get_statistics_mc(mc, multiclass) # Obtener estadisticos de la matriz de confución
        statics.append([ACCr, PPVa, TPRa, TNRa])
    statics = np.array(statics)
    return statics, MC

# Método para validar el Naive Bayes con kfold un determinado numero de experimentos 
def n_exps_kfold_naive_bayes(data, classes, multiclass=True, n_splits=5, n_experiments=10):
    experiments = []
    m = len(set(classes))
    c_matrix = np.zeros((m, m))
    for _ in range(n_experiments):
        statics, mc = kfold_naive_bayes(data, classes, multiclass=multiclass, n_splits=n_splits)
        statics_mean = np.mean(statics, 0)
        experiments.append(statics_mean)
        c_matrix += mc
    c_matrix = np.round(c_matrix/n_experiments)
    print(' Matriz de confución '.center(50, '='))
    print(c_matrix)
    df = pd.DataFrame(experiments, columns=['ACC', 'PPV', 'TPR', 'TNR'])
    print(df.describe())