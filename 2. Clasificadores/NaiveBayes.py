import numpy as np

def search_idx(array, item):
    if item in array:
        idx = np.where(array == item)[0][0]
        return True, idx
    return False, 0.000000000001

class NaiveBayes():
    def __init__(self):
        self.data_by_class = [] # Datos por clase
        self.class_priors = [] # Probabilidad de clase a priori
        self.likelihoods = [] # Probabilidad de caracteristica dada una clase
        self.label_likehoods = [] # Estiqueta de caracteristica dad una clase
        self.marginal = [] # Probabilidad de caracteristica a priori
        self.label_marginal = [] # Etiqueta de caracteristicas

    # Método de entrenamiento
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
    
    def predict(self, x_test):
        predictions = []
        for x in x_test:
            probability = []
            for i in range(len(self.data_by_class[0])):
                p_likelihood, p_marginal = 1, 1
                for j in range(len(x)):
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
                p = (p_likelihood * self.class_priors[i]) / p_marginal
                probability.append((p, self.data_by_class[0][i]))
            predicted = max(probability)[1]
            predictions.append(predicted)
        predictions = np.array(predictions)
        return predictions