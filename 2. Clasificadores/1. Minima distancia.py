from sklearn import datasets
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from Distance import *
from StatiticsPlots import *
from Dmin import *
from PCA import *
from Performance import *

# =============================== VARIABLES ===================================
# =============================================================================
f_distances = [euclidean, cosine_similarity, manhattan, minkowski, correlation, chebyshev]
f_label = ['euclidean', 'cosine_similarity', 'manhattan', 'minkowski', 'correlation', 'chebyshev']



# ================================ DATASET ====================================
# =============================================================================
dataset = fetch_openml(name='glass')
data = np.array(dataset.data)
targets = np.array(list(set(dataset.target)))
classes = dataset.target
for i in range(len(targets)):
    classes = classes.replace({targets[i]: int(i)})
classes = np.array(classes, dtype=int)
labels = list(dataset.feature_names)

# dataset = datasets.load_wine()
# data = dataset.data # Datos del dataset
# classes = dataset.target # Clases
# targets = dataset.target_names # Etiqueta de clase
# labels = dataset.feature_names # Etiquetas de los atributos



# =============================== ANALISIS ====================================
# =============================================================================
# summary_statistics(data, classes, targets, labels) # Vision estadistica de las caracteristicas
# feature_distribution(data, classes, targets, labels) # Distribución de las características
# feature_boxplot(data, classes, targets, labels) # Diagramas de caja de las caracteristicas
# feature_pairs(data, classes, targets, labels) # Combinaciones de pares de caracteristicas de las clases



# ================================ D-MIN ======================================
# =============================================================================
# f_distance = euclidean
# statics, mc = kfold_dmin(data, classes, f_distance, multiclass=True, n_splits=5)
# print(['ACC', 'PPV', 'TPR', 'TNR'])
# print(' k-fold '.center(50, '='))
# print(statics)
# print(' Average '.center(50, '='))
# print(np.mean(statics, 0))

# import time
# start = time.time()
# n_exps_kfold_dmin(data, classes, f_distances, f_label, multiclass=True, n_splits=5, n_experiments=100)
# end = time.time()
# print(f'Time = {end-start} s') # Segundos y microsegundos



# ================================= PCA =======================================
# =============================================================================
# standar_data = StandardScaler().fit_transform(data) # Estandarizar datos
# eigen_pairs, index = PCA(standar_data)
# data_pca = data[:,index[:4]]
# wm = weight_matrix(eigen_pairs, 1)
# data_pca = data @ wm
# statics, mc = kfold_dmin(data_pca, classes, f_distances[0], multiclass=True, n_splits=5)
# print(['ACC', 'PPV', 'TPR', 'TNR'])
# print(' k-fold '.center(50, '='))
# print(statics)
# print(' Average '.center(50, '='))
# print(np.mean(statics, 0))
# start = time.time()
# n_exps_kfold_dmin(data_pca, classes, f_distances, f_label, multiclass=True, n_splits=10, n_experiments=100)
# end = time.time()
# print(f'Time = {end-start} s') # Segundos y microsegundos



# ============================== Curva ROC ====================================
# =============================================================================
# f_distance = manhattan
# X_train, X_test, Y_train, Y_test = train_test_split(data, classes, train_size=0.5, shuffle=True)
# minimun_distance = MinimumDistance()
# minimun_distance.fit(X_train, Y_train)
# Y_predicted = minimun_distance.predict(X_test, f_distance, PROBABILITY=True)
# ROC_curve(Y_train, Y_test, Y_predicted, targets, pos_label=0, multiclass=True)