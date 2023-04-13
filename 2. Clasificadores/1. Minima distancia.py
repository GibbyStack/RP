from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from Distance import *
from StatiticsPlots import *
from Dmin import *
from Performance import *
from PCA import *
from CSV import *

# =============================== VARIABLES ===================================
# =============================================================================
f_distances = [euclidean, cosine_similarity, manhattan, minkowski, correlation]
f_label = ['euclidean', 'cosine_similarity', 'manhattan', 'minkowski', 'correlation']



# ================================ DATASET ====================================
# =============================================================================
dataset = datasets.load_iris()
data = dataset.data # Datos del dataset
classes = dataset.target # Clases
targets = dataset.target_names # Etiqueta de clase
labels = dataset.feature_names # Etiquetas de los atributos
# Bank
# df = pd.read_csv('../Datasets/bank-additional-full.csv', sep=';')
# df.drop(df[(df['default'] == 'unknown') | (df['housing'] == 'unknown') | (df['housing'] == 'unknown')].index, inplace=True)
# df.replace({'no': 0, 'yes': 1}, inplace=True)
# df.poutcome.replace({'failure': 0, 'nonexistent': 1, 'success': 2}, inplace=True)
# data = df.values[:, [0, 4, 5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19]].astype('float')
# classes = df.values[:, -1].astype('int')
# targets = np.array(['Yes', 'No'])
# labels = df.columns[[0, 4, 5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19]].values.astype('str')
# HIGGS
# df = pd.read_csv('../Datasets/HIGGS.csv', header=None)
# Estandarización
# standar_data = StandardScaler().fit_transform(data) # Estandarizar datos



# =============================== ANALISIS ====================================
# =============================================================================
# summary_statistics(data, classes, targets, labels) # Vision estadistica de las caracteristicas
# feature_distribution(data, classes, targets, labels) # Distribución de las características
# feature_boxplot(data, classes, targets, labels) # Diagramas de caja de las caracteristicas
# feature_pairs(data, classes, targets, labels) # Combinaciones de pares de caracteristicas de las clases



# ================================ D-MIN ======================================
# =============================================================================
# f_distance = euclidean
# statics, mc = kfold_dmin(data, classes, f_distance, multiclass=True, splits=5)
# print(['ACC', 'PPV', 'TPR', 'TNR'])
# print(' k-fold '.center(50, '='))
# print(statics)
# print(' Average '.center(50, '='))
# print(np.mean(statics, 0))
# import time
# start = time.time()
# n_exps_kfold_dmin(data, classes, f_distances, f_label, multiclass=True, n_splits=10, n_experiments=100)
# end = time.time()
# print(f'Time = {end-start} s') # Segundos y microsegundos



# ================================= PCA =======================================
# =============================================================================
# eigen_pairs, index = PCA(standar_data)
# data_pca = data[:,index[:2]]
# wm = weight_matrix(eigen_pairs, 2)
# data_pca = data @ wm
# statics, mc = kfold_dmin(data_pca, classes, f_distances[0], multiclass=False, n_splits=10)
# print(['ACC', 'PPV', 'TPR', 'TNR'])
# print(' k-fold '.center(50, '='))
# print(statics)
# print(' Average '.center(50, '='))
# print(np.mean(statics, 0))
# start = time.time()
# n_exps_kfold_dmin(data_pca, classes, f_distances, f_label, multiclass=False, n_splits=5, n_experiments=100)
# end = time.time()
# print(f'Time = {end-start} s') # Segundos y microsegundos



# ============================== Curva ROC ====================================
# =============================================================================
f_distance = euclidean
X_train, X_test, Y_train, Y_test = train_test_split(data, classes, train_size=0.5, shuffle=True)
prototypes = train_minimum_distance(X_train, Y_train)
Y_predicted = classify_minimum_distance(X_test, prototypes, f_distance, PROBABILITY=True)
# ROC_curve(Y_train, Y_test, Y_predicted, targets, pos_label=0, multiclass=True)