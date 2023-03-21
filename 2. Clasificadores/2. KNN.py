from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from Distance import *
from KNN import *
from PCA import *
from Performance import *

# =============================== VARIABLES ===================================
# =============================================================================
f_distances = [euclidean, cosine_similarity, manhattan, minkowski, correlation]
f_label = ['euclidean', 'cosine_similarity', 'manhattan', 'minkowski', 'correlation']



# ================================ DATASET ====================================
# =============================================================================
# dataset = datasets.load_breast_cancer()
# data = dataset.data # Datos del dataset
# classes = dataset.target # Clases
# targets = dataset.target_names # Etiqueta de clase
# labels = dataset.feature_names # Etiquetas de los atributos
# Bank
df = pd.read_csv('../Datasets/bank-additional-full.csv', sep=';')
df.drop(df[(df['default'] == 'unknown') | (df['housing'] == 'unknown') | (df['housing'] == 'unknown')].index, inplace=True)
df.replace({'no': 0, 'yes': 1}, inplace=True)
df.poutcome.replace({'failure': 0, 'nonexistent': 1, 'success': 2}, inplace=True)
data = df.values[:, [0, 4, 5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19]].astype('float')
classes = df.values[:, -1].astype('int')
targets = np.array(['Yes', 'No'])
labels = df.columns[[0, 4, 5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19]].values.astype('str')
# Estandarización
# data2 = StandardScaler().fit_transform(data) # Estandarizar datos



# ================================ K-NN =======================================
# =============================================================================
# statics, mc = kfold_kNN(data, classes, f_distances[0], type_KNN='Raul', k=3, multiclass=True, splits=5)
# print(['ACC', 'PPV', 'TPR', 'TNR'])
# print(' k-fold '.center(50, '='))
# print(statics)
# print(' Average '.center(50, '='))
# print(np.mean(statics, 0))
n_exps_kfold_knn(data, classes, f_distances, f_label, k=3, multiclass=False, n_splits=5, n_experiments=1)



# ================================= PCA =======================================
# =============================================================================
# eigen_pairs = PCA(data2)
# wm = weight_matrix(eigen_pairs, 2)
# data_pca = data @ wm
# statics, mc = kfold_kNN(data, classes, f_distance, k=5, multiclass=False, splits=5)
# print(['ACC', 'PPV', 'TPR', 'TNR'])
# print(' k-fold '.center(50, '='))
# print(statics)
# print(' Average '.center(50, '='))
# print(np.mean(statics, 0))
# n_exps_kfold_knn(data, classes, f_distances, f_label, k=5, multiclass=True, n_splits=5, n_experiments=10)



# ============================== Curva ROC ====================================
# =============================================================================
# f_distance = manhattan
# X_train, X_test, Y_train, Y_test = train_test_split(data, classes, train_size=0.5, shuffle=True)
# Y_predicted = kNNR(X_train, Y_train, X_test, f_distance, k=5, PROBABILITY=True)
# ROC_curve(Y_train, Y_test, Y_predicted, targets, pos_label=0, multiclass=False)



# from sklearn.neighbors import KNeighborsClassifier
# neigh = KNeighborsClassifier(n_neighbors=5)
# neigh.fit(X_train, Y_train)
# predic = neigh.predict(X_test)
# predic