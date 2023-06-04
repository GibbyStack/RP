# cls = list(set(classes))
# mincls = min(data_by_class(classes))
# data_roc = []
# classes_roc = []
# for i in range(len(cls)):
#     index = np.where(classes == cls[i])[0]
#     data_roc.append(data[index[:mincls]])
#     classes_roc.append(classes[index[:mincls]])
# data_roc = np.array(data_roc)
# data_roc = data_roc.reshape((mincls*2, data.shape[1]))
# classes_roc = np.array(classes_roc)
# classes_roc = classes_roc.reshape(-1)


# def build_confusion_matrix(Y_predicted, Y_test):
#     mc = np.zeros((2, 2))
#     for i in range(len(Y_predicted)):
#         if Y_predicted[i] == 0:
#             if Y_test[i] == 0:
#                 mc[0][0] += 1
#             else:
#                 mc[0][1] += 1
#         else:
#             if Y_predicted[i] == Y_test[i]:
#                 mc[1][1] += 1
#             else:
#                 mc[1][0] += 1
#     return mc

# statics = []
# size = len(cls) / len(data_roc)
# train_size = 0
# while train_size <= 1 - size:
#     train_size += size
#     X_train, X_test, Y_train, Y_test = train_test_split_by_class(data_roc, classes_roc, train_size=train_size)
#     X_test[:round(train_size * mincls)]
#     Y_test[:round(train_size * mincls)]
#     prototypes = train_minimum_distance(X_train, Y_train)
#     Y_predicted = classify_minimum_distance(X_test, prototypes, f_distance, PROBABILITY=False)
#     mc = confusion_matrix(Y_predicted, Y_test)
#     print('='*10)
#     print(mc)
#     _, _, TPR, TNR = get_statistics_mc(mc, multiclass=False)
#     statics.append([TPR, TNR])

# statics = np.array(statics)

# CALCULO DE DISTANCIAS
# x = [list(x)]*len(x_train)
# distances = list(map(f_distance, x_train, x))
# distances = np.expand_dims(distances, axis=0).T
# distances = np.append(distances, y_train, axis=1)
# print(y_train)

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


# from sklearn.neighbors import KNeighborsClassifier
# neigh = KNeighborsClassifier(n_neighbors=5)
# neigh.fit(X_train, Y_train)
# predic = neigh.predict(X_test)
# predic


# cls = list(set(classes))
# index = np.where(classes == 0)[0]
# for i in range(1, len(cls)):
#     idx = np.where(classes == cls[i])[0]
#     index = np.concatenate((index, idx))
# new_classes = classes[index]
# new_data = data[index]

# dbc = data_by_class(new_classes)
# max = 0
# for i in range(len(dbc) - 1):
#     max = np.sum(dbc[:i+2])
#     x_classes = new_classes[:max].copy()
#     x_data = new_data[:max].copy()
#     if i > 0:
#         x_classes[x_classes != i+1] = 0
#     print(' X_data '.center(50, '='))
#     print(x_classes)

# m = np.mean(data, axis=0)

# data_s = data - m

# data_s

# mc = [[0 for i in range(4)] for j in range(2)]

# for i in range(len(data)):
#     x = data_s[i]
#     if classes[i] == 0:
#         mc[0] += x * 1
#         mc[1] += x * 0
#     else:
#         mc[0] += x * 0
#         mc[1] += x * 1
# mc

# for x in data_s:
#     r = mc @ x
#     index = np.where(r == max(r))[0][0]
#     print(index)

# FPR_mean, TPR_mean = [], []
# limits = [0.0, 0.2, 0.4, 0.6, 0.8, 1.1]
# for i in range(len(limits) - 1):
#   print(f'Limits = {limits[i]}, {limits[i+1]}'.center(50, '='))
#   N_fpr, N_tpr = np.array([]), np.array([])
#   for j in range(len(FPR)):
#     fpr, tpr = FPR[j], TPR[j]
#     f_idx = np.where(fpr >= limits[i])[0]
#     n_fpr, n_tpr = fpr[f_idx], tpr[f_idx]
#     f_idx = np.where(n_fpr < limits[i+1])[0]
#     n_fpr, n_tpr = n_fpr[f_idx], n_tpr[f_idx]
#     N_fpr = np.concatenate((N_fpr, n_fpr))
#     N_tpr = np.concatenate((N_tpr, n_tpr))
#   if len(N_fpr):
#     FPR_mean.append(sum(N_fpr) / len(N_fpr))
#   if len(N_tpr):
#     TPR_mean.append(sum(N_tpr) / len(N_tpr))

# FPR_mean
# TPR_mean
# auc = metrics.auc(FPR_mean, TPR_mean)
# auc

# def train_test_by_class(x_train, x_test, y_train, y_test):
#     classes = list(set(y_train))
#     index_train, index_test = np.array([], dtype=int), np.array([], dtype=int)
#     for c in classes:
#         idx_train = np.where(y_train == c)[0]
#         idx_test = np.where(y_test == c)[0]
#         index_train = np.concatenate((index_train, idx_train))
#         index_test = np.concatenate((index_test, idx_test))
#     x_train, y_train = x_train[index_train], y_train[index_train]
#     x_test, y_test = x_test[index_test], y_test[index_test]
#     return x_train, x_test, y_train, y_test

# class CAP():

#     def __init__(self):
#         self.m = []
#         self.mc = []

#     # Método para entrenar el clasificador
#     def fit(self, x_train, y_train):
#         self.m = np.mean(x_train, axis=0)
#         x_train = x_train - self.m
#         n = len(x_train[0])
#         self.mc = [[0 for _ in range(n)] for _ in range(2)]
#         for i in range(len(x_train)):
#             x = x_train[i]
#             if y_train[i] == 0:
#                 self.mc[0] += x * 1
#             else:
#                 self.mc[1] += x * 1

#     # Método para obtener predicciones
#     def predict(self, x_test):
#         x_test = x_test - self.m
#         predictions = []
#         for x in x_test:
#             x_mc = self.mc @ x
#             prediction = np.argmax(x_mc)
#             predictions.append(prediction)
#         predictions = np.array(predictions)
#         return predictions


# KMEANS
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.datasets import load_digits

# # Cargar el conjunto de datos CIFAR-10
# cifar = load_digits()
# X = cifar.data.astype(np.float32)  # Obtener las imágenes en forma de matriz de características
# y = cifar.target  # Etiquetas de clase correspondientes a cada imagen

# # Normalizar los datos
# X /= 255.0

# # Aplicar k-means
# num_clusters = 10  # Número de clusters deseado
# kmeans = KMeans(n_clusters=num_clusters)
# kmeans.fit(X)

# # Obtener las etiquetas de cluster asignadas a cada imagen
# cluster_labels = kmeans.labels_

# # Imprimir los resultados
# for cluster in range(num_clusters):
#     print(f"Cluster {cluster}:")
#     cluster_samples = np.where(cluster_labels == cluster)[0]
#     for sample_idx in cluster_samples[:10]:  # Mostrar solo las primeras 5 imágenes por cluster
#         print(f"  - Imagen {sample_idx}: Clase {y[sample_idx]}")

# from sklearn.metrics import silhouette_score, adjusted_rand_score
# inertia = kmeans.inertia_
# print(inertia)
# silhouette_avg = silhouette_score(X, cluster_labels)
# print(silhouette_avg)
# adjusted_rand_index = adjusted_rand_score(y, cluster_labels)
# print(adjusted_rand_index)

# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# centroids = kmeans.cluster_centers_

# pca = PCA(n_components=2)
# reduced_images = pca.fit_transform(X)
# reduced_centroid = pca.fit_transform(centroids)

# # Visualizar las imágenes en función de los clusters asignados
# plt.scatter(reduced_images[:, 0], reduced_images[:, 1], c=cluster_labels, cmap='viridis')
# plt.scatter(reduced_centroid[:, 0], reduced_centroid[:, 1], marker='x', color='red', s=100, label='Centroids')
# plt.title('Clustering with K-Means')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.legend()
# plt.show()