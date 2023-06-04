import random
import numpy as np
from Distance import euclidean

class KMeans():

    # Constructor
    def __init__(self, n_clusters, distance=euclidean, max_iterations=500):
        self.n_clusters = n_clusters
        self.distance = distance
        self.max_iterations = max_iterations
        self.centroids = []
        self.labels = []
        self.inertia = 0
        self.intra_cluster_distance = 0
    
    # Método para inicializar los centroides de manera aleatoria
    def initialize_random_centroids(self, data):
        # Seleccionar k muestras aleatorias como centroides iniciales
        centroids_indices = random.sample(range(len(data)), self.n_clusters)
        self.centroids = [data[i] for i in centroids_indices]
    
    # Método para asignar cada muestra al centroide mas cercano
    def assign_samples_to_centroids(self, data):
        self.labels = []
        for sample in data:
            # Calcular la distancia entre la muestra y todos los centroides
            distances = [self.distance(sample, centroid) for centroid in self.centroids]
            # Asignar la muestra al centroide mas cercano
            centroid_min = np.argmin(distances)
            self.labels.append(centroid_min)
    
    # Método para actualizar los centroides
    def update_centroids(self, data):
        centroids = []
        for i in range(self.n_clusters):
            # Obtener las muestras asignadas al i-ésimo cluster
            cluster_samples = [data[j] for j in range(len(data)) if self.labels[j] == i]
            # Calcular el nuevo centroide como el promedio de las muestras asignadas
            centroid = np.mean(cluster_samples, axis=0)
            centroids.append(centroid)
        self.centroids = centroids

    # Método que calcula la inercia
    def calculate_inertia(self, data):
        for i, sample in enumerate(data):
            centroid = self.centroids[self.labels[i]]
            distance = np.linalg.norm(sample - centroid)
            self.inertia += distance ** 2
    
    # Método para calcular la distancia entre clase
    def calculate_intra_cluster_distance(self, data):
        total_distance = 0
        num_clusters = self.n_clusters

        for i in range(num_clusters):
            cluster_samples = data[self.labels == i]
            centroid = self.centroids[i]
            distance = np.linalg.norm(cluster_samples - centroid, axis=1)
            total_distance += np.sum(distance)

        self.intra_cluster_distance = total_distance / len(data)
            
    # Método para generar los cluster
    def fit(self, data):
        # Inicialización de los centroides de manera aleatoria
        self.initialize_random_centroids(data)

        for _ in range(self.max_iterations):
            # Asignar cada muestra al centroide más cercano
            self.assign_samples_to_centroids(data)
            # Recalcular los centroides
            self.update_centroids(data)
        self.centroids = np.array(self.centroids)
        self.labels = np.array(self.labels)
        # Calcular la inercia
        self.calculate_inertia(data)
        # Calcular la distancia entre clase
        self.calculate_intra_cluster_distance(data)
    



# from sklearn.datasets import load_digits

# digits = load_digits()
# images = digits.data.astype(np.float32)
# labels = digits.target
# images /= 255.0

# kmean = KMeans(n_clusters=10, max_iterations=300)
# kmean.fit(data=images)
# centroids = kmean.centroids
# cluster_labels = kmean.labels


# # PCA Para visualizar los cluster
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# pca = PCA(n_components=2)
# reduced_images = pca.fit_transform(images)
# reduced_centroid = pca.fit_transform(centroids)

# # Visualizar las imágenes en función de los clusters asignados
# plt.scatter(reduced_images[:, 0], reduced_images[:, 1], c=cluster_labels, cmap='viridis')
# plt.scatter(reduced_centroid[:, 0], reduced_centroid[:, 1], marker='x', color='red', s=100, label='Centroids')
# plt.title('Clustering with K-Means')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.legend()
# plt.show()

# # Imprimir los resultados
# for cluster in range(10):
#     print(f"Cluster {cluster}:")
#     cluster_samples = np.where(cluster_labels == cluster)[0]
#     for sample_idx in cluster_samples[:10]:  # Mostrar solo las primeras 5 imágenes por cluster
#         print(f"  - Imagen {sample_idx}: Clase {labels[sample_idx]}")


# from sklearn.metrics import silhouette_score, adjusted_rand_score
# inertia = kmean.inertia
# print(inertia)
# silhouette_avg = silhouette_score(images, cluster_labels)
# print(silhouette_avg)
# adjusted_rand_index = adjusted_rand_score(labels, cluster_labels)
# print(adjusted_rand_index)