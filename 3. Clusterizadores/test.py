import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from Colors import *

# Lectura de imagen
image = cv2.imread('./Dataset/objetos.png', 1)

# Aplicar un padding a la iamgen
image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

# Convertir la imagen BGR a RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)

# Convertir la imagen RGB a XYZ
# image_xyz = cv2.cvtColor(image, cv2.COLOR_RGB2XYZ)
# plt.imshow(image_xyz)

# Convertir la imagen XYZ a L*a*b
image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
plt.imshow(image_lab)

# Convertir la imagen 3D a 2D
vector_image = image_lab.reshape((-1, 3))

# KMeans
# kmeans = KMeans(n_clusters=2, max_iter=300)
kmeans = KMeans(n_clusters=3, max_iter=300)
kmeans.fit(vector_image)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
centroids = np.uint8(centroids) # Convertir valores al rango [0, 255]
result = centroids[labels.flatten()] # Obtener vector con los centroides
image_kmeans = result.reshape(image_lab.shape) # Convertir vector a imagen 3D
plt.imshow(image_kmeans)

# Convertir imagen a escala de grises
image_gray = cv2.cvtColor(image_kmeans, cv2.COLOR_RGB2GRAY)
plt.imshow(image_gray, cmap='gray')

# Aplicar el filtro medio a la imagen
# image_median = cv2.medianBlur(image_gray, ksize=7)
image_median = cv2.medianBlur(image_gray, ksize=29)
plt.imshow(image_median, cmap='gray')

# set(image_median.flatten())
# Binarización de la imagen
image_binary = binary(image_median, umbral=98)
plt.imshow(image_binary, cmap='gray')

# Extracción de bordes mediante Canny
image_edges = cv2.Canny(image_binary, 0, 1)
plt.imshow(image_edges)

# Realizar una operación de cierre en los bordes
kernel = np.ones((3, 3), np.uint8)
image_edges = cv2.morphologyEx(image_edges, cv2.MORPH_CLOSE, kernel)
plt.imshow(image_edges)

# Segmentación de la imagen
contours, _ = cv2.findContours(image_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Crear una máscara para los objetos encontrados
mask = np.zeros_like(image_edges)
# Dibujar los contornos en la máscara
cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
# Aplicar la máscara a la imagen original
image_segment = cv2.bitwise_and(image, image, mask=mask)
plt.imshow(image_segment)

# Crear una máscara para cada objeto encontrado y mostrarlo
for i, contour in enumerate(contours):
    # Crear una máscara en blanco del mismo tamaño que los bordes extraídos
    mask = np.zeros_like(image_edges)
    # Dibujar el contorno actual en la máscara
    cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
    # if np.count_nonzero(mask) > 2000:
    # Aplicar la máscara a la imagen original
    result = cv2.bitwise_and(image, image, mask=mask)
    # Mostrar el objeto segmentado
    plt.imshow(result)
    plt.show()