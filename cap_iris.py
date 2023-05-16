from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
import numpy as np

class AssociativeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.encoder_ = LabelEncoder()
        y_encoded = self.encoder_.fit_transform(y)
        self.nbrs_ = NearestNeighbors(n_neighbors=self.k)
        self.nbrs_.fit(X)
        self.classes_ = self.encoder_.classes_
        self.y_encoded_ = y_encoded
        return self

    def predict(self, X):
        check_is_fitted(self, ['nbrs_', 'classes_', 'y_encoded_'])
        X = check_array(X)
        _, indices = self.nbrs_.kneighbors(X)
        predictions = []
        for neighbors in indices:
            neighbor_classes = self.y_encoded_[neighbors]
            unique_classes, counts = np.unique(neighbor_classes, return_counts=True)
            most_common_class = unique_classes[np.argmax(counts)]
            predicted_class = self.classes_[most_common_class]
            predictions.append(predicted_class)
        return np.array(predictions)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml


# Cargar el conjunto de datos Iris
# iris = load_iris()
# X, y = iris.data, iris.target

dataset = fetch_openml(name='glass')
X = np.array(dataset.data)
targets = np.array(list(set(dataset.target)))
y = dataset.target
for i in range(len(targets)):
    y = y.replace({targets[i]: int(i)})
y = np.array(y, dtype=int)

# labels = list(dataset.feature_names)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear una instancia del clasificador asociativo de patrones
classifier = AssociativeClassifier(k=3)

# Entrenar el clasificador
classifier.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = classifier.predict(X_test)

# Calcular la precisión de las predicciones
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión: {accuracy}")
