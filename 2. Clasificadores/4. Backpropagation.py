from sklearn import datasets
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from Backpropagation import *
from PCA import *
from Performance import *
import time

# ================================ DATASET ====================================
# =============================================================================
dataset = fetch_openml(name='segment')
# dataset = fetch_openml(name='glass')
data = np.array(dataset.data)
targets = np.array(list(set(dataset.target)))
classes = dataset.target
for i in range(len(targets)):
    classes = classes.replace({targets[i]: i})
classes = np.array(classes, dtype=int)
labels = list(dataset.feature_names)

# dataset = datasets.load_iris()
# data = dataset.data
# classes = dataset.target
# targets = dataset.target_names
# labels = dataset.feature_names

data = StandardScaler().fit_transform(data) # Estandarizar datos
n_features, n_classes = len(data[0]), len(set(classes))


# ========================= BACKPROPAGATION ===================================
# =============================================================================
# for k in [5, 7, 10]:
#     print('')
#     print(f' {k} '.center(50, '='))
#     start = time.time()
    # n_exps_kfold_NN(data, classes, layers=[n_features, 4, n_classes], lr=0.005, epochs=1000, n_splits=k, n_experiments=2)
    # n_exps_kfold_NN(data, classes, layers=[n_features, 4, n_classes], lr=0.05, epochs=1000, n_splits=k, n_experiments=2)
    # end = time.time()
    # print(f'Time = {end-start} s') # Segundos y microsegundos



# ================================= PCA =======================================
# =============================================================================
# eigen_pairs, index = PCA(data)
# data_pca = data[:,index[:15]]
# wm = weight_matrix(eigen_pairs, 2)
# data_pca = data @ wm
# for k in [5, 7, 10]:
#     print('')
#     print(f' {k} '.center(50, '='))
#     start = time.time()
    # n_exps_kfold_NN(data_pca, classes, layers=[len(data_pca[1]), 4, n_classes], lr=0.05, epochs=1000, n_splits=k, n_experiments=2)
    # n_exps_kfold_NN(data_pca, classes, layers=[len(data_pca[1]), 4, n_classes], lr=0.005, epochs=1000, n_splits=k, n_experiments=2)
    # end = time.time()
    # print(f'Time = {end-start} s') # Segundos y microsegundos



# ============================== Curva ROC ====================================
# =============================================================================
X_train, X_test, Y_train, Y_test = train_test_split(data, classes, train_size=0.5, shuffle=True)
Y_train = one_hot_encoding(Y_train)
nn = NeuralNetwork(layers=[n_features, 4, n_classes])
nn.fit(X_train, Y_train, lr=0.005, epochs=1000)
Y_predicted = nn.predict(X_test, PROBABILITY=True)
ROC_curve(Y_test, Y_predicted, targets)