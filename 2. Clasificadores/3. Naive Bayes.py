from sklearn import datasets
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from NaiveBayes import *
from PCA import *
from Performance import *


# ================================ DATASET ====================================
# =============================================================================
dataset = fetch_openml(name='segment')
data = np.array(dataset.data)
targets = np.array(list(set(dataset.target)))
classes = dataset.target
for i in range(len(targets)):
    classes = classes.replace({targets[i]: i})
classes = np.array(classes, dtype=int)
labels = list(dataset.feature_names)

# dataset = datasets.load_wine()
# data = dataset.data
# classes = dataset.target
# targets = dataset.target_names
# labels = dataset.feature_names



# =========================== NAIVE BAYES =====================================
# =============================================================================
# statics, mc = kfold_naive_bayes(data, classes, multiclass=True, n_splits=5)
# print(['ACC', 'PPV', 'TPR', 'TNR'])
# print(' k-fold '.center(50, '='))
# print(statics)
# print(' Average '.center(50, '='))
# print(np.mean(statics, 0))
# import time
# start = time.time()
# n_exps_kfold_naive_bayes(data, classes, multiclass=True, n_splits=k, n_experiments=10)
# end = time.time()
# print(f'Time = {end-start} s') # Segundos y microsegundos


# ================================= PCA =======================================
# =============================================================================
# standar_data = StandardScaler().fit_transform(data) # Estandarizar datos
# eigen_pairs, index = PCA(standar_data)
# data_pca = data[:,index[:15]]
# wm = weight_matrix(eigen_pairs, 2)
# data_pca = data @ wm
# statics, mc = kfold_naive_bayes(data, classes, multiclass=True, n_splits=5)
# print(['ACC', 'PPV', 'TPR', 'TNR'])
# print(' k-fold '.center(50, '='))
# print(statics)
# print(' Average '.center(50, '='))
# print(np.mean(statics, 0))
# start = time.time()
# n_exps_kfold_naive_bayes(data_pca, classes, multiclass=True, n_splits=k, n_experiments=10)
# end = time.time()
# print(f'Time = {end-start} s') # Segundos y microsegundos



# ============================== Curva ROC ====================================
# =============================================================================
# X_train, X_test, Y_train, Y_test = train_test_split(data, classes, train_size=0.5, shuffle=True)
# naive_bayes = NaiveBayes()
# naive_bayes.fit(X_train, Y_train)
# Y_predicted = naive_bayes.predict(X_test, PROBABILITY=True)
# ROC_curve(Y_train, Y_test, Y_predicted, targets, pos_label=1, multiclass=True)