from sklearn import datasets
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from CAP import *
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

# dataset = datasets.load_wine()
# data = dataset.data
# classes = dataset.target
# targets = dataset.target_names
# labels = dataset.feature_names



# ========================= BACKPROPAGATION ===================================
# =============================================================================
# for k in [5, 7, 10]:
#     print('')
#     print(f' {k} '.center(50, '='))
#     start = time.time()
#     unique_classes = [0, 4, 3, 5, 6, 2, 1]
#     n_exps_kfold_cap(unique_classes, data, classes, n_splits=k, n_experiments=100)
#     end = time.time()
#     print(f'Time = {end-start} s') # Segundos y microsegundos


# ================================= PCA =======================================
# =============================================================================
# standar_data = StandardScaler().fit_transform(data) # Estandarizar datos
# eigen_pairs, index = PCA(standar_data)
# data_pca = data[:,index[:11]]
# for k in [5, 7, 10]:
#     print('')
#     print(f' {k} '.center(50, '='))
#     start = time.time()
#     unique_classes = [0, 4, 3, 5, 6, 2, 1]
#     n_exps_kfold_cap(unique_classes, data_pca, classes, n_splits=k, n_experiments=100)
#     end = time.time()
#     print(f'Time = {end-start} s') # Segundos y microsegundos



# ============================== Curva ROC ====================================
# =============================================================================
# from sklearn.preprocessing import LabelBinarizer
# from sklearn import metrics
# import matplotlib.pyplot as plt

# FPR, TPR = [], []
# l_max, n_classes = 0, len(targets)
# for i in range(n_classes):
#     index_p = np.where(classes == classes[i])[0]
#     index_n = np.where(classes != classes[i])[0]
#     classes[index_p] = 0
#     classes[index_n] = 1
#     X_train, X_test, Y_train, Y_test = train_test_split(data, classes, train_size=0.5, shuffle=True)
#     cap = CAP()
#     cap.fit(X_train, Y_train, uniclass=[0, 1])
#     Y_predicted = cap.predict(X_test, PROBABILITY=True)
#     label_binarizer = LabelBinarizer().fit(Y_test)
#     y_onehot_test = label_binarizer.transform(Y_test)
#     y_onehot_test = 1 - y_onehot_test
    
#     fpr, tpr, _ = metrics.roc_curve(y_onehot_test, Y_predicted[:, 0])
#     FPR.append(fpr)
#     TPR.append(tpr)
#     if len(tpr) > l_max: l_max = len(tpr)

# FPR_mean, TPR_mean = [], []
# for i in range(l_max):
#     sum_f, sum_t = 0, 0
#     count = 0
#     for j in range(n_classes):
#         fpr, tpr = FPR[j], TPR[j]
#         if len(tpr[i:i+1]) > 0:
#             sum_f += fpr[i:i+1][0]
#             sum_t += tpr[i:i+1][0]
#             count += 1
#     if count == 0: count = 1
#     FPR_mean.append(sum_f/count)
#     TPR_mean.append(sum_t/count)

# FPR_mean.sort()
# TPR_mean.sort()

# for i in range(n_classes):
#     fpr, tpr = FPR[i], TPR[i]
#     auc = metrics.auc(fpr, tpr)
#     plt.plot(fpr, tpr, label=targets[i]+f' (AUC = {auc:.2f})', alpha=0.25)
# auc = metrics.auc(FPR_mean, TPR_mean)
# plt.plot(FPR_mean, TPR_mean, label=f'Mean (AUC = {auc:.2f})')
# plt.title("Curve ROC")
# plt.plot([0, 1], [0, 1], "k--", label="Curva ROC para (AUC = 0.5)")
# plt.axis("square")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.legend()
# plt.show()