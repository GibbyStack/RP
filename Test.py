from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt

iris = datasets.load_iris() # Cargar dataset
data = iris.data[:100] # Datos de setosa y versicolour
classes = iris.target[:100] # Clases

X = data[:100, :]
x1 = data[:25, :]
x2 = data[50:75, :]

# Calcular medias
m1 = np.mean(x1, 0)
m2 = np.mean(x2, 0)

for i in range(50, 100):
    g = ((X[i].T @ m1) - (X[i].T @ m2)) - 0.5 * ((m1.T @ m1) - (m2.T @ m2))
    # print(f'[X[{i}]: class[{classes[i]}]] = [{g}]')
    print(g)

iris = datasets.load_iris() # Cargar dataset
data = iris.data
classes = iris.target

# plt.scatter(data[:, 0], data[:, 1], c=classes)

# train = np.concatenate((data[:30], data[50:80], data[100:130]))
# test = np.concatenate((data[30:50], data[80:100], data[130:150]))

# x1 = data[:30]
# x2 = data[50:80]
# x3 = data[100:130]

# m1 = np.mean(x1, 0)
# m2 = np.mean(x2, 0)
# m3 = np.mean(x3, 0)

# print(f'm1 = {m1}')
# print(f'm2 = {m2}')
# print(f'm3 = {m3}')

# g1 = ((data[51].T @ m1) - 0.5 * (m1.T @ m1))
# g2 = ((data[51].T @ m2) - 0.5 * (m2.T @ m2))
# g3 = ((data[51].T @ m3) - 0.5 * (m3.T @ m3))

# print(f'g1 = {g1}')
# print(f'g2 = {g2}')
# print(f'g3 = {g3}')

# def hola(data, classes, targets, splits, pos_label, multiclass=True):
#     m = len(set(classes))
#     np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
#     kf = KFold(n_splits=splits, shuffle=True) # Generador kfold
#     fprs, tprs = [[] for _ in range(m)], [[] for _ in range(m)]
#     for (train_index, test_index) in (kf.split(data)):
#         X_train, X_test, Y_train, Y_test  = data[train_index], data[test_index], classes[train_index], classes[test_index]
#         prototypes = train_minimum_distance(X_train, Y_train) # Generar los prototipos de clase
#         Y_predicted = classify_minimum_distance(X_test, prototypes, f_distance, PROBABILITY=True) # Valores predichos por D-min
#         label_binarizer = LabelBinarizer().fit(Y_train)
#         y_onehot_test = label_binarizer.transform(Y_test)
#         for i in range(m):
#             fpr, tpr, _ = roc_curve(y_onehot_test[:, i], Y_predicted[:, i], pos_label=pos_label, sample_weight=s)
#             fprs[i].append(fpr)
#             tprs[i].append(tpr)
#     return fprs
#     # fprs = np.mean(fprs, 1)
#     # tprs = np.mean(tprs, 1)
#     # fig, ax = plt.subplots(figsize=(6, 6))
#     # for i in range(3):
#     #     roc_auc = auc(fprs[i], tprs[i])
#     #     ax.plot(fprs[i], tprs[i], label=f'{targets[i]} vs el resto - (AUC {roc_auc:.2})')
#     # plt.title("One-vs-Rest multiclass ROC")
#     # plt.plot([0, 1], [0, 1], "k--", label="Curva ROC para (AUC = 0.5)")
#     # plt.axis("square")
#     # plt.xlabel("False Positive Rate")
#     # plt.ylabel("True Positive Rate")
#     # plt.legend()
#     # plt.show()