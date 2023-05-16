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