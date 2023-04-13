from sklearn import datasets
from sklearn.model_selection import train_test_split
from NaiveBayes import NaiveBayes
import numpy as np

# ================================ DATASET ====================================
# =============================================================================
dataset = datasets.load_iris()
data = dataset.data
classes = dataset.target
targets = dataset.target_names
labels = dataset.feature_names



# =========================== NAIVE BAYES =====================================
# =============================================================================
X_train, X_test, Y_train, Y_test = train_test_split(data, classes, train_size=0.5, shuffle=True)
naive_bayes = NaiveBayes()
naive_bayes.fit(X_train, Y_train)
Y_predicted = naive_bayes.predict(X_test)