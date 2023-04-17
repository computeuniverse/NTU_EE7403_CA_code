import numpy as np
from numpy.linalg import inv
from numpy import matmul, inner
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# compute Mahalanobis Distance
def Mahalanobis_D(x, mean1, mean2, sigma):
    d_1 = matmul(matmul((x - mean1).reshape(1, -1), inv(sigma)), (x - mean1).reshape(-1, 1))
    d_2 = matmul(matmul((x - mean2).reshape(1, -1), inv(sigma)), (x - mean2).reshape(-1, 1))

    if min(d_1, d_2) == d_1:
        return 'B'
    elif min(d_1, d_2) == d_2:
        return 'M'


def KNN(train,train_label, test, test_label):
    model = KNeighborsClassifier(n_neighbors=10)
    model.fit(train, train_label)
    prediction = model.predict(test)
    acc = accuracy_score(prediction, test_label)
    cm = confusion_matrix(prediction, test_label)

    sns.heatmap(cm, annot=True, cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title('KNN')
    plt.show()

    return acc, cm

def SVM(train,label, test, test_label):
    model = SVC(C=5.0)
    model.fit(train, label)
    prediction = model.predict(test)
    acc = accuracy_score(prediction, test_label)
    cm = confusion_matrix(prediction, test_label)

    sns.heatmap(cm, annot=True, cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title('SVM')
    plt.show()

    return acc, cm


def GNB(train,label, test, test_label):
    model = GaussianNB()
    model.fit(train, label)
    prediction = model.predict(test)
    acc = accuracy_score(prediction, test_label)
    cm = confusion_matrix(prediction, test_label)

    sns.heatmap(cm, annot=True, cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title('GNB')
    plt.show()

    return acc, cm


# Discriminate observed sample's category
def Discriminate(data_train, label_train, data_test, label_test):
    g1_mean = 0; g2_mean =0; n_g1 = 0; n_g2 = 0
    for i in range(data_train.shape[0]):
        if label_train[i] == 'B':
            g1_mean = g1_mean + data_train[i, :]
            n_g1 = n_g1 + 1
        elif label_train[i] == 'M':
            g2_mean = g2_mean + data_train[i, :]
            n_g2 = n_g2 + 1

    g1_mean = g1_mean/n_g1; g2_mean = g2_mean/n_g2
    covariance = np.cov(data_train.transpose())

    result_MD = []
    for u in range(data_test.shape[0]):
        x = data_test[u, :]
        result_MD.append(Mahalanobis_D(x, g1_mean, g2_mean, covariance))

    acc = accuracy_score(result_MD, label_test)
    cm = confusion_matrix(result_MD, label_test)

    sns.heatmap(cm, annot=True, cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title('Mahalanobis_D')
    plt.show()

    return acc, cm
