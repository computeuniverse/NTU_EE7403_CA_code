import numpy as np
from performance import Confusion_matrix
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy import matmul, inner
from numpy.linalg import eigh
from math import pow


def PCA(data_train, label_train):
    # transpose the matrix
    mean = np.mean(data_train, axis=0)
    data_train = data_train - mean
    data_train = data_train.transpose()

    # compute covariance matrix
    cov = np.cov(data_train)
    # cov = cov + 0.01 * np.eye(30)

    # compute Eigenvalue and Eigenvectors
    E, V = eigh(cov)
    dis_vector = []
    for i in range(len(E)):
        if E[i] > 10:
            dis_vector.append(V[:, i])
    print('Discriminate vectors:', len(dis_vector))
    print('特征值：', E)

    # Reduction train data dimension
    data_reduced = np.empty([len(dis_vector), data_train.shape[1]], dtype=float)

    for i in range(data_train.shape[1]):
        x = data_train[:, i]
        for j in range(len(dis_vector)):
            data_reduced[j, i] = matmul(dis_vector[j].reshape(1, -1), x.reshape(-1, 1))

    return data_reduced, len(dis_vector)

