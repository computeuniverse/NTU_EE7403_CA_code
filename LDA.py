import numpy as np
from performance import Confusion_matrix
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy import matmul, inner
from numpy.linalg import eigh
from math import pow


def LDA(data_train, label_train):
    # divide the dataset into 2 group
    g1 = np.empty([0, 30], dtype=float)
    g2 = np.empty([0, 30], dtype=float)

    for i in range(len(label_train)):
        if label_train[i] == 'B':
            g1 = np.vstack([g1, data_train[i, :]])
        elif label_train[i] == 'M':
            g2 = np.vstack([g2, data_train[i, :]])

    n_g1 = g1.shape[0]
    n_g2 = g2.shape[0]

    # compute the mean vector of every group
    g1_mean = np.mean(g1, axis=0)
    g2_mean = np.mean(g2, axis=0)

    # transpose the matrix
    g1 = g1.transpose()
    g2 = g2.transpose()

    # compute S_pool
    g1_cov = np.cov(g1)
    g2_cov = np.cov(g2)
    S_pool = (n_g1 * g1_cov + n_g2 * g2_cov) / data_train.shape[0]


    # Although classifier does not need to compute discriminate vectors
    # But I do it
    G_mean = np.transpose(np.mean(data_train, axis=0))
    B = n_g1/(n_g1+n_g2)*matmul((g1_mean - G_mean).reshape(-1, 1), (g1_mean - G_mean).reshape(1, -1)) + \
        n_g2/(n_g1+n_g2)*matmul((g2_mean - G_mean).reshape(-1, 1), (g2_mean - G_mean).reshape(1, -1))

    # compute matrix W
    W1 = 0; W2 = 0
    for i in range(g1.shape[1]):
        x = g1[:, i]
        W1 = W1 + matmul((x - g1_mean).reshape(-1, 1), (x - g1_mean).reshape(1, -1))
    for i in range(g2.shape[1]):
        x = g2[:, i]
        W2 = W2 + matmul((x - g2_mean).reshape(-1, 1), (x - g2_mean).reshape(1, -1))

    W = (W1 + W2)/(n_g1+n_g2)
    E, V = eigh((matmul(inv(W), B) + 5 * np.eye(30)))
    dis_vector = []
    for i in range(len(E)):
        if E[i] > 10:
            dis_vector.append(V[:, i])
    print('Discriminate vectors:', len(dis_vector))
    # print(B)
    # print(W)
    print('特征值：', E)

    # Reduction train data dimension
    data_reduced = np.empty([data_train.shape[0], len(dis_vector)], dtype=float)

    for i in range(data_train.shape[0]):
        x = data_train[i, :]
        for j in range(len(dis_vector)):
            data_reduced[i, j] = matmul(dis_vector[j].reshape(1, -1), x.reshape(-1, 1))

    return data_reduced, len(dis_vector)

    # # Plot train data
    # g1_x = []; g2_x = []; g3_x = []
    # g1_y = []; g2_y = []; g3_y = []
    # for j in range(g1.shape[1]):
    #     x = g1[:, j]
    #     g1_x.append(inner(dis_vector[0], x.transpose()))
    #     g1_y.append(inner(dis_vector[1], x.transpose()))
    #
    # for j in range(g2.shape[1]):
    #     x = g2[:, j]
    #     g2_x.append(inner(dis_vector[0], x.transpose()))
    #     g2_y.append(inner(dis_vector[1], x.transpose()))
    #
    # for j in range(g3.shape[1]):
    #     x = g3[:, j]
    #     g3_x.append(inner(dis_vector[0], x.transpose()))
    #     g3_y.append(inner(dis_vector[1], x.transpose()))
    #
    # plt.scatter(g1_x, g1_y, color='blue', s=20, label='$group 1$')
    # plt.scatter(g2_x, g2_y, color='red', s=20, label='$group 2$')
    # plt.scatter(g3_x, g3_y, color='green', s=20, label='$group 3$')
    # plt.xlabel('dis_vector_1')
    # plt.ylabel('dis_vector_2')
    # plt.legend()
    # plt.show()




