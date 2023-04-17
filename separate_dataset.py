
import numpy as np
import random

def saparete_dataset(data, label, n_test=100):
    n_feature = data.shape[1]
    random.seed(2023)
    test_samples = random.sample(range(0, 568), n_test)
    data_train = np.empty([0, n_feature], dtype=float)
    data_test = np.empty([0, n_feature], dtype=float)
    label_train = []
    label_test = []

    for i in range(569):
        if i in test_samples:
            data_test = np.vstack([data_test, data[i, :]])
            label_test.append(label[i])
        else:
            data_train = np.vstack([data_train, data[i, :]])
            label_train.append(label[i])

    return data_train, label_train, data_test, label_test
# #
# train_data, train_label, test_data, test_label = saparete_dataset(n_test=100)
# print(test_data.shape)
# print(train_data.shape)