from separate_dataset import saparete_dataset
from LDA import LDA
from PCA import PCA
from classifier import Discriminate, KNN, GNB, SVM
import csv
import numpy as np
import matplotlib.pyplot as plt
from regulation import z_score_normalize, min_max


csvfile = open('./Cancer_Data.csv', 'r')
reader = csv.reader(csvfile)
counter = 0
label = []
data = np.empty([569, 30], dtype=float)

for i in reader:
    label.append(i[1])
    for j in range(30):
        data[counter, j] = float(i[j + 2])
    counter = counter + 1

# data = z_score_normalize(data)
# data = min_max(data)


train_data, train_label, test_data, test_label = saparete_dataset(data=data, label=label, n_test=100)
print(Discriminate(train_data, train_label, test_data, test_label))
print(KNN(train_data, train_label, test_data, test_label))
print(GNB(train_data, train_label, test_data, test_label))



data_reduced, dim = LDA(data, label)
# data_reduced, dim = PCA(data, label)
# data_reduced = data_reduced.transpose()


g1 = np.empty([0, dim]); g2 = np.empty([0, dim])
for i in range(data_reduced.shape[0]):
    if label[i] == 'B':
        g1 = np.vstack([g1, data_reduced[i, :]])
    if label[i] == 'M':
        g2 = np.vstack([g2, data_reduced[i, :]])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(g1[:, dim-1], g1[:, dim-2], g1[:, dim-3],c='r', s=5)
ax.scatter(g2[:, dim-1], g2[:, dim-2], g2[:, dim-3],c='b', s=5)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

train_data, train_label, test_data, test_label = saparete_dataset(data=data_reduced, label=label, n_test=100)
print(Discriminate(train_data, train_label, test_data, test_label))
print(KNN(train_data, train_label, test_data, test_label))
print(GNB(train_data, train_label, test_data, test_label))

