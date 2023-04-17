import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def Confusion_matrix(pred, real):
    C = confusion_matrix(real, pred, labels=[1, 2, 3])
    print(C)
    plt.matshow(C, cmap=plt.cm.Greens)
    plt.colorbar()
    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[i, j], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    plt.show()


def plot_hist(data):
    # plot every feature's distribution
    plt.hist(x=list(data[0, :]), histtype='stepfilled', bins=50, label='$feature 1$')
    plt.hist(x=list(data[1, :]), histtype='stepfilled', bins=50, label='$feature 2$')
    plt.hist(x=list(data[2, :]), histtype='stepfilled', bins=50, label='$feature 3$')
    plt.hist(x=list(data[3, :]), histtype='stepfilled', bins=50, label='$feature 4$')
    plt.legend()
    plt.show()

