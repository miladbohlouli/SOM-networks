import numpy as np
from matplotlib import pyplot as plt

def read_data(path="data/2D.txt", plot=False, labels=True):
    data = np.loadtxt(path)
    if plot:
        for clus in np.unique(data[:, -1]):
            plt.scatter(data[data[:, -1] == clus, 0], data[data[:, -1] == clus, 1], s=3)
    plt.show()
    if labels == True:
        return data[:, 0:-1], data[:, -1]
    else:
        return data

def plot_data(new_data):
    for item in np.unique(new_data, axis=0):
        indexes = np.logical_and(new_data[:, 0] == item[0], new_data[:, 1] == item[1])
        plt.scatter(new_data[indexes, 0], new_data[indexes, 1])
    plt.grid(True, alpha=0.2)
    plt.show()

def plot_data(new_data, y_train):
    labels = []
    for item in np.unique(y_train):
        labels.append(str(int(item)))
        plt.scatter(new_data[y_train[:, 0] == item, 0], new_data[y_train[:, 0] == item, 1])
    plt.grid(True, alpha=0.2)
    plt.legend(labels)
    plt.show()
