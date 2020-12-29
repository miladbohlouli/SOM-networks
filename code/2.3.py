from mid_interface import *
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

num_classes = 16
x_train = read_data(path="data/HighDim.txt", labels=False)

y_train = np.zeros((1024, 1))
length = int(len(x_train) / num_classes)
for i in range(num_classes):
    y_train[i * length:(i + 1) * length, 0] += i

pca = PCA(n_components=2)
x_train = pca.fit_transform(x_train)
labels = []

for label in np.unique(y_train[:, 0]):
    plt.scatter(x_train[y_train[:, 0] == label, 0], x_train[y_train[:, 0] == label, 1])
    labels.append(str(int(label)))
plt.legend(labels, loc=1)
plt.grid(True, alpha=0.2)
plt.show()

