from mid_interface import *
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

########################################################################################################
#   This is a class of SOM(Self Organizing Maps), The goal is clustering a dataset.
#       parameters:
#           1. num_d:number of dimensions
#           2. size: size of the grid
#           3. radius: radius of the shape used for neighbours(set to None if no need for boundaries)
#           4. shape: defines the shape of the neighbourhood. Square, circle, hexagon
#           5. weight_policy: policy for distance calculation among nodes:
#               gaussian: sigma should be defined in this case
#               exponential: k should be defined
#           6. learning_policy: the policy to update the learning rate
#               linear
#               exponential
#               usual
#               constant
#           7. width_policy: the policy to update the width of neighbourhood in learning phase
#               linear
#               exponential
#               constant
########################################################################################################
class SOM:
    def __init__(self, num_d, size, radius, shape="square", weight_policy="exponential", sigma=20, k=0.1,
                 learning_policy="exponential", learning_rate=0.4, width_policy="exponential"):
        self.name = None
        self.weights = None
        self.size = size
        self.num_d = num_d
        self.shape = shape
        self.__sigma = sigma
        self.__initial_radius = radius
        self.__k = k
        self.__weight_policy = weight_policy
        self.__learning_policy = learning_policy
        self.__width_policy = width_policy
        self.__initial_learning_rate = learning_rate

        #   These are some assertions to check if the inputs are correct
        assert radius < size[0] and radius < size[1]
        assert self.shape == "square" or self.shape == "circle" or self.shape == "hexagon"
        assert self.__weight_policy == "gaussian" or self.__weight_policy == "exponential"
        assert self.__learning_policy == "linear" or self.__learning_policy == "exponential" \
               or self.__learning_policy == "usual" or self.__learning_policy == "constant"
        assert self.__width_policy == "linear" or self.__width_policy == "exponential" \
               or self.__width_policy == "constant"

    ###########################################################################
    #   This method trains the saved_images to find the weights
    ###########################################################################
    def train(self, x_data, num_epochs=100):
        assert x_data.shape[1] == self.num_d
        np.random.seed(25)
        self.weights = np.random.rand(self.size[0], self.size[1], self.num_d)

        scalar = MinMaxScaler()
        x_data = scalar.fit_transform(x_data)

        for epoch in range(num_epochs):

            learning_rate = self.__learning_rate(epoch, num_epochs)
            if self.__weight_policy != "constant":
                neighbours_dict = self.__neighbour_dict(epoch, num_epochs)

            elif epoch == 0:
                neighbours_dict = self.__neighbour_dict(epoch, num_epochs)

            if epoch % 10 == 0:
                print("Epoch(%d/%d)" %(epoch + 1, num_epochs))
            for point in x_data:
                mach = np.unravel_index(np.argmax(np.sum(np.power(point - self.weights, 2), axis=-1)), self.size)

                for neighbour in neighbours_dict[tuple(mach)]:
                    self.weights[neighbour] += learning_rate * \
                                               self.__neighbour_strength(mach, neighbour) * \
                                               (point - self.weights[neighbour])

    def cluster(self, x_data, plot=True):
        clus = dict()
        counter = 0
        pred_y = np.zeros(x_data.shape[0])
        frequency = np.zeros(self.size, dtype=np.int)

        for i, point in enumerate(x_data):
            mach = np.unravel_index(np.argmin(np.sum(np.power(point - self.weights, 2), axis=-1)), self.size)
            frequency[mach] += 1
            if mach not in clus:
                counter += 1
                clus[mach] = counter
            pred_y[i] = clus[mach]

        if plot:
            plt.figure()
            plt.suptitle("Results of clustering using shape %s and radius %d" % (str(self.shape), self.__initial_radius))
            plt.subplot(1, 2, 1)
            plt.xticks([]), plt.yticks([])
            for clus in np.unique(pred_y):
                plt.scatter(x_data[pred_y == clus, 0], x_data[pred_y == clus, 1], s=3)

            plt.subplot(1, 2, 2)
            sns.heatmap(frequency, annot=True, fmt="d", linewidths=.5)

        return pred_y

    ###########################################################################
    #   This function is used to reduce the dimensions of the x_data,
    #       The returned ndarray contains a data in 2 dimensions.
    ###########################################################################
    def reduce_dimensions(self, x_data):

        new_data = np.zeros((x_data.shape[0], 2))
        for i, point in enumerate(x_data):
            mach = np.unravel_index(np.argmin(np.sum(np.power(point - self.weights, 2), axis=-1)), self.size)
            new_data[i] = mach

        return new_data

    ###########################################################################
    #   In order to avoid double calculation we create a dictionary of the
    #       neighbours of each node in the grid to use in each epoch
    ###########################################################################
    def __neighbour_dict(self, current_iteration, total_iterations):
        neighbour_dict = dict()
        for i in range(0, self.size[0]):
            for j in range(0, self.size[1]):
                neighbour_dict[(i, j)] = self.__find_neighbours((i, j),
                                                                self.__neighbourhood_width(current_iteration, total_iterations),
                                                                self.shape)
        return neighbour_dict

    ###########################################################################
    #   Given the center and the shape of the finds the neighbours
    #       and returns as a list of tuples
    ###########################################################################
    def __find_neighbours(self, center, radius, shape):
        neighbours = []
        assert center[0] < self.size[0] and center[1] < self.size[1]

        if radius is None:
            for i in range(self.size[0]):
                for j in range(self.size[1]):
                    neighbours.append((i, j))

        else:
            for i in range(max(center[0] - int(radius), 0), min(center[0] + int(radius) + 1, self.size[0])):
                for j in range(max(center[1] - int(radius), 0), min(center[1] + int(radius) + 1, self.size[1])):

                    if shape == "square":
                        neighbours.append((i, j))

                    elif shape == "circle":
                        if np.linalg.norm([center[0] - i, center[1] - j]) <= radius ** 2:
                            neighbours.append((i, j))

                    elif shape == "hexagon":
                        cons = 3 ** 0.5
                        if abs(i - center[0]) <= cons / 2 * radius and \
                            abs(cons * (j - center[1]) + (i - center[0])) <= cons * radius and \
                            abs(cons * (j - center[1]) - (i - center[0])) <= cons * radius:
                            neighbours.append((i, j))
        return neighbours

    ###########################################################################
    #   This function calculates three types of distances among two points,
    #       There are three neighbour strength policies:
    #           1. linear decay
    #           2. gaussian decay: sigma has to be specified
    #           3. exponential: k has to be specified
    ###########################################################################
    def __neighbour_strength(self, point1, point2):
        point1 = np.array(point1)
        point2 = np.array(point2)
        euclidean_distance = np.sum(np.power(np.array(point1) - np.array(point2), 2)) ** 0.5
        if self.__weight_policy == "gaussian":
            return np.exp(-euclidean_distance ** 2 / (2 * self.__sigma ** 2))
        elif self.__weight_policy == "exponential":
            return np.exp(-self.__k * euclidean_distance)

    ###########################################################################
    #   This function applies the learning_rate updating policy
    ###########################################################################
    def __learning_rate(self, current_iteration, total_iterations):
        if self.__learning_policy == "usual":
            return 2 / (3 + current_iteration)

        elif self.__learning_policy == "linear":
            return self.__initial_learning_rate * (1 - current_iteration/total_iterations)

        elif self.__learning_policy == "exponential":
            return self.__initial_learning_rate * np.exp( - current_iteration / total_iterations)

        elif self.__learning_policy == "constant":
            return self.__initial_learning_rate

    ###########################################################################
    #   This function applies the neighbourhood width updating policy
    ###########################################################################
    def __neighbourhood_width(self, current_iteration, total_iterations):
        if self.__initial_radius is None:
            return None

        elif self.__width_policy == "linear":
            return self.__initial_radius * (1 - current_iteration/total_iterations)

        elif self.__width_policy == "exponential":
            return self.__initial_radius * np.exp(-current_iteration/total_iterations)

        elif self.__width_policy == "constant":
            return self.__initial_radius

    @staticmethod
    def purity(ground_truth, predictions):
        y_voted_labels = np.zeros(ground_truth.shape)
        labels = np.unique(ground_truth)
        ordered_labels = np.arange(labels.shape[0])
        for k in range(labels.shape[0]):
            ground_truth[ground_truth == labels[k]] = ordered_labels[k]
        labels = np.unique(ground_truth)
        bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

        for cluster in np.unique(predictions):
            hist, _ = np.histogram(ground_truth[predictions == cluster], bins=bins)
            winner = np.argmax(hist)
            y_voted_labels[predictions == cluster] = winner

        return accuracy_score(ground_truth, y_voted_labels)
