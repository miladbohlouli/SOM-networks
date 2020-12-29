from mid_interface import *
import pickle
from SOM import SOM

#   Importing the dataset
num_classes = 16
x_train = read_data(path="data/HighDim.txt", labels=False)
saving_model = "model/2.4"
load = True
y_train = np.zeros((1024, 1))
length = int(len(x_train) / num_classes)
for i in range(num_classes):
    y_train[i * length:(i + 1) * length, 0] += i

if load:
    part1 = pickle.load(open(saving_model + "/model.pickle", 'rb'))
    new_data = part1.reduce_dimensions(x_train)
    plot_data(new_data, y_train)

else:

    #   Reducing the dimensions
    part1 = SOM(num_d=x_train.shape[1], size=[10, 10], radius=3, shape="hexagon",
                learning_rate=0.1, learning_policy="exponential", weight_policy="exponential",
                width_policy="exponential")
    part1.train(x_train, num_epochs=20)
    new_data = part1.reduce_dimensions(x_train)
    plot_data(new_data, y_train)
    pickle.dump(part1, open(saving_model + "/model.pickle", 'wb'))