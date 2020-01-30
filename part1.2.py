from mid_interface import *
from SOM import SOM
import pickle
from matplotlib import pyplot as plt

saving_dir = "saved_images/1.1.2"
saving_model = "model/1.1.2"
radius_list = [9, 2, 4, 6]
x_data, y_data = read_data()
load = True
purities = []

if load:
    for i, radius in enumerate(radius_list):
        part1 = pickle.load(open(saving_model + "/" + str(radius) + ".pickle", 'rb'))
        predictions = part1.cluster(x_data, plot=True)
        plt.savefig(saving_dir + "/" + str(radius) + ".png")
        purities.append(SOM.purity(y_data, predictions))


else:
    for i, radius in enumerate(radius_list):
        part1 = SOM(num_d=x_data.shape[1], size=[10, 10], radius=radius, shape="hexagon", learning_rate=0.1,
                    learning_policy="exponential", weight_policy="exponential", width_policy="constant")
        part1.train(x_data, num_epochs=500)
        predictions = part1.cluster(x_data, plot=True)
        purities.append(SOM.purity(y_data, predictions))
        pickle.dump(part1, open(saving_model + "/" + str(radius) + ".pickle", 'wb'), )


plt.figure()
plt.title("Comparing the purity of different neighbourhood radius values")
plt.xlabel("radius value"), plt.ylabel("purity")
plt.grid(True, alpha=0.5)
plt.scatter(radius_list, purities, color="purple")
plt.savefig(saving_dir + "/" + "results.jpg")
plt.show()

