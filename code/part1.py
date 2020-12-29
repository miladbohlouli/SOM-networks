from mid_interface import *
from SOM import SOM
import pickle
from matplotlib import pyplot as plt

saving_dir = "saved_images/1.1"
saving_model = "model/1.1"
shapes = ["square", "circle", "hexagon"]
x_data, y_data = read_data()
load = False
purities = []

if load:
    for i, shape in enumerate(shapes):
        part1 = pickle.load(open(saving_model + "/" + shape + ".pickle", 'rb'))
        predictions = part1.cluster(x_data, plot=True)
        plt.savefig(saving_dir + "/" + shape + ".png")
        purities.append(SOM.purity(y_data, predictions))


else:
    for i, shape in enumerate(shapes):
        part1 = SOM(num_d=x_data.shape[1], size=[10, 10], radius=9, shape=shape, learning_rate=0.1,
                    learning_policy="exponential", weight_policy="exponential", width_policy="exponential")
        part1.train(x_data, num_epochs=500)
        predictions = part1.cluster(x_data, plot=True)
        purities.append(SOM.purity(y_data, predictions))
        pickle.dump(part1, open(saving_model + "/" + shape + ".pickle", 'wb'), )


plt.figure()
plt.title("Comparing the purity of different neighbourhood shapes")
plt.xlabel("shape type"), plt.ylabel("purity")
plt.grid(True, alpha=0.5)
plt.scatter(shapes, purities, color="purple")
plt.savefig(saving_dir + "/" + "results.jpg")
plt.show()

