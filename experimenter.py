from domain import *
import splitter
import data_generator
import network


def create_grid(epochs, sizes, batch_size, acc_sample_size, alpha, b1, b2, epsilon, reg_list, reg_power_list):
    grid = []
    for reg in reg_list:
        for reg_power in reg_power_list:
            grid.append(HyperParameters(epochs, sizes, batch_size,acc_sample_size, alpha, b1, b2, epsilon,
                                                    reg, reg_power))
    return grid


def run_experiments(dp, grid):
    # Create the dataset:
    full_images, full_labels = data_generator.single_relevant(100, 20, 0.5, 0.8, 0.5)
    dataset = splitter.split_data(full_images, full_labels, dp, is_stratified=True)
    for hp in grid:
        net = network.create_network(dataset, hp)
        net.run()


m = 2
n = 100
val_n = 50

epochs = 50
sizes = [20, 2]  # examples_per_class
lr = 0.3
batch_size = 20
acc_sample_size = 50
alpha = 0.01
b1 = 0.9
b2 = 0.999
epsilon = 0.00000001
reg_list = [1.0]
reg_power_list = [None]

dp = DatasetParameters(m, n, val_n)
grid = create_grid(epochs, sizes, batch_size, acc_sample_size, alpha, b1, b2, epsilon, reg_list, reg_power_list)
run_experiments(dp, grid)
