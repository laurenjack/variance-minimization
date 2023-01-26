import numpy as np

from domain import *


def split_data(full_images, full_labels, dp: DatasetParameters, is_stratified=False):
    num_classes = int(np.max(full_labels) + 1)
    n_data, num_pixels = full_images.shape
    n = dp.n
    if n is None:
        n = n_data
    assert dp.val_n < n
    train_n = n - dp.val_n
    assert train_n % 2 == 0
    assert dp.m % 2 == 0

    # Adjust now to the number of models
    train_n_per_m = train_n // 2

    # Shuffle the data for two purposes:
    # 1) Randomize what is in training and randomize what is in validation
    # 2) Randomize which model gets which training examples
    all_indices = np.arange(n_data)
    np.random.shuffle(all_indices)
    shuffled_images = full_images[all_indices]
    shuffled_labels = full_labels[all_indices]
    if is_stratified:
        indices = []
        # For each class get every index where it occurs
        per_class_indices_list = []
        for c in range(num_classes):
            is_c = np.equal(shuffled_labels, c)
            indices_of_c = np.argwhere(is_c).flatten()
            per_class_indices_list.append(indices_of_c)
        # Now take from each class one by one
        examples_per_class = n // num_classes
        for i in range(examples_per_class):
            for c in range(num_classes):
                index = per_class_indices_list[c][i]
                indices.append(index)

        images = shuffled_images[indices]
        labels = shuffled_labels[indices]
    else:
        images = shuffled_images[:n]
        labels = shuffled_labels[:n]

    train_images, val_images = images[:train_n], images[train_n:]
    train_labels, val_labels = labels[:train_n], labels[train_n:]
    # One hot encoding of the labels
    train_labels_hot = np.zeros((train_n, num_classes))
    train_indices = np.arange(train_n)
    train_labels_hot[train_indices, train_labels] = 1.0

    # Take a different sample per model. Each model trains on 50% of the training data. In the loop below,
    # we shuffle the data, and then the following happens:
    # 1) Even indexed models get the first half of the data
    # 2) Odd indexed models get the second half.
    # This ensures that every data point occurs in the entire training set with equal frequency, and that for
    # a given model, there is always at least one other model that is trained on entirely different data.
    re_samples = dp.m // 2
    train_x = []
    train_y = []
    train_y_hot = []
    for i in range(re_samples):
        np.random.shuffle(train_indices)
        first_indices = train_indices[0: train_n_per_m]
        second_indices = train_indices[train_n_per_m:]
        train_x.append(train_images[first_indices])
        train_x.append(train_images[second_indices])
        train_y.append(train_labels[first_indices])
        train_y.append(train_labels[second_indices])
        train_y_hot.append(train_labels_hot[first_indices])
        train_y_hot.append(train_labels_hot[second_indices])

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_y_hot = np.array(train_y_hot)
    val_x = np.array([val_images] * dp.m)
    val_y = np.array([val_labels] * dp.m)
    # train_x = train_images.reshape(2, train_n_per_m, num_pixels)
    # train_y = train_labels.reshape(2, train_n_per_m)
    # train_y_hot = train_labels_hot.reshape(2, train_n_per_m, num_classes)
    # val_x = np.array([val_images, val_images])
    # val_y = np.array([val_labels, val_labels])
    return DataSet(train_x, train_y, train_y_hot, val_x, val_y)