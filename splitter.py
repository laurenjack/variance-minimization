import numpy as np


def split_data(full_images, full_labels, val_n, n=None, is_stratified=False):
    num_classes = int(np.max(full_labels) + 1)
    n_data, num_pixels = full_images.shape
    if n is None:
        n = n_data
    assert val_n < n
    train_n = n - val_n
    assert train_n % 2 == 0

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

    # Reshape adding the model dimension, the same validation data is applied to both models so its
    # stacked to be the same per model
    train_x = train_images.reshape(2, train_n_per_m, num_pixels)
    train_y = train_labels.reshape(2, train_n_per_m)
    train_y_hot = train_labels_hot.reshape(2, train_n_per_m, num_classes)
    val_x = np.array([val_images, val_images])
    val_y = np.array([val_labels, val_labels])
    return train_x, train_y, train_y_hot, val_x, val_y