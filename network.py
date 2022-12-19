import numpy as np
import mnist

num_classes = 10
m = 2
epochs = 30
sizes = [784, 30, 10]
lr = 0.01
batch_size = 50
val_n = 10000
acc_sample_size = 1000


def split_data(full_images, full_labels):
    n, num_pixels = full_images.shape
    assert val_n < n
    train_n = n - val_n
    assert train_n % m == 0

    # Adjust now to the number of models
    train_n_per_m = train_n // m

    # Shuffle the data for two purposes:
    # 1) Randomize what is in training and randomize what is in validation
    # 2) Randomize which model gets which training examples
    indices = np.arange(n)
    np.random.shuffle(indices)
    full_images = full_images[indices]
    full_labels = full_labels[indices]
    train_images, val_images = full_images[:train_n], full_images[train_n:]
    train_labels, val_labels = full_labels[:train_n], full_labels[train_n:]
    # One hot encoding of the labels
    train_labels_hot = np.zeros((train_n, num_classes))
    train_indices = np.arange(train_n)
    train_labels_hot[train_indices, train_labels] = 1.0

    # Reshape adding the model dimension, the validation data is applied to both models so its model dimension is just
    # 1 rather than m
    train_x = train_images.reshape(m, train_n_per_m, num_pixels)
    train_y = train_labels.reshape(m, train_n_per_m)
    train_y_hot = train_labels_hot.reshape(m, train_n_per_m, num_classes)
    val_x = val_images.reshape(1, val_n, num_pixels)
    val_y = val_labels.reshape(1, val_n)
    return train_x, train_y, train_y_hot, val_x, val_y


def create_weights(sizes):
    ws = []
    for d_in, d_out in zip(sizes[:-1], sizes[1:]):
        w = weight_init(d_in, d_out)
        ws.append(w)
    return ws


def weight_init(d_in, d_out):
    return np.random.randn(m, d_in, d_out) / d_in ** 0.5


def feed_forward(ws, x):
    a = x
    activations = [a]
    for w in ws[:-1]:
        z = np.matmul(a, w)
        a = np.maximum(z, 0.0)
        activations.append(a)
    z = np.matmul(a, ws[-1])

    # Apply the softmax
    ez = np.exp(z)
    m, batch_size, _ = ez.shape
    denominator = np.sum(ez, axis=2).reshape(m, batch_size, 1)
    a = ez / denominator
    activations.append(a)
    return activations


def accuracy(ws, x, y):
    sample_size = x.shape[1]
    a = feed_forward(ws, x)[-1]

    # Find prediction (axis 2 is the class axis)
    prediction = np.argmax(a, axis=2)
    num_correct = np.sum(np.equal(prediction, y).astype(int), axis=1)
    return num_correct / sample_size


# Prepare the data for training
full_images, full_labels = mnist.get_train()
train_x, train_y, train_y_hot, val_x, val_y = split_data(full_images, full_labels)
_, n, d = train_x.shape
_, n_val, _ = val_x.shape
train_indices = np.arange(n)
val_indices = np.arange(n_val)

# Create the network
ws = create_weights(sizes)


for e in range(epochs):
    # Sample each set to evaluate accuracy
    train_sample = np.random.choice(train_indices, acc_sample_size, replace=False)
    train_acc = accuracy(ws, train_x[:, train_sample, :], train_y[:, train_sample])
    val_sample = np.random.choice(val_indices, acc_sample_size, replace=False)
    val_acc = accuracy(ws, val_x[:, val_sample, :], val_y[:, val_sample])
    print(e, "train acc:", train_acc, "val_acc", val_acc)

    np.random.shuffle(train_indices)
    # Create the batches
    for i in range(0, n, batch_size):

        train_batch = train_indices[i:i+batch_size]
        x = train_x[:, train_batch, :]
        y_hot = train_y_hot[:, train_batch, :]
        # We're just going to randomly choose validation examples for simplicity (instead of sequential batching)
        val_batch = np.random.choice(val_indices, batch_size, replace=False)
        x_val = val_x[:, val_batch, :]

        activations = feed_forward(ws, x)
        delta = activations.pop() - y_hot  # (m, batch_size, d_out)

        # Where a has shape (m, batch_size, d_in) and w has shape (m, d_in, d_out)
        for a, w in zip(activations[::-1], ws[::-1]):
            dw = np.matmul(a.transpose(0, 2, 1), delta)
            da = np.where(a > 0, 1.0, 0.0)  # (m, batch_size, d_in)
            delta = np.matmul(delta, w.transpose(0, 2, 1)) * da
            w -= lr * dw

