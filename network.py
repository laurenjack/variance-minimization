import numpy as np
import mnist
import data_generator
import splitter


epochs = 100
sizes = [20, 2]  # [784, 30, 10]
lr = 0.3
batch_size = 20
n = 320
val_n = 160
acc_sample_size = 80


def create_weights(sizes):
    ws = []
    for d_in, d_out in zip(sizes[:-1], sizes[1:]):
        w = weight_init(d_in, d_out)
        ws.append(w)
    return ws


def weight_init(d_in, d_out):
    return np.random.randn(2, d_in, d_out) / d_in ** 0.5


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
# full_images, full_labels = mnist.get_train()
full_images, full_labels = data_generator.single_relevant(320, 20, 0.1, 0.9, 0.5)
# num_pixels = full_images.shape[1]
train_x, train_y, train_y_hot, val_x, val_y = splitter.split_data(full_images, full_labels, val_n, n=n)
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

        bias_activations = feed_forward(ws, x)
        ab = bias_activations.pop()
        bias_delta = ab - y_hot  # (m, batch_size, d_out)

        var_activations = feed_forward(ws, x_val)
        av = var_activations.pop()
        var_delta = 0.5 * (av[0] - av[1])  # (batch_size, d_out) where d_out == num_classes
        var_delta = np.array([var_delta, var_delta])  # (2, batch_size, d_out)

        # Where ab has shape (2, batch_size, d_in),  and w has shape (2, d_in, d_out)
        for ab, av, w in zip(bias_activations[::-1], var_activations[::-1], ws[::-1]):
            # Bias back-prop
            dw = np.matmul(ab.transpose(0, 2, 1), bias_delta) / batch_size
            da = np.where(ab > 0, 1.0, 0.0)  # (m, batch_size, d_in)
            bias_delta = np.matmul(bias_delta, w.transpose(0, 2, 1)) * da

            # Variance back-prop
            _, d_in, d_out = w.shape
            var_delta *= 50.0
            aw = av.reshape(2, batch_size, d_in, 1) * w.reshape(2, 1, d_in, d_out)
            aw_with_grad = aw * var_delta.reshape(2, batch_size, 1, d_out)
            grad = np.abs(aw_with_grad[0] - aw_with_grad[1])
            # aw_delta = aw[0] - aw[1]  # (batch_size, d_int, d_out)
            # grad = np.abs(aw_delta.reshape(1, batch_size, d_in, d_out) * var_delta[0].reshape(batch_size, 1, d_out))
            dc = np.sum(grad, axis=0).reshape(1, d_in, d_out) / batch_size
            w_mag = np.sum(w ** 2, axis=0).reshape(1, d_in, d_out) ** 0.5
            dw_var = dc * w / w_mag  # (2, d_in, d_out)
            da_var = np.where(av > 0, 1.0, 0.0)  # (2, batch_size, d_in)
            var_delta = np.matmul(var_delta, w.transpose(0, 2, 1)) * da_var  # (2, batch_size, d_in)

            w -= lr * (1.0 * dw + 1.0 * dw_var)   #  5.0 * w / batch_size

# print(ws)
