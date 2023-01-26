import numpy as np
import mnist
import data_generator
import splitter
from scipy.stats import norm

from domain import *


class Network(object):

    def __init__(self, dataset: DataSet, hp: HyperParameters, ws):
        self.dataset = dataset
        self.hp = hp
        self.ws = ws

    def feed_forward(self, x):
        a = x
        activations = [a]
        for w in self.ws[:-1]:
            z = np.matmul(a, w)
            a = np.maximum(z, 0.0)
            activations.append(a)
        z = np.matmul(a, self.ws[-1])

        # Apply the softmax
        ez = np.exp(z)
        m, batch_size, _ = ez.shape
        denominator = np.sum(ez, axis=2).reshape(m, batch_size, 1)
        a = ez / denominator
        activations.append(a)
        return activations

    def accuracy(self, x, y):
        sample_size = x.shape[1]
        a = self.feed_forward(x)[-1]

        # Find prediction (axis 2 is the class axis)
        prediction = np.argmax(a, axis=2)
        # prediction_pooled = np.greater_equal(np.sum(prediction, axis=0), self.ds.m // 2)
        num_correct = np.sum(np.equal(prediction, y).astype(int), axis=1)
        # num_correct_pooled = np.sum(np.equal(prediction_pooled, y).astype(int), axis=0)
        return num_correct / sample_size

    def run(self):
        """
        Train and then evaluate the accuracy on the validation set.
        """
        # Unpack dataset
        train_x = self.dataset.train_x
        train_y = self.dataset.train_y
        train_y_hot = self.dataset.train_y_hot
        val_x = self.dataset.val_x
        val_y = self.dataset.val_y
        m, n, d = train_x.shape
        _, n_val, _ = val_x.shape
        train_indices = np.arange(n)
        val_indices = np.arange(n_val)

        # Unpack hyper parameters
        epochs = self.hp.epochs
        acc_sample_size = self.hp.acc_sample_size
        batch_size = self.hp.batch_size
        alpha = self.hp.alpha
        b1 = self.hp.b1
        b2 = self.hp.b2
        epsilon = self.hp.epsilon
        reg = self.hp.reg

        # Adam
        t = 1
        mt = [np.zeros(w.shape) for w in self.ws]
        vt = [np.zeros(w.shape) for w in self.ws]

        for e in range(epochs):
            # Sample each set to evaluate accuracy
            train_sample = np.random.choice(train_indices, acc_sample_size // 2, replace=False)
            train_acc = self.accuracy(train_x[:, train_sample, :], train_y[:, train_sample])
            val_sample = np.random.choice(val_indices, acc_sample_size, replace=False)
            val_acc = self.accuracy(val_x[:, val_sample, :], val_y[:, val_sample])
            print(e, "train acc:", train_acc, "val_acc", val_acc)

            np.random.shuffle(train_indices)

            # Create the batches
            for i in range(0, n, batch_size):

                train_batch = train_indices[i:i+batch_size]
                x = train_x[:, train_batch, :]
                y_hot = train_y_hot[:, train_batch, :]
                # We're just randomly choosing validation examples for simplicity (instead of sequential batching)
                val_batch = np.random.choice(val_indices, batch_size, replace=False)
                x_val = val_x[:, val_batch, :]

                bias_activations = self.feed_forward(x)
                ab = bias_activations.pop()
                bias_delta = ab - y_hot  # (m, batch_size, d_out)

                var_activations = self.feed_forward(x_val)
                av = var_activations.pop()
                var_delta = 0.5 * (av[0] - av[1])  # (batch_size, d_out) where d_out == num_classes
                var_delta = np.array([var_delta] * m)  # (m, batch_size, d_out)

                # Where ab has shape (m, batch_size, d_in), and w has shape (m, d_in, d_out)
                for ab, av, w, l in zip(bias_activations[::-1], var_activations[::-1], self.ws[::-1], range(len(self.ws))[::-1]):
                    # Bias back-prop
                    dw = np.matmul(ab.transpose(0, 2, 1), bias_delta) / batch_size
                    da = np.where(ab > 0, 1.0, 0.0)  # (m, batch_size, d_in)
                    bias_delta = np.matmul(bias_delta, w.transpose(0, 2, 1)) * da

                    # Variance back-prop
                    _, d_in, d_out = w.shape
                    # var_delta *= 1.0
                    # aw = av.reshape(m, batch_size, d_in, 1) * w.reshape(m, 1, d_in, d_out)
                    # aw_with_grad = aw * var_delta.reshape(m, batch_size, 1, d_out)
                    # grad = aw_with_grad[0] - aw_with_grad[1]
                    # # aw_delta = aw[0] - aw[1]  # (batch_size, d_int, d_out)
                    # # grad = np.abs(aw_delta.reshape(1, batch_size, d_in, d_out) * var_delta[0].reshape(batch_size, 1, d_out))
                    # dc = np.sum(grad, axis=0).reshape(1, d_in, d_out) / batch_size
                    # w_mag = np.sum(w ** 2, axis=0).reshape(1, d_in, d_out) ** 0.5
                    # dw_var = dc * w / w_mag  # (m, d_in, d_out)
                    # da_var = np.where(av > 0, 1.0, 0.0)  # (m, batch_size, d_in)
                    # var_delta = np.matmul(var_delta, w.transpose(0, 2, 1)) * da_var  # (m, batch_size, d_in)


                    g = dw # + reg * (1 - p_value) * w  # + 100000.0 * dw_var

                    # Adam
                    mt[l] = b1 * mt[l] + (1 - b1) * g
                    mt_hat = mt[l] / (1 - b1 ** t)
                    vt[l] = b2 * vt[l] + (1 - b2) * g ** 2
                    vt_hat = vt[l] / (1 - b2 ** t)

                    w -= alpha * mt_hat / (vt_hat ** 0.5 + epsilon)

                    w_mean = np.mean(w, axis=0).reshape(1, d_in, d_out)
                    z_value = w / (np.mean((w - w_mean) ** 2, axis=0).reshape(1, d_in, d_out) ** 0.5 + epsilon)
                    p_value = norm.cdf(np.abs(z_value))
                    w -= reg * (1 - p_value) * w
                    t += 1

                    # SGD
                    # w -= lr * g
        print(self.ws)


def create_weights(sizes, m):
    ws = []
    for d_in, d_out in zip(sizes[:-1], sizes[1:]):
        w = weight_init(d_in, d_out, m)
        ws.append(w)
    return ws


def weight_init(d_in, d_out, m):
    # the_same = np.random.randn(d_in, d_out) / d_in ** 0.5
    # return np.array([the_same, the_same])
    return np.random.randn(m, d_in, d_out) / d_in ** 0.5


def create_network(dataset: DataSet, hp: HyperParameters):
    m = dataset.train_x.shape[0]
    ws = create_weights(hp.sizes, m)
    return Network(dataset, hp, ws)
