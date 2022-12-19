import numpy as np

n = 20
batch_size = 10
epochs = 100
lr = 0.3
error_x0 = 1.0
error_x1 = 3.0


def gen_problem(n, error_x0, error_x1):
    x = np.random.randn(n, 3, 2)
    x[:, :, 1] /= error_x1
    y = x[:,:,0] + error_x0 * np.random.randn(n, 3)
    x_train = x[:, 0:2, :]
    x_test = x[:, 2, :].reshape((n, 1, 2))
    y_train = y[:, 0:2]
    y_test = y[:, 2]
    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = gen_problem(n, error_x0, error_x1)
b = np.random.randn(1, 2, 2)

indices = np.arange(n)
for e in range(epochs):
    train_batch = np.random.choice(indices, batch_size, replace=False)
    x_batch = x_train[train_batch]
    y_batch = y_train[train_batch]
    print(b)
    y_hat_train = np.sum(b * x_batch, axis=2)
    bias = (y_hat_train - y_batch).reshape((batch_size, 2, 1))
    bias_grad = np.mean(bias * x_batch, axis=0).reshape(1, 2, 2)

    test_batch = np.random.choice(indices, batch_size, replace=False)
    x_test_batch = x_test[test_batch]
    bx = b * x_test_batch
    # From here we go to 2 dimensions (n, d), the model dimension is collapsed
    y_hat_test = np.sum(bx, axis=2)
    y_hat_delta = (y_hat_test[:, 0] - y_hat_test[:, 1]).reshape(batch_size, 1)
    bx_delta = (bx[:, 0, :] - bx[:, 1, :]).reshape(batch_size, 2)
    # Back to 3 dimensions (n, m, d)
    per_point_grad = np.abs(y_hat_delta * bx_delta)
    c_delta = 0.5 * np.mean(per_point_grad, axis=0).reshape(1, 1, 2)
    b_mag = np.sum(b ** 2, axis=1).reshape(1, 1, 2) ** 0.5
    bvar_grad = c_delta * b / b_mag
    b -= lr * (0.3 * bias_grad + bvar_grad)

print('b OLS')
print(np.sum(y_train.reshape(n, 2, 1) * x_train, axis=0) / np.sum(x_train ** 2, axis=0))



