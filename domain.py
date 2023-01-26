class DatasetParameters(object):
    def __init__(self, m, n, val_n):
        self.m = m
        self.n = n
        self.val_n = val_n


class HyperParameters(object):
    def __init__(self, epochs, sizes, batch_size, acc_sample_size, alpha, b1, b2, epsilon, reg, reg_power):
        self.epochs = epochs
        self.sizes = sizes
        self.batch_size = batch_size
        self.acc_sample_size = acc_sample_size
        self.alpha = alpha
        self.b1 = b1
        self.b2 = b2
        self.epsilon = epsilon
        self.reg = reg
        self.reg_power = reg_power


class DataSet(object):

    def __init__(self, train_x, train_y, train_y_hot, val_x, val_y):
        self.train_x = train_x
        self.train_y = train_y
        self.train_y_hot = train_y_hot
        self.val_x = val_x
        self.val_y = val_y
