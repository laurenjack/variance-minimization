import numpy as np

import mnist
import splitter


def test_when_mnist_stratified_40_then_2_per_row():
    full_images, full_labels = mnist.get_train()
    _, train_y, _, _, val_y = splitter.split_data(full_images, full_labels, 20, n=40, is_stratified=True)
    zero_to_nine = np.arange(10)
    expected_train_labels = np.array([zero_to_nine, zero_to_nine])
    expected_val_labels = np.concatenate([zero_to_nine, zero_to_nine])
    np.testing.assert_equal(expected_train_labels, train_y)
    # The val labels are stacked
    np.testing.assert_equal(expected_val_labels, val_y[0])
    np.testing.assert_equal(expected_val_labels, val_y[1])


if __name__ == "__main__":
    test_when_mnist_stratified_40_then_2_per_row()