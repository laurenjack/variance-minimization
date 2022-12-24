import numpy as np
import data_generator


def test_when_small_data_set_and_signal_perfect_then_labels_and_signal_correct():
    x, y = data_generator.single_relevant(4, 8, 1.5, 1.0, 5.0)
    # Check the shapes
    assert (4, 8) == x.shape
    assert (4,) == y.shape
    # Check the signals
    assert x[0, 0] == 1.5
    assert x[1, 0] == 1.5
    assert x[2, 0] == -1.5
    assert x[3, 0] == -1.5
    # Check the labels
    np.testing.assert_array_equal(np.array([0, 0, 1, 1]), y)
    # Check some random noise spots are not zero or 1
    assert x[0, 3] != 0.0 and x[0, 3] != 1.0
    assert x[2, 2] != 0.0 and x[2, 2] != 1.0


def test_when_large_data_set_then_statistical_properties_in_expected_range():
    x, y = data_generator.single_relevant(10000, 2, 1.5, 0.9, 5.0)
    # Check the shapes
    assert (10000, 2) == x.shape
    assert (10000,) == y.shape
    # Check the signals
    first_half = x[:5000, 0]
    assert 0.85 < total_of(1.5, first_half) / 5000 < 0.95
    assert 0.05 < total_of(-1.5, first_half) / 5000 < 0.15
    second_half = x[5000:, 0]
    assert 0.85 < total_of(-1.5, second_half) / 5000 < 0.95
    assert 0.05 < total_of(1.5, second_half) / 5000 < 0.15
    # Check the labels
    assert total_of(0, y[:5000]) == 5000
    assert total_of(1, y[5000:]) == 5000


def total_of(c, array):
    """Count the number of c in array"""
    count = 0
    for i in range(array.shape[0]):
        if array[i] == c:
            count += 1
    return count


if __name__ == '__main__':
    test_when_small_data_set_and_signal_perfect_then_labels_and_signal_correct()
    test_when_large_data_set_then_statistical_properties_in_expected_range()