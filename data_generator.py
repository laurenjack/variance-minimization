import numpy as np


def single_relevant(n, d, magnitude_signal, percent_correct, sd_non_signals, balanced=False):
    x = sd_non_signals * np.random.randn(n, d)
    examples_per_class = n // 2  # Just 2 classes
    # The first class is in the first half of the data
    p = [percent_correct, 1 - percent_correct]
    if balanced:
        signal = np.ones(int(examples_per_class * p[0]))
        anti_signal = np.ones(int(examples_per_class * p[1]))
        signal_0 = np.concatenate([magnitude_signal * signal, -magnitude_signal * anti_signal])
        signal_1 = np.concatenate([-magnitude_signal * signal, magnitude_signal * anti_signal])
    else:
        signal_0 = np.random.choice([magnitude_signal, -magnitude_signal], size=examples_per_class, p=p)
        signal_1 = np.random.choice([-magnitude_signal, magnitude_signal], size=examples_per_class, p=p)
    signal = np.concatenate([signal_0, signal_1])
    x[:, 0] = signal
    y = np.concatenate([np.zeros(examples_per_class, dtype=int), np.ones(examples_per_class, dtype=int)])
    return x, y
