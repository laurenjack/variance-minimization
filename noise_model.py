from scipy import special


def binary_classification(n, d):
    assert n % 2 == 0  # Model only works for even numbered n
    num_choices = n // 2
    x = num_choices
    px_greater_than_or_eq_x = 0.0
    nd = n ** d
    while x <= num_choices / 2:
        px_greater_than_or_eq_x += _px_equals_x(num_choices, x)
        p_x_less_than_x = 1 - px_greater_than_or_eq_x
        (1 - p_x_less_than_x) ** nd






def _px_equals_x(num_choices, x):
    one_side_combinations = special.comb(num_choices, x)
    total_combinations = one_side_combinations ** 2.0
    # Handle symmetry of combinations
    if num_choices // 2 == x:
        return total_combinations
    return total_combinations * 2
