"""Die One-Hot-Zustandsdarstellung von Hand."""

import numpy as np

from pg_features import one_hot


def test_one_hot_shape_and_values():
    x = one_hot(3, n_states=5)
    assert x.shape == (5,)
    assert np.array_equal(x, [0, 0, 0, 1, 0])


def test_one_hot_covers_every_state():
    for s in range(5):
        x = one_hot(s, n_states=5)
        assert x.sum() == 1.0 and x[s] == 1.0
