"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

from radiantkit.stat import quantile_from_counts


def test_quantile_from_counts():
    assert 1.5 == quantile_from_counts([1, 2, 3, 4, 5], [1, 1, 1, 1, 1], 0.25)
