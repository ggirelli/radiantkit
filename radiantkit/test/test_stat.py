"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

from radiantkit import stat


def test_quantile_from_counts():
    assert 1.5 == stat.quantile_from_counts([1, 2, 3, 4, 5], [1, 1, 1, 1, 1], 0.25)
