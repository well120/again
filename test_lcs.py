import pytest
import numpy as np
from math import sqrt, ceil

from lcs import lcs_bf, lcs


def _test_lcs(fn, a, b, c):
    r = fn(a, b)
    assert r.shape == c.shape
    if np.prod(r.shape) > 0:
        assert np.all(r == c)


@pytest.mark.lcs_bf
def test_empty_bf():
    _test_lcs(lcs_bf, np.array([]), np.array([]), np.array([]))

@pytest.mark.lcs
def test_empty():
    _test_lcs(lcs, np.array([]), np.array([]), np.array([]))


@pytest.mark.lcs_bf
def test_one_bf():
    _test_lcs(lcs_bf, np.array([1]), np.array([1]), np.array([1]))
    _test_lcs(lcs_bf, np.array([1]), np.array([2]), np.array([]))
    _test_lcs(lcs_bf, np.array([1]), np.array([]), np.array([]))
    _test_lcs(lcs_bf, np.array([]), np.array([1]), np.array([]))

@pytest.mark.lcs
def test_one():
    _test_lcs(lcs, np.array([1]), np.array([1]), np.array([1]))
    _test_lcs(lcs, np.array([1]), np.array([2]), np.array([]))
    _test_lcs(lcs, np.array([1]), np.array([]), np.array([]))
    _test_lcs(lcs, np.array([]), np.array([1]), np.array([]))


ea = np.array([0, 5, 3, 9, 3, 3, 8, 4, 2, 5])
eb = np.array([9, 0, 7, 6, 5, 0, 3, 7, 1, 0])
ec = np.array([0, 5, 3])

@pytest.mark.lcs_bf
def test_example_bf():
    _test_lcs(lcs_bf, ea, eb, ec)

@pytest.mark.lcs
def test_example():
    _test_lcs(lcs, ea, eb, ec)


def _emplace(rng, dst, src):
    rn = np.arange(len(dst))
    rng.shuffle(rn)
    pos = np.sort(rn[:len(src)])
    dst[pos] = src

def _test_lcs_gen(fn, n, m, k):
    assert n >= k and m >= k
    rng = np.random.RandomState(0xDEADBEEF)

    c = rng.choice(ceil(sqrt(k)), k) if ceil(sqrt(k)) > 0 else np.array([])

    un = np.setdiff1d(np.arange(n + m + k), c)
    rng.shuffle(un)

    a = un[:n].copy()
    _emplace(rng, a, c)

    b = un[n:n+m].copy()
    _emplace(rng, b, c)

    _test_lcs(fn, a, b, c)


def _test_n_lcs(fn, n):
    _test_lcs_gen(fn, n, n, n)
    _test_lcs_gen(fn, n, n, 1)
    _test_lcs_gen(fn, n, n, 0)
    _test_lcs_gen(fn, n, n, ceil(sqrt(n)))
    _test_lcs_gen(fn, n, n//2, ceil(sqrt(n)))
    _test_lcs_gen(fn, n//2, n, ceil(sqrt(n)))

@pytest.mark.timeout(60)
@pytest.mark.lcs_bf
def test_10_bf():
    _test_n_lcs(lcs_bf, 10)

@pytest.mark.timeout(60)
@pytest.mark.lcs
def test_10():
    _test_n_lcs(lcs, 10)

@pytest.mark.timeout(60)
@pytest.mark.lcs
def test_100():
    _test_n_lcs(lcs, 100)

@pytest.mark.timeout(60)
@pytest.mark.lcs
def test_1000():
    _test_n_lcs(lcs, 1000)
