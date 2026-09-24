from __future__ import annotations

import warnings

import numpy as np
import pytest

import dask.array as da
from dask.array import assert_eq
from dask.array.rolling import rolling_quantile
from dask.callbacks import Callback


def reference_rolling_quantile(x, q, window, axis=-1, center=True, min_periods=None, ignore_nan=True, method="linear"):
    """Naive per-position rolling quantile used as ground truth."""
    x = np.asarray(x, dtype=float)
    axis = axis % x.ndim
    x = np.moveaxis(x, axis, -1)
    n = x.shape[-1]
    if min_periods is None:
        min_periods = window
    if center:
        left = (window - 1) // 2
        right = window // 2
    else:
        left, right = window - 1, 0
    qs = np.atleast_1d(q)
    out = np.full(x.shape + (len(qs),), np.nan)
    for i in range(n):
        w = x[..., max(i - left, 0) : i + right + 1]
        if ignore_nan:
            count = np.sum(~np.isnan(w), axis=-1)
        else:
            count = np.broadcast_to(w.shape[-1], x.shape[:-1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vals = np.nanquantile(w, qs, axis=-1, method=method)
        vals = np.moveaxis(vals, 0, -1)
        if not ignore_nan:
            vals = np.where(np.isnan(w).any(axis=-1)[..., None], np.nan, vals)
        ok = count >= min_periods
        out[..., i, :] = np.where(ok[..., None], vals, np.nan)
    out = np.moveaxis(out, -2, axis)
    if np.isscalar(q):
        out = out[..., 0]
    return out


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    x = rng.normal(size=60)
    x[::7] = np.nan
    return x


@pytest.mark.parametrize("chunks", [1, 3, 7, 60])
@pytest.mark.parametrize("center", [True, False])
def test_matches_reference_regardless_of_chunking(data, chunks, center):
    window = 11
    expected = reference_rolling_quantile(data, 0.5, window, center=center)
    x = da.from_array(data, chunks=chunks)
    result = rolling_quantile(x, 0.5, window, center=center)
    assert_eq(result, expected)


@pytest.mark.parametrize("window", [1, 2, 3, 4, 5])
def test_odd_and_even_windows(data, window):
    x = da.from_array(data, chunks=9)
    for center in [True, False]:
        expected = reference_rolling_quantile(data, 0.95, window, center=center)
        result = rolling_quantile(x, 0.95, window, center=center)
        assert_eq(result, expected)


def test_multiple_quantiles_share_windows(data):
    qs = [0.05, 0.5, 0.95]
    x = da.from_array(data, chunks=8)
    result = rolling_quantile(x, qs, 7)
    assert result.shape == (data.size, 3)
    # the quantile axis is explicit and a single chunk
    assert result.chunks[-1] == (3,)
    expected = reference_rolling_quantile(data, qs, 7)
    assert_eq(result, expected)
    # quantiles computed together match quantiles computed one by one
    for i, qi in enumerate(qs):
        single = rolling_quantile(x, qi, 7)
        assert_eq(result[..., i], single)


def test_scalar_q_adds_no_quantile_axis(data):
    x = da.from_array(data, chunks=8)
    result = rolling_quantile(x, 0.5, 5)
    assert result.shape == data.shape
    assert result.ndim == 1


def test_single_element_q_adds_quantile_axis(data):
    x = da.from_array(data, chunks=8)
    result = rolling_quantile(x, [0.5], 5)
    assert result.shape == (data.size, 1)
    assert_eq(result[..., 0], rolling_quantile(x, 0.5, 5))


@pytest.mark.parametrize("axis", [0, 1, 2, -1, -3])
def test_axis(axis):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(6, 7, 8))
    a[a > 1.5] = np.nan
    x = da.from_array(a, chunks=(2, 3, 4))
    q = [0.25, 0.75]
    result = rolling_quantile(x, q, 4, axis=axis, min_periods=2)
    expected = reference_rolling_quantile(a, q, 4, axis=axis, min_periods=2)
    assert result.shape == expected.shape
    assert_eq(result, expected)


def test_min_periods(data):
    x = da.from_array(data, chunks=6)
    for min_periods in [1, 3, 5]:
        expected = reference_rolling_quantile(data, 0.5, 5, min_periods=min_periods)
        result = rolling_quantile(x, 0.5, 5, min_periods=min_periods)
        assert_eq(result, expected)


def test_default_min_periods_requires_full_window():
    x = da.arange(10, chunks=4, dtype=float)
    result = rolling_quantile(x, 0.5, 5, center=False).compute()
    assert np.all(np.isnan(result[:4]))
    assert not np.any(np.isnan(result[4:]))


def test_ignore_nan_false_propagates():
    a = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
    x = da.from_array(a, chunks=2)
    result = rolling_quantile(x, 0.5, 3, center=False, min_periods=1, ignore_nan=False).compute()
    expected = reference_rolling_quantile(a, 0.5, 3, center=False, min_periods=1, ignore_nan=False)
    np.testing.assert_equal(result, expected)
    # windows containing the NaN sample are poisoned
    assert np.isnan(result[1]) and np.isnan(result[2]) and np.isnan(result[3])
    assert result[0] == 1.0 and result[4] == 4.0


def test_all_nan_window():
    a = np.full(10, np.nan)
    a[0] = 1.0
    x = da.from_array(a, chunks=3)
    # min_periods=1: windows with at least one non-NaN sample still compute
    result = rolling_quantile(x, [0.5, 0.95], 3, min_periods=1).compute()
    assert result[0].tolist() == [1.0, 1.0]
    assert np.all(np.isnan(result[3:]))
    # all-NaN windows are NaN, without raising All-NaN warnings
    all_nan = rolling_quantile(da.full(6, np.nan, chunks=2), 0.5, 3).compute()
    assert np.all(np.isnan(all_nan))


def test_integer_input_gives_float64():
    x = da.arange(10, chunks=3)
    result = rolling_quantile(x, [0.5, 0.95], 3)
    assert result.dtype == np.float64
    expected = reference_rolling_quantile(np.arange(10.0), [0.5, 0.95], 3)
    assert_eq(result, expected)


def test_float32_input_gives_float64():
    x = da.ones(10, chunks=3, dtype=np.float32)
    result = rolling_quantile(x, 0.5, 3)
    assert result.dtype == np.float64


def test_window_larger_than_axis():
    rng = np.random.default_rng(1)
    a = rng.normal(size=4)
    x = da.from_array(a, chunks=2)
    # truncated windows everywhere; every sample is observed everywhere
    result = rolling_quantile(x, 0.5, 101, min_periods=1)
    expected = reference_rolling_quantile(a, 0.5, 101, min_periods=1)
    assert_eq(result, expected)
    # default min_periods=window can never be satisfied
    result = rolling_quantile(x, 0.5, 101)
    assert_eq(result, np.full(4, np.nan))


def test_last_short_chunk():
    rng = np.random.default_rng(2)
    a = rng.normal(size=11)
    x = da.from_array(a, chunks=(4, 4, 3))
    for center in [True, False]:
        result = rolling_quantile(x, [0.5, 0.95], 6, center=center, min_periods=2)
        expected = reference_rolling_quantile(a, [0.5, 0.95], 6, center=center, min_periods=2)
        assert_eq(result, expected)


def test_window_spanning_many_chunks_is_not_merged(data):
    x = da.from_array(data, chunks=3)
    window = 21
    result = rolling_quantile(x, 0.5, window)
    # the axis is rechunked to fit the halo, but not merged into one chunk
    assert result.numblocks[0] > 1
    assert max(result.chunks[0]) < data.size
    expected = reference_rolling_quantile(data, 0.5, window)
    assert_eq(result, expected)


@pytest.mark.parametrize("method", ["linear", "nearest", "lower", "midpoint", "weibull"])
def test_method(data, method):
    x = da.from_array(data, chunks=5)
    result = rolling_quantile(x, [0.3, 0.7], 6, method=method)
    expected = reference_rolling_quantile(data, [0.3, 0.7], 6, method=method)
    assert_eq(result, expected)


def test_empty_axis():
    x = da.from_array(np.array([]), chunks=1)
    result = rolling_quantile(x, 0.5, 5)
    assert result.shape == (0,)
    assert_eq(result, np.array([]))
    result = rolling_quantile(x, [0.5, 0.95], 5)
    assert result.shape == (0, 2)
    assert_eq(result, np.empty((0, 2)))


def test_array_like_input():
    a = np.arange(10, dtype=float)
    result = rolling_quantile(a, 0.5, 3, min_periods=1)
    expected = reference_rolling_quantile(a, 0.5, 3, min_periods=1)
    assert_eq(result, expected)


def test_no_data_tasks_on_construction():
    x = da.arange(100, chunks=7)
    executed = []
    with Callback(pretask=lambda *args: executed.append(args)):
        result = rolling_quantile(x, [0.5, 0.95], 11)
        result.__dask_graph__()
    assert isinstance(result, da.Array)
    assert executed == []


@pytest.mark.parametrize("window", [0, -1, 2.5, True, "3"])
def test_invalid_window(window):
    x = da.arange(10, chunks=3)
    with pytest.raises(ValueError, match="window"):
        rolling_quantile(x, 0.5, window)


@pytest.mark.parametrize("q", [-0.1, 1.1, [0.5, 2.0]])
def test_invalid_quantile(q):
    x = da.arange(10, chunks=3)
    with pytest.raises(ValueError, match="quantile"):
        rolling_quantile(x, q, 3)


def test_empty_quantile_sequence():
    x = da.arange(10, chunks=3)
    with pytest.raises(ValueError, match="at least one"):
        rolling_quantile(x, [], 3)


@pytest.mark.parametrize("min_periods", [0, -1, 1.5])
def test_invalid_min_periods(min_periods):
    x = da.arange(10, chunks=3)
    with pytest.raises(ValueError, match="min_periods"):
        rolling_quantile(x, 0.5, 3, min_periods=min_periods)


def test_invalid_method():
    x = da.arange(10, chunks=3)
    with pytest.raises(ValueError, match="method"):
        rolling_quantile(x, 0.5, 3, method="not-a-method")
