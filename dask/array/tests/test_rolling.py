from __future__ import annotations

import pytest

pytest.importorskip("numpy")

import numpy as np

import dask.array as da
from dask.array.numpy_compat import AxisError
from dask.array.rolling import _rolling_quantile_block, rolling_quantile

if da._array_expr_enabled():
    pytest.skip("rolling_quantile is not supported with array-expr", allow_module_level=True)


def reference_rolling_quantile(
    a,
    q,
    window,
    axis=-1,
    center=False,
    min_periods=None,
    ignore_nan=True,
    method="linear",
):
    """Naive per-position implementation used as ground truth."""
    a = np.asarray(a, dtype=np.float64)
    q = np.atleast_1d(np.asarray(q, dtype=np.float64))
    a = np.moveaxis(a, axis, 0)
    axis = axis % a.ndim
    n = a.shape[0]
    flat = a.reshape(n, -1)
    if min_periods is None:
        min_periods = window
    if center:
        left, right = (window - 1) // 2, window // 2
    else:
        left, right = window - 1, 0
    out = np.full((n, flat.shape[1], len(q)), np.nan)
    for i in range(n):
        lo = max(0, i - left)
        hi = min(n, i + right + 1)
        w = flat[lo:hi]
        for j in range(flat.shape[1]):
            col = w[:, j]
            nan = np.isnan(col)
            if ignore_nan:
                vals = col[~nan]
                if len(vals) >= max(min_periods, 1):
                    out[i, j, :] = np.quantile(vals, q, method=method)
            else:
                if nan.any() or len(col) < min_periods:
                    continue
                out[i, j, :] = np.quantile(col, q, method=method)
    out = out.reshape((n,) + a.shape[1:] + (len(q),))
    return np.moveaxis(out, 0, axis)


def assert_rolling_eq(a, chunks, **kwargs):
    x = da.from_array(a, chunks=chunks)
    result = rolling_quantile(x, **kwargs)
    expected = reference_rolling_quantile(a, **kwargs)
    assert result.shape == expected.shape
    assert result.dtype == np.float64
    actual = result.compute()
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    return result


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_trailing_window(rng):
    a = rng.normal(size=37)
    assert_rolling_eq(a, 5, q=[0.5, 0.95], window=5)


def test_centered_window(rng):
    a = rng.normal(size=37)
    assert_rolling_eq(a, 5, q=[0.5, 0.95], window=5, center=True)
    # even window: centering is off by one, like pandas
    assert_rolling_eq(a, 5, q=[0.5], window=4, center=True)


@pytest.mark.parametrize("axis", [0, 1, 2, -1, -2])
def test_axis(rng, axis):
    a = rng.normal(size=(4, 9, 7))
    assert_rolling_eq(a, (2, 3, 4), q=[0.25, 0.5, 0.95], window=3, axis=axis)


def test_quantile_axis_is_trailing(rng):
    a = rng.normal(size=(4, 9))
    x = da.from_array(a, chunks=(2, 3))
    result = rolling_quantile(x, [0.5, 0.95], 3, axis=1)
    assert result.shape == (4, 9, 2)
    assert result.chunks[-1] == (2,)
    # a scalar q still adds an explicit quantile axis of length 1
    result = rolling_quantile(x, 0.5, 3, axis=1)
    assert result.shape == (4, 9, 1)
    assert result.chunks[-1] == (1,)


@pytest.mark.parametrize("chunks", [3, 5, 7, 100, (10, 10, 10, 7)])
def test_chunk_invariance(rng, chunks):
    a = rng.normal(size=37)
    expected = reference_rolling_quantile(a, [0.5, 0.95], 6, center=True, min_periods=2)
    x = da.from_array(a, chunks=chunks)
    actual = rolling_quantile(x, [0.5, 0.95], 6, center=True, min_periods=2).compute()
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_window_larger_than_chunks_does_not_collapse_axis(rng):
    a = rng.normal(size=40)
    x = da.from_array(a, chunks=3)
    result = rolling_quantile(x, 0.5, 8)
    # chunks are merged only enough to cover the window, never into
    # a single task spanning the whole axis
    assert result.numblocks[0] > 1
    assert all(c >= 7 for c in result.chunks[0])
    np.testing.assert_allclose(
        result.compute(),
        reference_rolling_quantile(a, 0.5, 8),
        equal_nan=True,
    )


def test_axis_shorter_than_window(rng):
    a = rng.normal(size=4)
    x = da.from_array(a, chunks=2)
    # default min_periods=window: no window is ever full
    result = rolling_quantile(x, [0.5, 0.95], 10)
    assert result.shape == (4, 2)
    assert np.isnan(result.compute()).all()
    # with min_periods, partial windows produce values
    assert_rolling_eq(a, 2, q=[0.5, 0.95], window=10, min_periods=1)
    assert_rolling_eq(a, 3, q=0.5, window=10, min_periods=3, center=True)


def test_last_short_chunk(rng):
    a = rng.normal(size=(3, 17))
    assert_rolling_eq(a, (3, 5), q=[0.5, 0.95], window=5, axis=1)
    assert_rolling_eq(a, (3, 5), q=[0.5, 0.95], window=5, axis=1, center=True, min_periods=2)


def test_multiple_quantiles_share_windows(rng):
    a = rng.normal(size=30)
    x = da.from_array(a, chunks=4)
    qs = [0.1, 0.5, 0.9, 0.95]
    combined = rolling_quantile(x, qs, 6, min_periods=2).compute()
    for i, q in enumerate(qs):
        single = rolling_quantile(x, q, 6, min_periods=2).compute()
        np.testing.assert_allclose(combined[..., i], single[..., 0], equal_nan=True)
    assert_rolling_eq(a, 4, q=qs, window=6, min_periods=2)


@pytest.mark.parametrize("min_periods", [1, 2, 3, 5])
def test_min_periods(rng, min_periods):
    a = rng.normal(size=20)
    assert_rolling_eq(a, 4, q=[0.5], window=5, min_periods=min_periods)
    assert_rolling_eq(a, 4, q=[0.5], window=5, min_periods=min_periods, center=True)


def test_min_periods_default_is_window(rng):
    a = rng.normal(size=10)
    x = da.from_array(a, chunks=3)
    result = rolling_quantile(x, 0.5, 4).compute()
    # the first window - 1 positions never have a full window
    assert np.isnan(result[:3]).all()
    assert not np.isnan(result[3:]).any()


def test_ignore_nan(rng):
    a = rng.normal(size=30)
    a[::3] = np.nan
    # NaNs are excluded; min_periods counts non-NaN observations
    assert_rolling_eq(a, 4, q=[0.5, 0.95], window=5, min_periods=2, ignore_nan=True)
    # any NaN in a window poisons the result
    assert_rolling_eq(a, 4, q=[0.5, 0.95], window=5, min_periods=2, ignore_nan=False)


def test_ignore_nan_false_poisons_window():
    a = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
    x = da.from_array(a, chunks=2)
    result = rolling_quantile(x, 0.5, 3, min_periods=1, ignore_nan=False).compute()
    assert result[0, 0] == 1.0  # partial window without NaN
    assert np.isnan(result[1:4]).all()  # windows containing the NaN
    assert result[4, 0] == 4.0


def test_all_nan_window():
    a = np.arange(12, dtype=float)
    a[4:8] = np.nan
    result = assert_rolling_eq(a, 3, q=[0.5, 0.95], window=4, min_periods=1)
    computed = result.compute()
    # the position whose trailing window only contains NaNs is NaN ...
    assert np.isnan(computed[7]).all()
    # ... while positions with at least one valid observation are not
    assert not np.isnan(computed[[0, 3, 4, 8, 11]]).any()

    a = np.full(12, np.nan)
    x = da.from_array(a, chunks=3)
    assert np.isnan(rolling_quantile(x, [0.5, 0.95], 4, min_periods=1).compute()).all()


def test_empty_window_is_nan():
    # min_periods=0 still yields NaN for windows without any observation
    a = np.arange(5, dtype=float)
    x = da.from_array(a, chunks=2)
    result = rolling_quantile(x, 0.5, 3, min_periods=0).compute()
    expected = reference_rolling_quantile(a, 0.5, 3, min_periods=0)
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_integer_input_output_dtype(rng):
    a = rng.integers(0, 100, size=(5, 30))
    x = da.from_array(a, chunks=(2, 4))
    result = rolling_quantile(x, [0.5, 0.95], 5, axis=1)
    # like np.quantile, integer input produces float64 output
    assert result.dtype == np.float64
    assert result.compute().dtype == np.float64
    np.testing.assert_allclose(
        result.compute(),
        reference_rolling_quantile(a, [0.5, 0.95], 5, axis=1),
        equal_nan=True,
    )


@pytest.mark.parametrize("method", ["linear", "lower", "higher", "midpoint", "nearest", "inverted_cdf"])
def test_method(rng, method):
    a = rng.normal(size=25)
    assert_rolling_eq(a, 6, q=[0.1, 0.5, 0.9], window=7, method=method)
    assert_rolling_eq(a, 6, q=[0.1, 0.5, 0.9], window=7, method=method, center=True)


def test_window_one(rng):
    a = rng.normal(size=10)
    x = da.from_array(a, chunks=3)
    result = rolling_quantile(x, [0.5, 0.95], 1).compute()
    np.testing.assert_allclose(result[..., 0], a)
    np.testing.assert_allclose(result[..., 1], a)


def test_no_tasks_run_during_construction(rng, monkeypatch):
    a = rng.normal(size=(4, 20))
    x = da.from_array(a, chunks=(2, 3))
    calls = []

    def spy(block, block_info=None, **kwargs):
        calls.append(1)
        return _rolling_quantile_block(block, block_info=block_info, **kwargs)

    monkeypatch.setattr("dask.array.rolling._rolling_quantile_block", spy)
    result = rolling_quantile(x, [0.5, 0.95], 4, axis=1)
    # building the graph must not execute any data task
    assert calls == []
    assert isinstance(result, da.Array)
    result.compute()
    assert len(calls) > 0


def test_output_block_structure(rng):
    a = rng.normal(size=(4, 20))
    x = da.from_array(a, chunks=(2, 5))
    result = rolling_quantile(x, [0.5, 0.95], 4, axis=1)
    # one output block per input block along the axis, no merging
    assert result.numblocks == (2, 4, 1)
    assert result.chunks[:2] == x.chunks


def test_invalid_arguments(rng):
    x = da.from_array(rng.normal(size=10), chunks=3)
    with pytest.raises(ValueError, match="at least 1"):
        rolling_quantile(x, 0.5, 0)
    with pytest.raises(TypeError, match="window must be an integer"):
        rolling_quantile(x, 0.5, 2.5)
    with pytest.raises(ValueError, match="interval"):
        rolling_quantile(x, 1.5, 3)
    with pytest.raises(ValueError, match="interval"):
        rolling_quantile(x, [0.5, -0.1], 3)
    with pytest.raises(ValueError, match="non-empty"):
        rolling_quantile(x, [], 3)
    with pytest.raises(ValueError, match="min_periods"):
        rolling_quantile(x, 0.5, 3, min_periods=4)
    with pytest.raises(ValueError, match="min_periods"):
        rolling_quantile(x, 0.5, 3, min_periods=-1)
    with pytest.raises(ValueError, match="method"):
        rolling_quantile(x, 0.5, 3, method="not-a-method")
    with pytest.raises(AxisError):
        rolling_quantile(x, 0.5, 3, axis=5)
