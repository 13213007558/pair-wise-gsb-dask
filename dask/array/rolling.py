from __future__ import annotations

import warnings
from numbers import Integral

import numpy as np

from dask.array.core import asarray
from dask.array.numpy_compat import normalize_axis_index
from dask.array.overlap import (
    _get_overlap_rechunked_chunks,
    coerce_depth,
    map_overlap,
)

_QUANTILE_METHODS = {
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "linear",
    "median_unbiased",
    "normal_unbiased",
    "lower",
    "higher",
    "nearest",
    "midpoint",
}


def _rolling_quantile_block(
    block,
    block_info=None,
    *,
    q,
    window,
    axis,
    left,
    right,
    min_periods,
    ignore_nan,
    method,
):
    """Compute rolling quantiles on a single overlapped block.

    The block already contains ``left``/``right`` extra elements from its
    neighbors along ``axis`` (except at the array boundaries).  Quantiles
    are evaluated for every position of the extended block and the overlap
    regions are trimmed off again before returning.
    """
    info = block_info[0]
    chunk_location = info["chunk-location"][axis]
    num_chunks = info["num-chunks"][axis]

    # Cast to float64, matching the dtype np.quantile computes in and
    # returns, also for integer or float32 inputs.
    x = np.asarray(block, dtype=np.float64)

    # Pad with NaN so that sliding_window_view yields exactly one window
    # per position of the (extended) block.  Padding at the array boundaries
    # produces the partial windows required by min_periods semantics;
    # padding inside overlap regions is trimmed away below.
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (left, right)
    padded = np.pad(x, pad_width, constant_values=np.nan)
    real = np.pad(np.ones(x.shape, dtype=bool), pad_width)
    data_nan = np.pad(np.isnan(x), pad_width)

    sliding_window_view = np.lib.stride_tricks.sliding_window_view
    windows = sliding_window_view(padded, window, axis=axis)
    real_windows = sliding_window_view(real, window, axis=axis)
    nan_windows = sliding_window_view(data_nan, window, axis=axis)

    if ignore_nan:
        # Only actual, non-NaN observations count towards min_periods
        valid = (real_windows & ~nan_windows).sum(axis=-1) >= min_periods
    else:
        # Any NaN inside the window poisons the result, like np.quantile
        valid = (real_windows.sum(axis=-1) >= min_periods) & ~nan_windows.any(axis=-1)

    with warnings.catch_warnings():
        # All-NaN (e.g. empty) windows legitimately produce NaN below
        warnings.simplefilter("ignore", RuntimeWarning)
        values = np.nanquantile(windows, q, axis=-1, method=method)
    # np.nanquantile prepends the quantile axes; move ours to the end
    values = np.moveaxis(values, 0, -1)
    out = np.where(valid[..., np.newaxis], values, np.nan)

    # Trim the overlap regions; blocks at the array boundary have no overlap
    trim_left = left if chunk_location > 0 else 0
    trim_right = right if chunk_location < num_chunks - 1 else 0
    index = [slice(None)] * out.ndim
    stop = out.shape[axis] - trim_right if trim_right else out.shape[axis]
    index[axis] = slice(trim_left, stop)
    return out[tuple(index)]


def rolling_quantile(
    x,
    q,
    window,
    axis=-1,
    *,
    center=False,
    min_periods=None,
    ignore_nan=True,
    method="linear",
):
    r"""Exact rolling (moving) quantiles along one axis of an array.

    For every position along ``axis`` the quantiles of the surrounding
    window of ``window`` observations are computed exactly (no
    approximation), which is useful e.g. for anomaly detection based on a
    rolling median or a rolling 95% quantile.

    The computation is block-local: each output block only reads the input
    block at the same location plus ``window - 1`` neighboring elements, so
    the axis is never merged into a single task and the memory footprint of
    a task is determined by the chunk size and the window length only.
    Chunks smaller than the window are automatically merged just enough to
    cover the window, hence the result does not depend on how the axis is
    chunked.

    Parameters
    ----------
    x : da.Array
        Input array.
    q : float or sequence of float
        Quantile(s) to compute, each in the interval [0, 1].  Several
        quantiles can be computed in one pass over the data; the result
        always gains one new trailing *quantile axis* of length
        ``len(q)`` (also when a single quantile is requested).
    window : int
        Length of the rolling window.  Must be at least 1.  If ``window``
        is larger than the axis, all windows are partial (see
        ``min_periods``).
    axis : int, optional
        Axis along which to compute the rolling quantiles.  Default is
        the last axis.
    center : bool, optional
        If False (default), use a backward-looking (trailing) window:
        position ``i`` aggregates ``x[i - window + 1 : i + 1]``.
        If True, center the window on each position: position ``i``
        aggregates ``x[i - (window - 1) // 2 : i + window // 2 + 1]``,
        matching ``pandas.rolling(..., center=True)``.
    min_periods : int, optional
        Minimum number of observations in a window required to produce a
        value; otherwise the result is NaN at that position.  With
        ``ignore_nan=True`` (default) only non-NaN observations are
        counted.  Defaults to ``window`` (matching pandas), use a smaller
        value to get values for the partial windows at the array
        boundaries.
    ignore_nan : bool, optional
        If True (default), NaN values are excluded from the window, like
        ``np.nanquantile``.  If False, any NaN inside a window makes the
        result NaN at that position, like ``np.quantile``.
    method : str, optional
        Interpolation method used when the quantile lies between two
        observations.  Accepts the same values as the ``method`` keyword
        of ``np.quantile`` (e.g. ``"linear"`` (default), ``"lower"``,
        ``"higher"``, ``"midpoint"``, ``"nearest"``).

    Returns
    -------
    da.Array
        Array of dtype ``float64`` with shape
        ``x.shape + (len(q),)``; the last axis is the quantile axis.
        The output dtype is always ``float64``, also for integer input,
        matching the return dtype of ``np.quantile``.

    Notes
    -----
    Windows that contain no observations at all (empty windows, e.g. at
    the array boundaries when ``window`` exceeds the axis length, or
    windows whose observations are all NaN with ``ignore_nan=True``)
    always produce NaN, regardless of ``min_periods``.

    Examples
    --------
    >>> import dask.array as da
    >>> x = da.arange(10, chunks=4)
    >>> r = rolling_quantile(x, 0.5, 3)
    >>> r.shape  # one trailing quantile axis of length 1
    (10, 1)
    >>> r.compute().ravel()
    array([nan, nan,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.])

    Several quantiles at once, with partial windows at the boundaries:

    >>> r = rolling_quantile(x, [0.5, 1.0], 4, min_periods=1)
    >>> r.compute()[0]
    array([0., 0.])

    See Also
    --------
    numpy.quantile
    dask.array.map_overlap
    """
    if not isinstance(window, Integral) or isinstance(window, bool):
        raise TypeError(f"window must be an integer, got {window!r}")
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")

    q = np.atleast_1d(np.asarray(q, dtype=np.float64))
    if q.ndim != 1 or q.size == 0:
        raise ValueError("q must be a scalar or a non-empty 1-D sequence of floats")
    if np.isnan(q).any() or (q < 0).any() or (q > 1).any():
        raise ValueError(f"quantiles must be in the interval [0, 1], got {q!r}")

    if min_periods is None:
        min_periods = window
    if not isinstance(min_periods, Integral) or isinstance(min_periods, bool):
        raise TypeError(f"min_periods must be an integer, got {min_periods!r}")
    if not 0 <= min_periods <= window:
        raise ValueError(f"min_periods must be between 0 and window ({window}), got {min_periods}")

    if method not in _QUANTILE_METHODS:
        raise ValueError(f"method must be one of {sorted(_QUANTILE_METHODS)}, got {method!r}")

    x = asarray(x)
    axis = normalize_axis_index(axis, x.ndim)

    if center:
        left, right = (window - 1) // 2, window // 2
    else:
        left, right = window - 1, 0

    # Overlap cannot exceed what the axis can supply; when the window is
    # larger than the axis every window already reaches the array boundary,
    # so capping the depth does not change the result (the axis collapses
    # to a single chunk in that case).
    n = x.shape[axis]
    depth = {axis: (min(left, max(n - 1, 0)), min(right, max(n - 1, 0)))}
    # Chunks smaller than the window are merged just enough to cover the
    # window; replicate the rechunking map_overlap performs so the
    # output chunks (including the new quantile axis) can be declared.
    rechunked = _get_overlap_rechunked_chunks(x, coerce_depth(x.ndim, depth))
    chunks = tuple(rechunked) + (len(q),)

    return map_overlap(
        _rolling_quantile_block,
        x,
        depth=depth,
        boundary="none",
        trim=False,
        chunks=chunks,
        new_axis=x.ndim,
        dtype=np.float64,
        meta=np.empty((0,) * (x.ndim + 1), dtype=np.float64),
        q=q,
        window=int(window),
        axis=axis,
        left=left,
        right=right,
        min_periods=int(min_periods),
        ignore_nan=bool(ignore_nan),
        method=method,
    )
