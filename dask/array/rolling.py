from __future__ import annotations

import warnings
from numbers import Integral, Real

import numpy as np

from dask.array.core import Array, asarray, map_blocks
from dask.array.numpy_compat import normalize_axis_index
from dask.array.overlap import overlap, trim_internal
from dask.array.utils import meta_from_array


def _rolling_quantile_chunk(
    x,
    q,
    window,
    axis,
    left,
    right,
    min_periods,
    ignore_nan,
    method,
    q_scalar,
    out_dtype,
):
    """Compute exact rolling quantiles for every position of a block.

    ``x`` is a block that already overlaps with its neighbors by ``left``
    samples on the lower side and ``right`` samples on the upper side along
    ``axis``.  The result has the same shape as ``x`` plus, for non-scalar
    ``q``, a trailing quantile axis.  The overlap zones are trimmed away by
    the caller.
    """
    n = x.shape[axis]
    if q_scalar:
        out_shape = x.shape
    else:
        out_shape = x.shape + (len(q),)
    if n == 0:
        return np.empty(out_shape, dtype=out_dtype)

    # Pad with NaN so that every position is the center (or end) of a
    # full-length window.  Padding stands in for samples that do not exist
    # (outside of the array or of this block's trimmed core) and is never
    # counted as an observation.  Non-inexact dtypes (e.g. integers) are
    # upcast so that they can hold the NaN padding.
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (left, right)
    padded = np.pad(
        x.astype(out_dtype, copy=False),
        pad_width,
        mode="constant",
        constant_values=np.nan,
    )

    # sliding_window_view appends the window axis at the end; the view does
    # not copy, so memory stays proportional to the block plus the padding.
    windows = np.lib.stride_tricks.sliding_window_view(padded, window, axis=axis)

    # Number of real (non-padding) samples in the window of every position.
    idx = np.arange(n)
    lo = np.maximum(idx - left, 0)
    hi = np.minimum(idx + right, n - 1)
    bshape = [1] * x.ndim
    bshape[axis] = n
    nreal = (hi - lo + 1).reshape(bshape)

    nvalid = np.sum(~np.isnan(windows), axis=-1)
    # Real samples that are NaN (padding is NaN too, but never real).
    ndata_nan = nreal - nvalid

    with warnings.catch_warnings():
        # All-NaN windows return NaN, which is exactly what we want; the
        # warning would be noise (and an error under ``-W error``).
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        res = np.nanquantile(windows, q[0] if q_scalar else q, axis=-1, method=method)
    if not q_scalar:
        # nanquantile prepends the quantile axis; move it to the end.
        res = np.moveaxis(res, 0, -1)

    if ignore_nan:
        insufficient = nvalid < min_periods
        nan_propagate = np.zeros(x.shape, dtype=bool)
    else:
        insufficient = nreal < min_periods
        # Match np.quantile: any NaN among the real samples poisons the result.
        nan_propagate = ndata_nan > 0
    mask = insufficient | nan_propagate
    if not q_scalar:
        mask = mask[..., None]
    res = np.where(mask, np.nan, res)
    return res.astype(out_dtype, copy=False)


def rolling_quantile(
    x,
    q,
    window,
    axis=-1,
    center=True,
    min_periods=None,
    ignore_nan=True,
    method="linear",
):
    """Exact rolling (moving window) quantile along an axis.

    For every position along ``axis``, the quantile(s) ``q`` of the samples
    in a window of length ``window`` are computed.  This is useful e.g. for
    anomaly detection based on a rolling median or a rolling 95% quantile.

    The computation is exact: every window is materialized from the original
    samples, so results are identical no matter how the input is chunked.
    Windows that span chunk boundaries are handled by sharing a halo of
    ``window - 1`` samples between neighboring blocks; the axis is *not*
    merged into a single chunk (unless the window is as long as the axis
    itself, in which case the whole axis is necessarily one window).  Peak
    memory per task is proportional to the size of one block times the
    window length.

    Parameters
    ----------
    x : array_like
        Input data.
    q : float or sequence of float
        Quantile(s) to compute, each in the range [0, 1] (e.g. 0.5 for the
        median, 0.95 for the 95% quantile).  If a sequence is given, all
        quantiles are computed in one pass over each window and the result
        gains a dedicated trailing *quantile axis* of length ``len(q)``.
        If a scalar is given, no extra axis is added.
    window : int
        Length of the rolling window.  Must be >= 1.
    axis : int, optional
        Axis along which to compute the rolling quantile.  Default: -1.
    center : bool, optional
        If True (default), the window is centered on each position and
        covers ``(window - 1) // 2`` samples before and ``window // 2``
        samples after it.  If False, the window is trailing (backwards
        looking) and covers the ``window`` samples up to and including the
        current position.
    min_periods : int, optional
        Minimum number of observations required to produce a value;
        positions with fewer observations are NaN.  When ``ignore_nan`` is
        True, observations are the non-NaN samples in the window; otherwise
        every real (non-padding) sample counts.  Defaults to ``window``,
        i.e. a complete window is required (matching ``pandas.rolling``).
    ignore_nan : bool, optional
        If True (default), NaN samples are excluded from the quantile
        computation (like ``np.nanquantile``).  If False, any NaN in a
        window makes the result NaN (like ``np.quantile``).
    method : str, optional
        Interpolation method passed on to ``np.quantile`` (e.g. "linear",
        "nearest", "lower", "higher", "midpoint", "weibull", ...).
        Defaults to "linear".

    Returns
    -------
    dask.array.Array
        Array with the same shape as ``x`` when ``q`` is a scalar, or with
        an additional trailing axis of length ``len(q)`` when ``q`` is a
        sequence.  The dtype is the result type of the input dtype and
        ``float64``; in particular integer and float32 input produces
        float64 output, matching ``np.quantile``.

    Notes
    -----
    Windows are truncated at the boundaries of the array: positions closer
    to an edge than half the window size are computed from fewer samples,
    or are NaN if fewer than ``min_periods`` observations are available.
    A window that contains only NaN samples (or no samples at all, which
    can only happen for an empty input axis) results in NaN.  An empty
    input axis produces an empty output.

    Examples
    --------
    >>> import dask.array as da

    Trailing rolling median of the last 3 samples:

    >>> x = da.arange(10, chunks=4)
    >>> da.rolling_quantile(x, 0.5, 3, center=False, min_periods=1).compute()
    array([0. , 0.5, 1. , 2. , 3. , 4. , 5. , 6. , 7. , 8. ])

    Centered window (the default) requires a full window by default, so the
    edges are NaN:

    >>> da.rolling_quantile(x, 0.5, 3).compute()
    array([nan,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8., nan])

    Multiple quantiles at once add a trailing quantile axis:

    >>> da.rolling_quantile(x, [0.5, 1.0], 3, min_periods=1).compute()
    array([[0.5, 1. ],
           [1. , 2. ],
           [2. , 3. ],
           [3. , 4. ],
           [4. , 5. ],
           [5. , 6. ],
           [6. , 7. ],
           [7. , 8. ],
           [8. , 9. ],
           [8.5, 9. ]])
    """
    if not isinstance(window, Integral) or isinstance(window, bool) or window < 1:
        raise ValueError(f"window must be a positive integer, got {window!r}")
    window = int(window)

    q_scalar = isinstance(q, Real) and not isinstance(q, bool)
    if q_scalar:
        qs = (float(q),)
    else:
        try:
            qs = tuple(float(qi) for qi in q)
        except TypeError:
            raise TypeError(f"q must be a float or a sequence of floats, got {q!r}") from None
        if not qs:
            raise ValueError("q must contain at least one quantile")
    for qi in qs:
        if not 0 <= qi <= 1:
            raise ValueError(f"quantiles must be in the range [0, 1], got {qi}")

    if min_periods is None:
        min_periods = window
    elif not isinstance(min_periods, Integral) or isinstance(min_periods, bool) or min_periods < 1:
        raise ValueError(f"min_periods must be a positive integer, got {min_periods!r}")
    min_periods = int(min_periods)

    if not isinstance(x, Array):
        x = asarray(x)
    axis = normalize_axis_index(axis, x.ndim)

    # Validate the interpolation method eagerly (constant-time check on a
    # dummy array, mirroring np.quantile's own validation).
    np.quantile(np.zeros(2), 0.5, method=method)

    if center:
        left = (window - 1) // 2
        right = window // 2
    else:
        left = window - 1
        right = 0

    # The overlap cannot fetch more samples than the rest of the axis holds;
    # cap the halo accordingly (relevant when window >= axis length).
    n = x.shape[axis]
    ov_left = min(left, max(n - 1, 0))
    ov_right = min(right, max(n - 1, 0))
    depth = {axis: (ov_left, ov_right)}

    out_dtype = np.result_type(x.dtype, np.float64)
    q_scalar = len(qs) == 1 and q_scalar

    x2 = overlap(x, depth=depth, boundary="none")
    if q_scalar:
        chunks = x2.chunks
        new_axis = None
        out_ndim = x.ndim
    else:
        chunks = tuple(x2.chunks) + ((len(qs),),)
        new_axis = [x.ndim]
        out_ndim = x.ndim + 1
    meta = meta_from_array(x._meta, ndim=out_ndim, dtype=out_dtype)

    y = map_blocks(
        _rolling_quantile_chunk,
        x2,
        q=qs,
        window=window,
        axis=axis,
        left=left,
        right=right,
        min_periods=min_periods,
        ignore_nan=ignore_nan,
        method=method,
        q_scalar=q_scalar,
        out_dtype=out_dtype,
        chunks=chunks,
        new_axis=new_axis,
        meta=meta,
        dtype=out_dtype,
    )
    return trim_internal(y, depth, boundary="none")
