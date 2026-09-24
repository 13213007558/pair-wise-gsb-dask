"""A set of NumPy functions to apply per chunk"""

from __future__ import annotations

import contextlib
from collections.abc import Container, Iterable, Sequence
from functools import wraps
from numbers import Integral

import numpy as np
from tlz import concat

from dask.core import flatten


def keepdims_wrapper(a_callable):
    """
    A wrapper for functions that don't provide keepdims to ensure that they do.
    """

    @wraps(a_callable)
    def keepdims_wrapped_callable(x, axis=None, keepdims=None, *args, **kwargs):
        r = a_callable(x, *args, axis=axis, **kwargs)

        if not keepdims:
            return r

        axes = axis

        if axes is None:
            axes = range(x.ndim)

        if not isinstance(axes, (Container, Iterable, Sequence)):
            axes = [axes]

        r_slice = tuple()
        for each_axis in range(x.ndim):
            if each_axis in axes:
                r_slice += (None,)
            else:
                r_slice += (slice(None),)

        r = r[r_slice]

        return r

    return keepdims_wrapped_callable


# Wrap NumPy functions to ensure they provide keepdims.
sum = np.sum
prod = np.prod
min = np.min
max = np.max
argmin = keepdims_wrapper(np.argmin)
nanargmin = keepdims_wrapper(np.nanargmin)
argmax = keepdims_wrapper(np.argmax)
nanargmax = keepdims_wrapper(np.nanargmax)
any = np.any
all = np.all
nansum = np.nansum
nanprod = np.nanprod

nancumprod = np.nancumprod
nancumsum = np.nancumsum

nanmin = np.nanmin
nanmax = np.nanmax
mean = np.mean

with contextlib.suppress(AttributeError):
    nanmean = np.nanmean

var = np.var

with contextlib.suppress(AttributeError):
    nanvar = np.nanvar

std = np.std

with contextlib.suppress(AttributeError):
    nanstd = np.nanstd


def coarsen(reduction, x, axes, trim_excess=False, **kwargs):
    """Coarsen array by applying reduction to fixed size neighborhoods

    Parameters
    ----------
    reduction: function
        Reduction function (for example ``np.sum`` or ``np.mean``).

        The function must accept:

        - an ``array_like`` positional input,
        - an ``axis=`` keyword containing a tuple of axes,
        - and any extra ``**kwargs`` forwarded by ``coarsen``.

        In practice, NumPy-style reductions and Array-API-compatible
        reductions work well.
    x: np.ndarray
        Array to be coarsened
    axes: dict
        Mapping of axis to coarsening factor

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5, 6])
    >>> coarsen(np.sum, x, {0: 2})
    array([ 3,  7, 11])
    >>> coarsen(np.max, x, {0: 3})
    array([3, 6])

    Provide dictionary of scale per dimension

    >>> x = np.arange(24).reshape((4, 6))
    >>> x
    array([[ 0,  1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10, 11],
           [12, 13, 14, 15, 16, 17],
           [18, 19, 20, 21, 22, 23]])

    >>> coarsen(np.min, x, {0: 2, 1: 3})
    array([[ 0,  3],
           [12, 15]])

    You must avoid excess elements explicitly

    >>> x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    >>> coarsen(np.min, x, {0: 3}, trim_excess=True)
    array([1, 4])
    """
    # Insert singleton dimensions if they don't exist already
    for i in range(x.ndim):
        if i not in axes:
            axes[i] = 1

    if trim_excess:
        ind = tuple(
            slice(0, -(d % axes[i])) if d % axes[i] else slice(None, None)
            for i, d in enumerate(x.shape)
        )
        x = x[ind]

    # (10, 10) -> (5, 2, 5, 2)
    newshape = tuple(concat([(x.shape[i] // axes[i], axes[i]) for i in range(x.ndim)]))

    return reduction(x.reshape(newshape), axis=tuple(range(1, x.ndim * 2, 2)), **kwargs)


def trim(x, axes=None):
    """Trim boundaries off of array

    >>> x = np.arange(24).reshape((4, 6))
    >>> trim(x, axes={0: 0, 1: 1})
    array([[ 1,  2,  3,  4],
           [ 7,  8,  9, 10],
           [13, 14, 15, 16],
           [19, 20, 21, 22]])

    >>> trim(x, axes={0: 1, 1: 1})
    array([[ 7,  8,  9, 10],
           [13, 14, 15, 16]])
    """
    if isinstance(axes, Integral):
        axes = [axes] * x.ndim
    if isinstance(axes, dict):
        axes = [axes.get(i, 0) for i in range(x.ndim)]

    return x[tuple(slice(ax, -ax if ax else None) for ax in axes)]


def topk(a, k, axis, keepdims):
    """Chunk and combine function of topk

    Extract the k largest elements from a on the given axis.
    If k is negative, extract the -k smallest elements instead.
    Note that, unlike in the parent function, the returned elements
    are not sorted internally.
    """
    assert keepdims is True
    axis = axis[0]
    if abs(k) >= a.shape[axis]:
        return a

    a = np.partition(a, -k, axis=axis)
    k_slice = slice(-k, None) if k > 0 else slice(-k)
    return a[tuple(k_slice if i == axis else slice(None) for i in range(a.ndim))]


def topk_aggregate(a, k, axis, keepdims):
    """Final aggregation function of topk

    Invoke topk one final time and then sort the results internally.
    """
    assert keepdims is True
    a = topk(a, k, axis, keepdims)
    axis = axis[0]
    a = np.sort(a, axis=axis)
    if k < 0:
        return a
    return a[
        tuple(
            slice(None, None, -1) if i == axis else slice(None) for i in range(a.ndim)
        )
    ]


def argtopk_preprocess(a, idx):
    """Preparatory step for argtopk

    Put data together with its original indices in a tuple.
    """
    return a, idx


def argtopk(a_plus_idx, k, axis, keepdims):
    """Chunk and combine function of argtopk

    Extract the indices of the k largest elements from a on the given axis.
    If k is negative, extract the indices of the -k smallest elements instead.
    Note that, unlike in the parent function, the returned elements
    are not sorted internally.
    """
    assert keepdims is True
    axis = axis[0]

    if isinstance(a_plus_idx, list):
        a_plus_idx = list(flatten(a_plus_idx))
        a = np.concatenate([ai for ai, _ in a_plus_idx], axis)
        idx = np.concatenate(
            [np.broadcast_to(idxi, ai.shape) for ai, idxi in a_plus_idx], axis
        )
    else:
        a, idx = a_plus_idx

    if abs(k) >= a.shape[axis]:
        return a_plus_idx

    idx2 = np.argpartition(a, -k, axis=axis)
    k_slice = slice(-k, None) if k > 0 else slice(-k)
    idx2 = idx2[tuple(k_slice if i == axis else slice(None) for i in range(a.ndim))]
    return np.take_along_axis(a, idx2, axis), np.take_along_axis(idx, idx2, axis)


def argtopk_aggregate(a_plus_idx, k, axis, keepdims):
    """Final aggregation function of argtopk

    Invoke argtopk one final time, sort the results internally, drop the data
    and return the index only.
    """
    assert keepdims is True
    a_plus_idx = a_plus_idx if len(a_plus_idx) > 1 else a_plus_idx[0]
    a, idx = argtopk(a_plus_idx, k, axis, keepdims)
    axis = axis[0]

    idx2 = np.argsort(a, axis=axis)
    idx = np.take_along_axis(idx, idx2, axis)
    if k < 0:
        return idx
    return idx[
        tuple(
            slice(None, None, -1) if i == axis else slice(None) for i in range(idx.ndim)
        )
    ]


def _topk_stable_key(a, idx, k):
    """Structured (value, index) array used as the stable top-k selection key.

    The key is a total order: values compare with the usual NumPy semantics
    (NaN is the largest value; -0.0 and +0.0 compare equal), and ties are
    broken by the original position along the axis. For k > 0 (top k), the
    index is negated so that, among equal values, the *smallest* original
    index compares as the largest.
    """
    key = np.empty(a.shape, dtype=[("v", a.dtype), ("i", np.intp)])
    key["v"] = a
    key["i"] = idx if k < 0 else -idx
    return key


def _argtopk_stable_select(a, idx, k, axis):
    """Select the top-|k| candidates, breaking ties by original index.

    Return a (values, indices) tuple with the best abs(k) elements of a along
    axis, in unspecified internal order. The selection rule is the same at
    every step of a tree reduction, so intermediate candidates already obey
    the final stable ordering.
    """
    kk = abs(k)
    n = a.shape[axis]
    if kk == 0:
        empty = tuple(slice(0, 0) if i == axis else slice(None) for i in range(a.ndim))
        return a[empty], idx[empty]
    if kk >= n:
        return a, idx

    key = _topk_stable_key(a, idx, k)
    if k > 0:
        kth = n - kk
        k_slice = slice(n - kk, None)
    else:
        kth = kk - 1
        k_slice = slice(kk)
    part = np.argpartition(key, kth, axis=axis)
    part = part[tuple(k_slice if i == axis else slice(None) for i in range(a.ndim))]
    return np.take_along_axis(a, part, axis), np.take_along_axis(idx, part, axis)


def _argtopk_stable_sort(a, idx, k, axis):
    """Sort stable top-k candidates by value and, on ties, by original index.

    Sort by value descending (ascending original index on ties) if k > 0,
    and by value ascending (ascending original index on ties) if k < 0.
    """
    key = _topk_stable_key(a, idx, k)
    order = np.argsort(key, axis=axis)
    a = np.take_along_axis(a, order, axis)
    idx = np.take_along_axis(idx, order, axis)
    if k > 0:
        rev = tuple(
            slice(None, None, -1) if i == axis else slice(None) for i in range(a.ndim)
        )
        a = a[rev]
        idx = idx[rev]
    return a, idx


def argtopk_stable(a_plus_idx, k, axis, keepdims):
    """Chunk and combine function of argtopk with stable tie-breaking

    Like argtopk, but equal values are selected by their original position
    along the axis. This rule is applied to the intermediate candidates at
    every step of the tree reduction, so the final result does not depend on
    the chunking or on split_every.
    """
    assert keepdims is True
    axis = axis[0]

    if isinstance(a_plus_idx, list):
        a_plus_idx = list(flatten(a_plus_idx))
        a = np.concatenate([ai for ai, _ in a_plus_idx], axis)
        idx = np.concatenate(
            [np.broadcast_to(idxi, ai.shape) for ai, idxi in a_plus_idx], axis
        )
    else:
        a, idx = a_plus_idx

    return _argtopk_stable_select(a, idx, k, axis)


def argtopk_stable_aggregate(a_plus_idx, k, axis, keepdims):
    """Final aggregation function of argtopk with stable tie-breaking

    Invoke argtopk_stable one final time, sort the candidates by value and
    original index, drop the data and return the index only.
    """
    assert keepdims is True
    a_plus_idx = a_plus_idx if len(a_plus_idx) > 1 else a_plus_idx[0]
    a, idx = argtopk_stable(a_plus_idx, k, axis, keepdims)
    _, idx = _argtopk_stable_sort(a, idx, k, axis[0])
    return idx


def topk_stable_aggregate(a_plus_idx, k, axis, keepdims):
    """Final aggregation function of topk with stable tie-breaking

    Invoke argtopk_stable one final time, sort the candidates by value and
    original index, drop the index and return the data only. The selected
    elements are the same ones pointed to by argtopk_stable_aggregate.
    """
    assert keepdims is True
    a_plus_idx = a_plus_idx if len(a_plus_idx) > 1 else a_plus_idx[0]
    a, idx = argtopk_stable(a_plus_idx, k, axis, keepdims)
    a, _ = _argtopk_stable_sort(a, idx, k, axis[0])
    return a


def arange(start, stop, step, length, dtype, like=None):
    from dask.array.utils import arange_safe

    res = arange_safe(start, stop, step, dtype, like=like)
    return res[:-1] if len(res) > length else res


def linspace(start, stop, num, endpoint=True, dtype=None):
    from dask.array.core import Array

    if isinstance(start, Array):
        start = start.compute()

    if isinstance(stop, Array):
        stop = stop.compute()

    return np.linspace(start, stop, num, endpoint=endpoint, dtype=dtype)


def astype(x, astype_dtype=None, **kwargs):
    return x.astype(astype_dtype, **kwargs)


def view(x, dtype, order="C"):
    if order == "C":
        try:
            x = np.ascontiguousarray(x, like=x)
        except TypeError:
            x = np.ascontiguousarray(x)
        return x.view(dtype)
    else:
        try:
            x = np.asfortranarray(x, like=x)
        except TypeError:
            x = np.asfortranarray(x)
        return x.T.view(dtype).T


def slice_with_int_dask_array(x, idx, offset, x_size, axis):
    """Chunk function of `slice_with_int_dask_array_on_axis`.
    Slice one chunk of x by one chunk of idx.

    Parameters
    ----------
    x: ndarray, any dtype, any shape
        i-th chunk of x
    idx: ndarray, ndim=1, dtype=any integer
        j-th chunk of idx (cartesian product with the chunks of x)
    offset: ndarray, shape=(1, ), dtype=int64
        Index of the first element along axis of the current chunk of x
    x_size: int
        Total size of the x da.Array along axis
    axis: int
        normalized axis to take elements from (0 <= axis < x.ndim)

    Returns
    -------
    x sliced along axis, using only the elements of idx that fall inside the
    current chunk.
    """
    from dask.array.utils import asarray_safe, meta_from_array

    idx = asarray_safe(idx, like=meta_from_array(x))

    # Needed when idx is unsigned
    idx = idx.astype(np.int64)

    # Normalize negative indices
    idx = np.where(idx < 0, idx + x_size, idx)

    # A chunk of the offset dask Array is a numpy array with shape (1, ).
    # It indicates the index of the first element along axis of the current
    # chunk of x.
    idx = idx - offset

    # Drop elements of idx that do not fall inside the current chunk of x
    idx_filter = (idx >= 0) & (idx < x.shape[axis])
    idx = idx[idx_filter]

    # np.take does not support slice indices
    # return np.take(x, idx, axis)
    return x[tuple(idx if i == axis else slice(None) for i in range(x.ndim))]


def slice_with_int_dask_array_aggregate(idx, chunk_outputs, x_chunks, axis):
    """Final aggregation function of `slice_with_int_dask_array_on_axis`.
    Aggregate all chunks of x by one chunk of idx, reordering the output of
    `slice_with_int_dask_array`.

    Note that there is no combine function, as a recursive aggregation (e.g.
    with split_every) would not give any benefit.

    Parameters
    ----------
    idx: ndarray, ndim=1, dtype=any integer
        j-th chunk of idx
    chunk_outputs: ndarray
        concatenation along axis of the outputs of `slice_with_int_dask_array`
        for all chunks of x and the j-th chunk of idx
    x_chunks: tuple
        dask chunks of the x da.Array along axis, e.g. ``(3, 3, 2)``
    axis: int
        normalized axis to take elements from (0 <= axis < x.ndim)

    Returns
    -------
    Selection from all chunks of x for the j-th chunk of idx, in the correct
    order
    """
    # Needed when idx is unsigned
    idx = idx.astype(np.int64)

    # Normalize negative indices
    idx = np.where(idx < 0, idx + sum(x_chunks), idx)

    x_chunk_offset = 0
    chunk_output_offset = 0

    # Assemble the final index that picks from the output of the previous
    # kernel by adding together one layer per chunk of x
    # FIXME: this could probably be reimplemented with a faster search-based
    # algorithm
    idx_final = np.zeros_like(idx)
    for x_chunk in x_chunks:
        idx_filter = (idx >= x_chunk_offset) & (idx < x_chunk_offset + x_chunk)
        idx_cum = np.cumsum(idx_filter)
        idx_final += np.where(idx_filter, idx_cum - 1 + chunk_output_offset, 0)
        x_chunk_offset += x_chunk
        if idx_cum.size > 0:
            chunk_output_offset += idx_cum[-1]

    # np.take does not support slice indices
    # return np.take(chunk_outputs, idx_final, axis)
    return chunk_outputs[
        tuple(
            idx_final if i == axis else slice(None) for i in range(chunk_outputs.ndim)
        )
    ]


def getitem(obj, index):
    """Getitem function

    This function creates a copy of the desired selection for array-like
    inputs when the selection is smaller than half of the original array. This
    avoids excess memory usage when extracting a small portion from a large array.
    For more information, see
    https://numpy.org/doc/stable/reference/arrays.indexing.html#basic-slicing-and-indexing.

    Parameters
    ----------
    obj: ndarray, string, tuple, list
        Object to get item from.
    index: int, list[int], slice()
        Desired selection to extract from obj.

    Returns
    -------
    Selection obj[index]

    """
    try:
        result = obj[index]
    except IndexError as e:
        raise ValueError(
            "Array chunk size or shape is unknown. "
            "Possible solution with x.compute_chunk_sizes()"
        ) from e

    try:
        if not result.flags.owndata and obj.size >= 2 * result.size:
            result = result.copy()
    except AttributeError:
        pass

    return result
