from __future__ import annotations

from tlz import groupby, join

from dask.tokenize import tokenize


def barrier(*args):
    return None


def getitem(x, key):
    """Like :func:`operator.getitem`, but allows setting key using partial
    ``partial(chunk.getitem, key=key)
    """
    return x[key]


def foldby_combine2(combine, acc, x):
    return combine(acc, x[1])


def groupby_tasks_group_hash(x, hash, grouper):
    return hash(grouper(x)), x


def var_chunk(seq):
    squares, total, n = 0.0, 0.0, 0
    for x in seq:
        squares += x**2
        total += x
        n += 1
    return squares, total, n


def var_aggregate(x, ddof):
    squares, totals, counts = list(zip(*x))
    x2, x, n = float(sum(squares)), float(sum(totals)), sum(counts)
    result = (x2 / n) - (x / n) ** 2
    return result * n / (n - ddof)


def join_partition(keyfunc, npartitions, side, data):
    """Split a partition into buckets by join key for a hash shuffle.

    Records with equal keys are guaranteed to land in the same bucket,
    regardless of which process evaluates the task.  Keys must be hashable;
    unhashable keys raise a TypeError that points out the offending side.
    """
    out = [[] for _ in range(npartitions)]
    for x in data:
        key = keyfunc(x)
        try:
            hash(key)
        except TypeError:
            raise TypeError(
                "Join keys must be hashable, got unhashable key of type "
                f"{type(key).__name__!r} on the {side} side of the join"
            ) from None
        out[int(tokenize(key), 16) % npartitions].append(x)
    return out


def join_chunk(on_other, other, on_self, data, how, missing):
    """Join two co-located partitions.

    Emits ``(other_item, self_item)`` pairs, consistent with ``Bag.join``.
    With ``how="left"`` every element of ``data`` (the left, i.e. ``self``,
    collection) is emitted exactly once even without a match, paired with
    ``missing``.
    """
    if how == "inner":
        return list(join(on_other, other, on_self, data))
    groups = groupby(on_other, other)
    result = []
    for x in data:
        matches = groups.get(on_self(x))
        if matches:
            result.extend((match, x) for match in matches)
        else:
            result.append((missing, x))
    return result
