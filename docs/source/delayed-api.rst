API
===

The ``dask.delayed`` interface consists of one function, ``delayed``:

- ``delayed`` wraps functions

   Wraps functions. Can be used as a decorator, or around function calls
   directly (i.e. ``delayed(foo)(a, b, c)``). Outputs from functions wrapped in
   ``delayed`` are proxy objects of type ``Delayed`` that contain a graph of
   all operations done to get to this result.

- ``delayed`` wraps objects

   Wraps objects. Used to create ``Delayed`` proxies directly.

- ``if_then_else`` selects a delayed branch

   Evaluates a delayed condition, then schedules only the selected branch.

``Delayed`` objects can be thought of as representing a key in the dask task
graph. A ``Delayed`` supports *most* python operations, each of which creates
another ``Delayed`` representing the result:

- Most operators (``*``, ``-``, and so on)
- Item access and slicing (``a[0]``)
- Attribute access (``a.size``)
- Method calls (``a.index(0)``)

Operations that aren't supported include:

- Mutating operators (``a += 1``)
- Mutating magic methods such as ``__setitem__``/``__setattr__`` (``a[0] = 1``, ``a.foo = 1``)
- Iteration. (``for i in a: ...``)
- Use as a predicate (``if a: ...``)

The last two points in particular mean that ``Delayed`` objects cannot be used
for ordinary Python control flow. In other words you can't iterate over a
``Delayed`` object, or use it directly in an ``if`` statement. Use
``if_then_else`` for an ``if`` whose condition or branches are delayed.
``Delayed`` objects can otherwise be used in the body of a loop or ``if``
statement.
Even with this limitation, many workflows can easily be parallelized.

.. currentmodule:: dask.delayed

.. autosummary::
   delayed
   if_then_else
   Delayed

.. autofunction:: delayed
.. autofunction:: if_then_else
.. autoclass:: Delayed
