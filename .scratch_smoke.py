import dask
from dask.delayed import cond, delayed
from dask._task_spec import Conditional, DataNode, Task, TaskRef

# resurrection: shared dep released before condition resolves
calls = []
def rec(name, x=None):
    calls.append(name)
    return 'S' if x is None else x
S = delayed(rec)('S')
T = delayed(lambda s: s)(S)
C = delayed(lambda t: True)(T)
B = delayed(lambda s: rec('B', s))(S)
out = cond(C, B, 'x')
assert out.compute(scheduler='sync') == 'S'
print('resurrection calls:', calls)
assert calls.count('S') == 2, calls
assert calls.count('B') == 1

# unsupported scheduler: static execution of Conditional raises clear error
task = Conditional('out', TaskRef('c'), TaskRef('t'), TaskRef('f'))
try:
    task({'c': True})
except RuntimeError as e:
    assert 'conditional' in str(e).lower() or 'Conditional' in str(e)
    print('call error ok:', e)
else:
    raise AssertionError('no error')

# dask.core.get (execute_graph) has no dynamic support -> clear error
import dask.core
dsk = {
    'c': DataNode('c', True),
    't': Task('t', lambda: 1),
    'f': Task('f', lambda: 2),
    'out': Conditional('out', TaskRef('c'), TaskRef('t'), TaskRef('f')),
}
try:
    dask.core.get(dsk, 'out')
except RuntimeError as e:
    print('core.get error ok')
else:
    raise AssertionError('no error from core.get')

# pickle roundtrip
import pickle
t2 = pickle.loads(pickle.dumps(task))
assert t2 == task and t2.key == 'out'
print('pickle ok')

# tokenize deterministic
from dask.base import tokenize
assert tokenize(task) == tokenize(t2)
print('tokenize ok')
print('EDGE OK')

