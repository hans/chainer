import unittest

import numpy as np

import chainer
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.structures import typed_list
from chainer.testing import condition


parameterizations = testing.product_dict(
    [{'list_size': 1},
     {'list_size': 2},
     {'list_size': 10}],
    [{'shape': (10,)},],
    [{'n_idxs': 0},
     {'n_idxs': 1},
     {'n_idxs': 2},
     {'n_idxs': 5},
     {'n_idxs': 10}],
)

@testing.parameterize(*parameterizations)
class TestTypedListInsert(unittest.TestCase):

    def setUp(self):
        self.dtype = np.float32
        self.tl = typed_list.TypedList(self.list_size, self.shape,
                self.dtype)

    def _make_x(self):
        if self.n_idxs == 0:
            return np.random.rand(*self.shape).astype(self.dtype)
        return np.random.rand(self.n_idxs, *self.shape).astype(self.dtype)

    def _make_idxs(self):
        if self.n_idxs == 0:
            return np.random.randint(self.list_size)
        elif self.n_idxs > self.list_size:
            return None
        return np.random.choice(self.list_size, size=self.n_idxs, replace=False)

    def check_forward(self, x, idx):
        x = chainer.Variable(x)
        self.tl[idx] = x

        gradient_check.assert_allclose(self.tl.raw[idx], x.data)

    @condition.repeat(5)
    def test_forward_cpu(self):
        idxs = self._make_idxs()
        if idxs is None:
            return
        self.check_forward(self._make_x(), idxs)

    def check_backward(self, x, idx, grad):
        idx = self.tl._idx_guard(idx)
        gradient_check.check_backward(typed_list.TypedListInsert(self.tl),
                (x, idx), grad)

    @condition.repeat(5)
    def test_backward_cpu(self):
        idxs = self._make_idxs()
        if idxs is None:
            return

        grad = np.random.rand(self.list_size, *self.shape).astype(self.dtype)

        self.check_backward(self._make_x(), idxs, grad)
