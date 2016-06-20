import numpy as np

from chainer import cuda
from chainer import function
from chainer import Variable
from chainer.utils import type_check


class TypedList(object):

    def __init__(self, size, item_shape, dtype, inplace=True):
        self.size = size
        self.item_shape = item_shape
        self.dtype = dtype

        self.inplace = inplace
        data_shape = (size,) + item_shape
        self.data = Variable(np.zeros(data_shape, dtype=dtype))

    @property
    def raw(self):
        return self.data.data

    def _idx_guard(self, idxs):
        if isinstance(idxs, int):
            return np.array([idxs])
        elif isinstance(idxs, (list, tuple)):
            assert isinstance(idxs[0], int)
            return np.array(list(idxs))
        elif isinstance(idxs, (np.ndarray, cuda.ndarray)):
            return idxs

        raise ValueError("Typed list indices must be int, int seq, or ndarray")

    def __getitem__(self, idxs):
        self.data = TypedListGetitem(self)(self._idx_guard(idxs))
        return self

    def __setitem__(self, idxs, x):
        self.data = TypedListInsert(self)(x, self._idx_guard(idxs))
        return self


class TypedListGetitem(function.Function):

    def __init__(self, tl):
        self.tl = tl

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].ndim == 1)

    def forward(self, inputs):
        idxs, = inputs
        return self.tl.raw[idxs],

    def backward(self, inputs, grad_outputs):
        m = cuda.get_array_module(*inputs)
        idxs, = inputs

        g = m.zeros_like(self.tl.raw)
        assert g.dtype == idxs.dtype
        g[idxs] = grad_outputs
        return g,


class TypedListInsert(function.Function):

    def __init__(self, tl):
        self.tl = tl

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, idxs_type = in_types

        type_check.expect(x_type.ndim <= self.tl.raw.ndim)
        type_check.expect(idxs_type.ndim == 1)

    def forward(self, inputs):
        x, idxs = inputs

        if self.tl.inplace:
            ret = self.tl.raw
        else:
            ret = np.copy(self.tl.raw)
        ret[idxs] = x

        return ret,

    def backward(self, inputs, grad_outputs):
        x, idxs = inputs
        g, = grad_outputs

        ret = g[idxs]
        if ret.ndim > x.ndim:
            ret = ret.squeeze()
        return ret, None
