""" Base class for all abstract domain elements. """

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, List, Iterator, Union

from torch import Tensor, nn
from torch.nn import functional as F

from diffabs.utils import valid_lb_ub


class AbsDom(ABC):
    """ A dispatcher to access different objects in each abstract domain implementation.
        Using module as dispatcher makes it harder for serialization.
    """
    name = 'to be overridden by each Dom'

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    @abstractmethod
    def __getattr__(self, name: str) -> object:
        """ Return the corresponding Ele, Conv, ReLU, Tanh, etc. object/caller in each implementation file. """
        raise NotImplementedError()  # must run in each implementation file, since the globals() are different
    pass


class AbsEle(ABC):
    """ The abstract element propagated throughout the layers. A series of computation rules is defined. """

    @classmethod
    @abstractmethod
    def by_intvl(cls, lb: Tensor, ub: Tensor) -> AbsEle:
        """ Abstract a box to abstract elements by its lower/upper bounds. """
        raise NotImplementedError()

    @classmethod
    def by_pt(cls, pt: Tensor) -> AbsEle:
        """ A degenerated abstraction that only contains one instance. """
        return cls.by_intvl(pt, pt)

    @abstractmethod
    def __iter__(self) -> Iterator[Tensor]:
        """ To register hooks in PyTorch, the arguments to forward() must be Tensor or tuple of Tensors, but not an
            AbsEle instance. To work around this, call *AbsEle to get a tuple of all information tensors as the
            arguments for forward(), and reconstruct AbsEle right after entering forward(). This requires the
            AbsEle to take only Tensors in the constructor.
        """
        raise NotImplementedError()

    def __getitem__(self, key):
        """ It may only need to compute some rows but not all in the abstract element. Select those rows from here. """
        raise NotImplementedError()

    @abstractmethod
    def size(self):
        """ Return the size of any concretized data point from this abstract element. """
        raise NotImplementedError()

    @abstractmethod
    def dim(self):
        """ Return the number of dimensions for any concretized data point from this abstract element. """
        raise NotImplementedError()

    @abstractmethod
    def lb(self) -> Tensor:
        """ Lower Bound. """
        raise NotImplementedError()

    @abstractmethod
    def ub(self) -> Tensor:
        """ Upper Bound. """
        raise NotImplementedError()

    def gamma(self) -> Tuple[Tensor, Tensor]:
        """ Transform the abstract elements back into Lower Bounds and Upper Bounds. """
        lb = self.lb()
        ub = self.ub()
        assert valid_lb_ub(lb, ub)
        return lb, ub

    # ===== Below are pre-defined operations that every abstract element must support. =====

    @abstractmethod
    def view(self, *shape) -> AbsEle:
        raise NotImplementedError()

    @abstractmethod
    def contiguous(self) -> AbsEle:
        raise NotImplementedError()

    def to_dense(self) -> AbsEle:
        return self

    def squeeze(self, dim=None) -> AbsEle:
        shape = list(self.size())
        if dim is not None and shape[dim] != 1:
            # nothing to squeeze
            return self

        if dim is None:
            while 1 in shape:
                shape.remove(1)
        else:
            shape = shape[:dim] + shape[dim+1:]
        return self.view(*shape)

    def unsqueeze(self, dim) -> AbsEle:
        if dim < 0:
            # following PyTorch doc
            dim = dim + self.dim() + 1

        shape = list(self.size())
        shape = shape[:dim] + [1] + shape[dim:]
        return self.view(*shape)

    @abstractmethod
    def transpose(self, dim0, dim1) -> AbsEle:
        raise NotImplementedError()

    @abstractmethod
    def matmul(self, other: Tensor) -> AbsEle:
        raise NotImplementedError()

    @abstractmethod
    def __add__(self, other) -> AbsEle:
        raise NotImplementedError()
    pass


class MetaFunc(object):
    """ All meta functions that can be used in properties definition. """

    def col_le_val(self, e: AbsEle, idx: int, threshold: float, mean: float = 0., range: float = 1.):
        """ Property: idx-th column value <= threshold.
            @mean and @range are for de-normalization since it is about absolute value.
        """
        raise NotImplementedError()

    def col_ge_val(self, e: AbsEle, idx: int, threshold: float, mean: float = 0., range: float = 1.):
        """ Property: idx-th column value >= threshold.
            @mean and @range are for de-normalization since it is about absolute value.
        """
        raise NotImplementedError()

    def cols_not_max(self, e: AbsEle, *idxs: int):
        """ Property: Forall idx-th column value is not maximal among all. """
        raise NotImplementedError()

    def cols_is_max(self, e: AbsEle, *idxs: int):
        """ Property: Exists idx-th column value is the maximal among all. """
        raise NotImplementedError()

    def cols_not_min(self, e: AbsEle, *idxs: int):
        """ Property: Forall idx-th column value is not minimal among all. """
        raise NotImplementedError()

    def cols_is_min(self, e: AbsEle, *idxs: int):
        """ Property: Exists idx-th column value is the minimal among all. """
        raise NotImplementedError()

    def labels_predicted(self, e: AbsEle, labels: Tensor):
        """ Property: Forall batched input, their prediction should match the corresponding label.
        :param labels: same number of batches as self
        """
        raise NotImplementedError()

    def labels_not_predicted(self, e: AbsEle, labels: Tensor):
        """ Property: Forall batched input, none of their prediction matches the corresponding label.
        :param labels: same number of batches as self
        """
        raise NotImplementedError()

    # ===== Finally, some utility functions shared by different domains. =====

    @staticmethod
    def _idxs_not(e: Union[Tensor, AbsEle], *idxs: int) -> List[int]:
        """ Validate and get other column indices that are not specified. """
        col_size = e.size()[-1]
        assert len(idxs) > 0 and all([0 <= i < col_size for i in idxs])
        assert len(set(idxs)) == len(idxs)  # no duplications
        others = [i for i in range(col_size) if i not in idxs]
        assert len(others) > 0
        return others
    pass


class AbsDist(MetaFunc):
    """ Distance of abstract element to the boundary of various properties, with the guarantee that: if the dist <= 0,
        then all concrete instances in this abstract element satisfy the property. This distance can also be used as the
        metric for optimization.

        Note, however, that this distance may not be the best candidate for optimization objective. This distance is
        basically the L1 norm loss in a regression problem, which is absolutely suboptimal for many problems. Hence, I
        didn't bother to keep the negative part of distances, they are reset to 0 by relu(). See AbsBlackSheep for a
        better mechanism of loss.
    """

    def col_le_val(self, e: AbsEle, idx: int, threshold: float, mean: float = 0., range: float = 1.) -> Tensor:
        t = e.ub()[..., idx]
        threshold = (threshold - mean) / range
        d = t - threshold
        return F.relu(d)

    def col_ge_val(self, e: AbsEle, idx: int, threshold: float, mean: float = 0., range: float = 1.) -> Tensor:
        t = e.lb()[..., idx]
        threshold = (threshold - mean) / range
        d = threshold - t
        return F.relu(d)
    pass


class AbsBlackSheep(MetaFunc):
    """ Rather than 'distance', it returns an instance regarding the boundary of various properties. If some part of
        the abstract element is violating the property, it will return one such instance (no guarantee of returning the
        worst, though), thus called black sheep; if all are safe, a safe instance will be returned.
    """
    # TODO
    pass


def forward_linear(layer: nn.Linear, e: AbsEle) -> AbsEle:
    """ The linear layer computation is shared among all affine abstract domains. """
    out = e.matmul(layer.weight.t())
    if layer.bias is not None:
        out += layer.bias
    return out
