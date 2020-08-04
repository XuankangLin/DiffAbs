""" Implement the DeepPoly abstract domain (POPL'19) based on PyTorch.

    Note that due to the nature of DeepPoly, both ReLU, Conv, and Pool layers may take huge amount of memory.
    I have slightly more memory friendly version of Conv and Pool, but ReLU is sometimes the memory eater,
    so it's not that useful to optimize the other two alone. Perhaps just use smaller batch_size in training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union, Tuple, List, Iterator

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from diffabs.abs import AbsDom, AbsEle, AbsDist, AbsBlackSheep, forward_linear
from diffabs.utils import valid_lb_ub, divide_pos_neg


class Dom(AbsDom):
    name = Path(__file__).with_suffix('').name  # use file name (without extension) as domain name

    def __getattr__(self, name: str) -> object:
        assert name in globals()
        return eval(name)
    pass


class Ele(AbsEle):
    def __init__(self, lcoef: Tensor, lcnst: Tensor, ucoef: Tensor, ucnst: Tensor, dlb: Tensor, dub: Tensor):
        """
        :param lcoef: Lower Coefficients in constraint on input delta bounds, with size Batch x FlatDim0 x Dims...
        :param ucoef: Upper Coefficients in constraint on input delta bounds, with size Batch x FlatDim0 x Dims...
        :param lcnst: Lower Constants in constraint on input delta bounds, with size Batch x 1 x Dims...
        :param ucnst: Upper Constants in constraint on input delta bounds, with size Batch x 1 x Dims...
        :param dlb: Concrete Lower Bounds of input deltas, with size Batch x 1 x FlatDim0
        :param dub: Concrete Upper Bounds of input deltas, with size Batch x 1 x FlatDim0
        """
        assert lcoef.device == lcnst.device == ucoef.device == ucnst.device == dlb.device == dub.device
        self._lcoef = lcoef
        self._ucoef = ucoef
        self._lcnst = lcnst
        self._ucnst = ucnst
        self.dlb = dlb
        self.dub = dub
        return

    @classmethod
    def by_intvl(cls, lb: Tensor, ub: Tensor, *args, **kwargs) -> Ele:
        assert lb.device == ub.device
        assert valid_lb_ub(lb, ub)

        base = lb
        dlb = torch.zeros_like(lb)
        dub = ub - lb

        size = base.size()
        batch_size = size[0]
        unflat_sizes = size[1:]
        flat_size = torch.tensor(unflat_sizes).prod().item()  # Batch x ...

        ''' Coefficients: Batch x FlatDim0 x Dims...
            Constants: Batch x 1 x Dims..., as last column (for constant biases) in linear formula.
            These formats are better for affine transformation in linear layer.
        '''
        coefs = torch.eye(flat_size, device=base.device)
        coefs = coefs.expand(batch_size, flat_size, flat_size)  # Batch x FlatDim0 x FlatDim0
        coefs = coefs.view(batch_size, flat_size, *unflat_sizes)  # Batch x FlatDim0 x Dims...
        cnsts = base.unsqueeze(dim=1)  # Batch x 1 x Dims...

        dlb = dlb.view(batch_size, flat_size)  # Batch x FlatDim0
        dub = dub.view(batch_size, flat_size)
        dlb = dlb.unsqueeze(dim=1)  # Batch x 1 x FlatDim0, unsqueezed for easier bmm() in getting lb/ub
        dub = dub.unsqueeze(dim=1)
        return Ele(coefs, cnsts, coefs, cnsts, dlb, dub)

    def __iter__(self) -> Iterator[Tensor]:
        return iter((self._lcoef, self._lcnst, self._ucoef, self._ucnst, self.dlb, self.dub))

    def __getitem__(self, key):
        return Ele(self._lcoef[key], self._lcnst[key], self._ucoef[key], self._ucnst[key], self.dlb[key], self.dub[key])

    def size(self):
        return self._lcnst.squeeze(dim=1).size()

    def dim(self):
        return self._lcnst.squeeze(dim=1).dim()

    @staticmethod
    def lb_of(coefs: Tensor, cnsts: Tensor, dlb: Tensor, dub: Tensor) -> Tensor:
        """ Derive Lower Bound of a coefficients tensor by resolving input delta bounds.
        :param coefs: Tensor of size Batch x FlatDim0 x Dims...
        :param cnsts: Tensor of size Batch x 1 x Dims...
        :param dlb: Concrete Lower Bounds of input deltas, with size Batch x 1 x FlatDim0
        :param dub: Concrete Upper Bounds of input deltas, with size Batch x 1 x FlatDim0
        :rtype: Tensor of size Batch x Dims...
        """
        size = coefs.size()
        batch_size = size[0]  # Batch
        flat_size = size[1]  # FlatDim0
        unflat_sizes = size[2:]  # Dims...

        coefs = coefs.contiguous().view(batch_size, flat_size, -1)  # Batch x FlatDim0 x Dim'
        cnsts = cnsts.contiguous().view(batch_size, 1, -1)  # Batch x 1 x Dim'
        pos_coefs, neg_coefs = divide_pos_neg(coefs)  # Batch x FlatDim0 x FlatDim0

        lb_pos = torch.bmm(dlb, pos_coefs)  # Batch x 1 x FlatDim0 <BMM> Batch x FlatDim0 x FlatDim0
        lb_neg = torch.bmm(dub, neg_coefs)  # => Batch x 1 x FlatDim0

        lb = lb_pos + lb_neg  # Batch x 1 x FlatDim0
        res = lb + cnsts  # Batch x 1 x FlatDim0
        res = res.squeeze(dim=1)  # Batch x FlatDim0
        return res.view(batch_size, *unflat_sizes)  # Batch x Dims...

    def ub_of(self, coefs: Tensor, cnsts: Tensor, dlb: Tensor, dub: Tensor) -> Tensor:
        """ Derive Upper Bound of a coefficients tensor by resolving input delta bounds.
        :param coefs: Tensor of size Batch x FlatDim0 x Dims...
        :param cnsts: Tensor of size Batch x 1 x Dims...
        :param dlb: Concrete Lower Bounds of input deltas, with size Batch x 1 x FlatDim0
        :param dub: Concrete Upper Bounds of input deltas, with size Batch x 1 x FlatDim0
        :rtype: Tensor of size Batch x Dims...
        """
        size = coefs.size()
        batch_size = size[0]  # Batch
        flat_size = size[1]  # FlatDim0
        unflat_sizes = size[2:]  # Dims...

        coefs = coefs.contiguous().view(batch_size, flat_size, -1)  # Batch x FlatDim0 x Dim'
        cnsts = cnsts.contiguous().view(batch_size, 1, -1)  # Batch x 1 x Dim'
        pos_coefs, neg_coefs = divide_pos_neg(coefs)  # Batch x FlatDim0 x FlatDim0

        ub_pos = torch.bmm(dub, pos_coefs)  # Batch x 1 x FlatDim0 <BMM> Batch x FlatDim0 x FlatDim0
        ub_neg = torch.bmm(dlb, neg_coefs)  # => Batch x 1 x FlatDim0

        ub = ub_pos + ub_neg  # Batch x 1 x FlatDim0
        res = ub + cnsts  # Batch x 1 x FlatDim0
        res = res.squeeze(dim=1)  # Batch x FlatDim0
        return res.view(batch_size, *unflat_sizes)  # Batch x Dims...

    def lb(self) -> Tensor:
        try:
            return self._lb
        except AttributeError:
            # cache, assuming no changes to those coefficients
            self._lb = self.lb_of(self._lcoef, self._lcnst, self.dlb, self.dub)
            return self._lb

    def ub(self) -> Tensor:
        try:
            return self._ub
        except AttributeError:
            # cache, assuming no changes to those coefficients
            self._ub = self.ub_of(self._ucoef, self._ucnst, self.dlb, self.dub)
            return self._ub

    def view(self, *shape) -> Ele:
        assert len(shape) > 1

        flat_size = self._lcoef.size()[1]
        shape_coefs = list(shape)
        shape_coefs.insert(1, flat_size)  # Batch x FlatDim0 x ...
        shape_cnsts = list(shape)
        shape_cnsts.insert(1, 1)  # Batch x 1 x ...

        newl_coefs = self._lcoef.view(*shape_coefs)
        newl_cnsts = self._lcnst.view(*shape_cnsts)
        newu_coefs = self._ucoef.view(*shape_coefs)
        newu_cnsts = self._ucnst.view(*shape_cnsts)
        return Ele(newl_coefs, newl_cnsts, newu_coefs, newu_cnsts, self.dlb, self.dub)

    def contiguous(self) -> Ele:
        return Ele(self._lcoef.contiguous(), self._lcnst.contiguous(),
                   self._ucoef.contiguous(), self._ucnst.contiguous(),
                   self.dlb, self.dub)

    def transpose(self, dim0, dim1) -> Ele:
        if dim0 == 0 or dim1 == 0:
            raise ValueError('Who would transpose the batch dimension?!')

        dim0 += 1
        dim1 += 1
        return Ele(
            self._lcoef.transpose(dim0, dim1),
            self._lcnst.transpose(dim0, dim1),
            self._ucoef.transpose(dim0, dim1),
            self._ucnst.transpose(dim0, dim1),
            self.dlb, self.dub
        )

    def matmul(self, weights: Tensor) -> Ele:
        """ Basically,
                L' = max(0, w) * L + min(0, w) * U
                U' = max(0, w) * U + min(0, w) * L
        """
        pos_ws, neg_ws = divide_pos_neg(weights)

        def _new_lower(csl, csu):
            """
            :param csl: coefficients for lower
            :param csu: coefficients for upper
            """
            return csl.matmul(pos_ws) + csu.matmul(neg_ws)  # Batch x FlatDim0 (or 1) x ... x Dim_in <matmul> Dim_in x Dim_out

        newl_coef = _new_lower(self._lcoef, self._ucoef)
        newl_cnst = _new_lower(self._lcnst, self._ucnst)

        def _new_upper(csl, csu):
            """
            :param csl: coefficients for lower
            :param csu: coefficients for upper
            """
            return csu.matmul(pos_ws) + csl.matmul(neg_ws)  # Batch x FlatDim0 (or 1) x ... x Dim_in <matmul> Dim_in x Dim_out

        newu_coef = _new_upper(self._lcoef, self._ucoef)
        newu_cnst = _new_upper(self._lcnst, self._ucnst)

        return Ele(newl_coef, newl_cnst, newu_coef, newu_cnst, self.dlb, self.dub)

    def __add__(self, other: Union[Ele, int, float, Tensor]) -> Ele:
        """ Addition only changes the constant part in coefficients. """
        if isinstance(other, Ele):
            assert torch.equal(self.dlb, other.dlb) and torch.equal(self.dub, other.dub)
            return Ele(self._lcoef + other._lcoef, self._lcnst + other._lcnst,
                       self._ucoef + other._ucoef, self._ucnst + other._ucnst, self.dlb, self.dub)
        else:
            biases = other
            return Ele(self._lcoef, self._lcnst + biases, self._ucoef, self._ucnst + biases, self.dlb, self.dub)

    def __radd__(self, other) -> Ele:
        return self.__add__(other)

    def __mul__(self, flt) -> Ele:
        if isinstance(flt, Tensor) and flt.dim() == 1 and flt.shape[0] == self.size()[-1]:
            # each output vector dimension has its own factor
            flt_coefs = flt.expand_as(self._lcoef)
            flt_cnsts = flt.expand_as(self._lcnst)
            if (flt > 0).all():
                return Ele(self._lcoef * flt_coefs, self._lcnst * flt_cnsts,
                           self._ucoef * flt_coefs, self._ucnst * flt_cnsts,
                           self.dlb, self.dub)
            elif (flt < 0).all():
                return Ele(self._ucoef * flt_coefs, self._ucnst * flt_cnsts,
                           self._lcoef * flt_coefs, self._lcnst * flt_cnsts,
                           self.dlb, self.dub)
            else:
                raise NotImplementedError()

        elif not (isinstance(flt, float) or isinstance(flt, int)):
            raise ValueError('Unsupported multiplication with', str(flt), type(flt))

        flt = float(flt)
        if flt >= 0:
            return Ele(self._lcoef * flt, self._lcnst * flt, self._ucoef * flt, self._ucnst * flt, self.dlb, self.dub)
        else:
            return Ele(self._ucoef * flt, self._ucnst * flt, self._lcoef * flt, self._lcnst * flt, self.dlb, self.dub)

    def __rmul__(self, flt) -> Ele:
        return self.__mul__(flt)
    pass


class Dist(AbsDist):
    """ DeepPoly domain is relational, thus it can be computed symbolically beforehand. """
    def __init__(self, eps: float = 1e-5):
        """
        :param eps: add to break the tie when choosing max/min.
        """
        self.eps = eps
        return

    def cols_not_max(self, e: Ele, *idxs: int) -> Tensor:
        """ Intuitively, always-not-max => exists col . target < col is always true.
            Therefore, target_col.UB() - other_col.LB() should < 0, if not, that is the distance.
            As long as some of the others < 0, it's OK (i.e., min).
        """
        others_idxs = self._idxs_not(e, *idxs)
        others_coef = e._lcoef[:, :, others_idxs]  # Batch x Dim0 x (Dim-|idxs|)
        others_cnst = e._lcnst[:, :, others_idxs]  # Batch x 1 x (Dim-|idxs|)

        res = []
        for i in idxs:
            target_coef = e._ucoef[:, :, [i]]  # Batch x Dim0 x 1
            target_cnst = e._ucnst[:, :, [i]]  # Batch x 1 x 1
            diff_coefs = target_coef - others_coef  # will broadcast
            diff_cnsts = target_cnst - others_cnst

            diff = e.ub_of(diff_coefs, diff_cnsts, e.dlb, e.dub)  # Batch x (Dim-|ids|)
            diff = F.relu(diff + self.eps)
            mins, _ = torch.min(diff, dim=-1)
            res.append(mins)
        return sum(res)

    def cols_is_max(self, e: Ele, *idxs: int) -> Tensor:
        """ Intuitively, some-is-max => exists target . target > all_others is always true.
            Therefore, other_col.UB() - target_col.LB() should < 0, if not, that is the distance.
            All of the others should be accounted (i.e., max).
        """
        others_idxs = self._idxs_not(e, *idxs)
        others_coef = e._ucoef[:, :, others_idxs]  # Batch x Dim0 x (Dim-|idxs|)
        others_cnst = e._ucnst[:, :, others_idxs]  # Batch x 1 x (Dim-|idxs|)

        res = []
        for i in idxs:
            target_coef = e._lcoef[:, :, [i]]  # Batch x Dim0 x 1
            target_cnst = e._lcnst[:, :, [i]]  # Batch x 1 x 1
            diff_coefs = others_coef - target_coef  # will broadcast
            diff_cnsts = others_cnst - target_cnst

            diffs = e.ub_of(diff_coefs, diff_cnsts, e.dlb, e.dub)  # Batch x (Dim-|ids|)
            diffs = F.relu(diffs + self.eps)
            res.append(diffs)

        if len(idxs) == 1:
            all_diffs = res[0]
        else:
            all_diffs = torch.stack(res, dim=-1)
            all_diffs, _ = torch.min(all_diffs, dim=-1)  # it's OK to have either one to be max, thus use torch.min()

        # then it needs to surpass everybody else, thus use torch.max() for maximum distance
        diffs, _ = torch.max(all_diffs, dim=-1)
        return diffs

    def cols_not_min(self, e: Ele, *idxs: int) -> Tensor:
        """ Intuitively, always-not-min => exists col . col < target is always true.
            Therefore, other_col.UB() - target_col.LB() should < 0, if not, that is the distance.
            As long as some of the others < 0, it's OK (i.e., min).
        """
        others_idxs = self._idxs_not(e, *idxs)
        others_coef = e._ucoef[:, :, others_idxs]  # Batch x Dim0 x (Dim-|idxs|)
        others_cnst = e._ucnst[:, :, others_idxs]  # Batch x 1 x (Dim-|idxs|)

        res = []
        for i in idxs:
            target_coef = e._lcoef[:, :, [i]]  # Batch x Dim0 x 1
            target_cnst = e._lcnst[:, :, [i]]  # Batch x 1 x 1
            diff_coefs = others_coef - target_coef  # will broadcast
            diff_cnsts = others_cnst - target_cnst

            diff = e.ub_of(diff_coefs, diff_cnsts, e.dlb, e.dub)  # Batch x (Dim-|ids|)
            diff = F.relu(diff + self.eps)
            mins, _ = torch.min(diff, dim=-1)
            res.append(mins)
        return sum(res)

    def cols_is_min(self, e: Ele, *idxs: int) -> Tensor:
        """ I am implementing a hinge-loss version here for distance to safety. One alternative is to collect other
            columns' LB and apply to CrossEntropy style loss. However, that way it may lose information of the relation
            between two specific columns (using approximated LBs but not UB_i - LB_j).
        """
        others_idxs = self._idxs_not(e, *idxs)
        others_coef = e._lcoef[:, :, others_idxs]  # Batch x Dim0 x (Dim-|idxs|)
        others_cnst = e._lcnst[:, :, others_idxs]  # Batch x 1 x (Dim-|idxs|)

        res = []
        for i in idxs:
            target_coef = e._ucoef[:, :, [i]]  # Batch x Dim0 x 1
            target_cnst = e._ucnst[:, :, [i]]  # Batch x 1 x 1
            diff_coefs = target_coef - others_coef  # will broadcast
            diff_cnsts = target_cnst - others_cnst

            diffs = e.ub_of(diff_coefs, diff_cnsts, e.dlb, e.dub)  # Batch x (Dim-|ids|)
            diffs = F.relu(diffs + self.eps)
            res.append(diffs)

        if len(idxs) == 1:
            all_diffs = res[0]
        else:
            all_diffs = torch.stack(res, dim=-1)
            all_diffs, _ = torch.min(all_diffs, dim=-1)  # it's OK to have either one to be min, thus use torch.min()

        # then it needs to surpass everybody else, thus use torch.max() for maximum distance
        diffs, _ = torch.max(all_diffs, dim=-1)
        return diffs
    pass


class BlackSheep(AbsBlackSheep):
    def labels_predicted(self, e: Ele, labels: Tensor) -> Tensor:
        """ Intuitively, this is specifying a label_is_max for every input abstraction. """
        # TODO to review again
        res = []
        for i in range(len(labels)):
            cat = labels[i]
            piece = e[[i]]

            # default lb-ub or ub-lb doesn't know that target domain has distance 0, so specify that explicitly
            lefts_coef = piece._ucoef[..., :cat]
            lefts_cnst = piece._ucnst[..., :cat]
            rights_coef = piece._ucoef[..., cat+1:]
            rights_cnst = piece._ucnst[..., cat+1:]
            target_coef = piece._lcoef[..., [cat]]  # Batch x FlatDim0 x 1
            target_cnst = piece._lcnst[..., [cat]]  # Batch x 1 x 1

            full_coef = torch.cat((lefts_coef, target_coef, rights_coef), dim=-1)  # Batch x FlatDim0 x Dims
            full_cnst = torch.cat((lefts_cnst, target_cnst, rights_cnst), dim=-1)

            diff_coef = full_coef - target_coef  # will broadcast
            diff_cnst = full_cnst - target_cnst
            diffs = e.ub_of(diff_coef, diff_cnst, piece.dlb, piece.dub)  # Batch x Dims
            # no need to ReLU here, negative values are also useful
            res.append(diffs)

        res = torch.cat(res, dim=0)
        return res

    def labels_not_predicted(self, e: Ele, labels: Tensor) -> Tensor:
        """ Intuitively, this is specifying a label_not_max for every input abstraction.
        :param label: same number of batches as self
        """
        res = []
        for i in range(len(labels)):
            cat = labels[i]
            piece = e[[i]]

            # default lb-ub or ub-lb doesn't know that target domain has distance 0, so specify that explicitly
            lefts_coef = piece._lcoef[..., :cat]
            lefts_cnst = piece._lcnst[..., :cat]
            rights_coef = piece._lcoef[..., cat+1:]
            rights_cnst = piece._lcnst[..., cat+1:]
            target_coef = piece._ucoef[..., [cat]]  # Batch x FlatDim0 x 1
            target_cnst = piece._ucnst[..., [cat]]  # Batch x 1 x 1

            full_coef = torch.cat((lefts_coef, target_coef, rights_coef), dim=-1)  # Batch x FlatDim0 x Dims
            full_cnst = torch.cat((lefts_cnst, target_cnst, rights_cnst), dim=-1)

            diff_coef = target_coef - full_coef  # will broadcast
            diff_cnst = target_cnst - full_cnst
            diffs = e.ub_of(diff_coef, diff_cnst, piece.dlb, piece.dub)  # Batch x Dims
            # no need to ReLU here, negative values are also useful
            res.append(diffs)

        res = torch.cat(res, dim=0)
        # TODO
        raise NotImplementedError('To use this as distance, it has to have target category not being max, ' +
                                  'thus use torch.min(dim=-1) then ReLU().')
        return res
    pass


# ===== Below are customized layers that can take and propagate abstract elements. =====


class Linear(nn.Linear):
    def __str__(self):
        return f'{Dom.name}.' + super().__str__()

    @classmethod
    def from_module(cls, src: nn.Linear) -> Linear:
        with_bias = src.bias is not None
        new_lin = Linear(src.in_features, src.out_features, with_bias)
        new_lin.load_state_dict(src.state_dict())
        return new_lin

    def export(self) -> nn.Linear:
        with_bias = self.bias is not None
        lin = nn.Linear(self.in_features, self.out_features, with_bias)
        lin.load_state_dict(self.state_dict())
        return lin

    def forward(self, *ts: Union[Tensor, Ele]) -> Union[Tensor, Ele, Tuple[Tensor, ...]]:
        """
        :param ts: either Tensor, Ele, or Ele tensors
        :rtype: corresponding to inputs, Tensor for Tensor, Ele for Ele, Ele tensors for Ele tensors
        """
        input_is_ele = True
        if len(ts) == 1:
            if isinstance(ts[0], Tensor):
                return super().forward(ts[0])  # plain tensor, no abstraction
            elif isinstance(ts[0], Ele):
                e = ts[0]  # abstract element
            else:
                raise ValueError(f'Not supported argument type {type(ts[0])}.')
        else:
            input_is_ele = False
            e = Ele(*ts)  # reconstruct abstract element

        out = forward_linear(self, e)
        return out if input_is_ele else tuple(out)
    pass


class Conv2d(nn.Conv2d):
    """ Conv2d actually shows a disadvantage of the DeepPoly implementation here, where the memory usage during training
        could be huge. The reason is perhaps in the matmul() part. Even if I use as_strided() to save the memory usage
        during conv enumeration, when doing matmul() for Ele, it still needs to _ * pos_ws + _ * neg_ws, so 4 copies of
        gradients still have to be stored for backprop. This may be alleviated by using center symmetric domains like
        Zonotope where the computation only applies to the numbers once.
    """
    def __str__(self):
        return f'{Dom.name}.' + super().__str__()

    def forward(self, *ts: Union[Tensor, Ele]) -> Union[Tensor, Ele, Tuple[Tensor, ...]]:
        """ I have to implement the forward computation by myself, because F.conv2d() requires input to be Tensors.
        :param ts: either Tensor, Ele, or Ele tensors
        :rtype: corresponding to inputs, Tensor for Tensor, Ele for Ele, Ele tensors for Ele tensors
        """
        input_is_ele = True
        if len(ts) == 1:
            if isinstance(ts[0], Tensor):
                return super().forward(ts[0])  # plain tensor, no abstraction
            elif isinstance(ts[0], Ele):
                e = ts[0]  # abstract element
            else:
                raise ValueError(f'Not supported argument type {type(ts[0])}.')
        else:
            input_is_ele = False
            e = Ele(*ts)  # reconstruct abstract element

        ''' See 'https://github.com/vdumoulin/conv_arithmetic' for animated illustrations.
            It's not hard to support them, but we just don't need that right now.
        '''
        if self.dilation != (1, 1):
            raise NotImplementedError(f'Unsupported dilation {self.dilation}')
        if self.groups != 1:
            raise NotImplementedError(f'Unsupported groups {self.groups}')

        assert e.dim() == 4
        img_b, flat_size, img_c, img_h, img_w = e._lcoef.size()  # Batch x FlatDim0 x C x H x W

        fil_h, fil_w = self.kernel_size
        pad_h, pad_w = self.padding
        stride_h, stride_w = self.stride

        # formula: (W - F + 2P) / S + 1
        cnt_h = (img_h - fil_h + 2 * pad_h) / stride_h + 1
        cnt_w = (img_w - fil_w + 2 * pad_w) / stride_w + 1
        assert int(cnt_h) == cnt_h and int(cnt_w) == cnt_w, "img and filter dimensions don't fit?"
        cnt_h = int(cnt_h)
        cnt_w = int(cnt_w)

        ''' Pad the original image just in case (this is different for each abstract domain).
            First pad the left and right (width), then pad the top and bottom (height).
            ########### 2 ##########
            ## 1 ##  center  ## 1 ##
            ########### 2 ##########
        '''
        def _pad(orig: Tensor) -> Tensor:
            dim1 = orig.size()[1]  # either flat_size for coefs or 1 for cnsts
            if pad_w > 0:
                zs = torch.zeros(img_b, dim1, img_c, img_h, pad_w, device=orig.device)
                orig = torch.cat((zs, orig, zs), dim=-1)
            if pad_h > 0:
                zs = torch.zeros(img_b, dim1, img_c, pad_h, img_w + 2 * pad_w, device=orig.device)  # width has increased
                orig = torch.cat((zs, orig, zs), dim=-2)
            return orig

        # utils.pp_cuda_mem('Conv: Before padding')

        full_lb_coefs = _pad(e._lcoef)
        full_lb_cnsts = _pad(e._lcnst)
        full_ub_coefs = _pad(e._ucoef)
        full_ub_cnsts = _pad(e._ucnst)

        # utils.pp_cuda_mem('Conv: After padding')
        ''' The following code is faster that _mem_efficient_conv() (4.15s vs 5.74s)
            but allocates/caches more memory (max 10.2GB vs max 4.4GB).
        '''
        out = self._conv(e, img_b, flat_size, cnt_h, cnt_w, fil_h, fil_w, stride_h, stride_w,
                              full_lb_coefs, full_lb_cnsts, full_ub_coefs, full_ub_cnsts)
        return out if input_is_ele else tuple(out)

    def _conv(self, e, img_b, flat_size, cnt_h, cnt_w, fil_h, fil_w, stride_h, stride_w,
                   full_lb_coefs, full_lb_cnsts, full_ub_coefs, full_ub_cnsts) -> Ele:
        """ Collect all filtered sub-images in a large batch. """
        filtered_lb_coefs, filtered_lb_cnsts = [], []
        filtered_ub_coefs, filtered_ub_cnsts = [], []
        for i in range(cnt_h):
            row_lb_coefs, row_lb_cnsts = [], []
            row_ub_coefs, row_ub_cnsts = [], []
            for j in range(cnt_w):
                h_start = i * stride_h
                h_end = h_start + fil_h
                w_start = j * stride_w
                w_end = w_start + fil_w

                sub_lb_coefs = full_lb_coefs[..., h_start : h_end, w_start : w_end]  # Batch x FlatDim0 x InC x FilterH x FilterW
                sub_lb_cnsts = full_lb_cnsts[..., h_start : h_end, w_start : w_end]
                sub_ub_coefs = full_ub_coefs[..., h_start : h_end, w_start : w_end]
                sub_ub_cnsts = full_ub_cnsts[..., h_start : h_end, w_start : w_end]

                row_lb_coefs.append(sub_lb_coefs)
                row_lb_cnsts.append(sub_lb_cnsts)
                row_ub_coefs.append(sub_ub_coefs)
                row_ub_cnsts.append(sub_ub_cnsts)

            row_lb_coefs = torch.stack(row_lb_coefs, dim=2)  # dim=2: right after Batch x FlatDim0 x ...
            row_lb_cnsts = torch.stack(row_lb_cnsts, dim=2)  # Now Batch x FlatDim0 x OutW x InC x FilterH x FilterW FIXME
            row_ub_coefs = torch.stack(row_ub_coefs, dim=2)
            row_ub_cnsts = torch.stack(row_ub_cnsts, dim=2)
            filtered_lb_coefs.append(row_lb_coefs)
            filtered_lb_cnsts.append(row_lb_cnsts)
            filtered_ub_coefs.append(row_ub_coefs)
            filtered_ub_cnsts.append(row_ub_cnsts)

        filtered_lb_coefs = torch.stack(filtered_lb_coefs, dim=2)  # dim=2: right after Batch x FlatDim0 x ... again
        filtered_lb_cnsts = torch.stack(filtered_lb_cnsts, dim=2)  # Now Batch x FlatDim0 x OutH x OutW x InC x FilterH x FilterW FIXME
        filtered_ub_coefs = torch.stack(filtered_ub_coefs, dim=2)
        filtered_ub_cnsts = torch.stack(filtered_ub_cnsts, dim=2)

        # utils.pp_cuda_mem('Conv: After conv enumeration')

        ''' I also tried to use torch.unfold() to do the job, but it seems there's no memory gain and it's actually
            slower than doing unfold by myself...
        '''
        # def _unfold(cs):
        #     # Batch' x NumFilPixels x NumOutPixels
        #     return F.unfold(cs, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        #
        # num_fil_pixels = img_c * fil_h * fil_w  # number of pixels to matmul later
        # num_out_pixels = cnt_h * cnt_w  # number of outputing image pixels to store
        # filtered_lb_coefs = _unfold(full_lb_coefs)
        # filtered_lb_cnsts = _unfold(full_lb_cnsts)
        # filtered_ub_coefs = _unfold(full_ub_coefs)
        # filtered_ub_cnsts = _unfold(full_ub_cnsts)
        #
        # # pp_cuda_mem('Conv: After unfold enumeration')
        #
        # filtered_lb_coefs = filtered_lb_coefs.view(img_b, flat_size, num_fil_pixels, num_out_pixels)
        # filtered_lb_cnsts = filtered_lb_cnsts.view(img_b, 1, num_fil_pixels, num_out_pixels)
        # filtered_ub_coefs = filtered_ub_coefs.view(img_b, flat_size, num_fil_pixels, num_out_pixels)
        # filtered_ub_cnsts = filtered_ub_cnsts.view(img_b, 1, num_fil_pixels, num_out_pixels)

        ws = self.weight.view(self.out_channels, -1).t()  # (InC * FilterH * FilterW) x OutC
        newe = Ele(filtered_lb_coefs, filtered_lb_cnsts, filtered_ub_coefs, filtered_ub_cnsts, e.dlb, e.dub)
        # reshape everything to directly apply matmul (slightly faster to do reshape here than in the loop, and same memory usage)
        newe = newe.view(img_b, cnt_h, cnt_w, -1)  # Batch x FlatDim0/1 x OutH x OutW x (InC * FilterH * FilterW)
        newe = newe.matmul(ws) + self.bias  # Batch x FlatDim0/1 x OutH x OutW x OutC

        # utils.pp_cuda_mem('Conv: After reshaping and matmul')

        reorders = [0, 1, 4, 2, 3]
        newe_lcoefs = newe._lcoef.permute(*reorders)  # Batch x FlatDim0 x OutC x OutH x OutW
        newe_lcnsts = newe._lcnst.permute(*reorders)
        newe_ucoefs = newe._ucoef.permute(*reorders)
        newe_ucnsts = newe._ucnst.permute(*reorders)

        # utils.pp_cuda_mem('Conv: After permutation')
        return Ele(newe_lcoefs, newe_lcnsts, newe_ucoefs, newe_ucnsts, newe.dlb, newe.dub)

    def _mem_efficient_conv(self, e, img_b, flat_size, cnt_h, cnt_w, fil_h, fil_w, stride_h, stride_w,
                            full_lb_coefs, full_lb_cnsts, full_ub_coefs, full_ub_cnsts) -> Ele:
        """ Here it may consume lots of memory if expanding everything like that in vanilla_intvl, because DeepPoly
            already introduces a large FlatDim0 dimension. This may be a problem for smaller memory GPU. The forward()
            is faster than this method (4.15s vs 5.74s) but allocates/caches more memory (max 10.2GB vs max 4.4GB).
        """
        ws = self.weight.view(self.out_channels, -1).t()  # (InC * FilterH * FilterW) x OutC

        # directly compute the values on each location, to reduce memory usage
        full_newes = []
        for i in range(cnt_h):
            row_newes = []
            for j in range(cnt_w):
                h_start = i * stride_h
                h_end = h_start + fil_h
                w_start = j * stride_w
                w_end = w_start + fil_w

                tmp_lb_coefs = full_lb_coefs[..., h_start: h_end,
                               w_start: w_end]  # Batch x FlatDim0 x InC x FilterH x FilterW
                tmp_lb_cnsts = full_lb_cnsts[..., h_start: h_end, w_start: w_end]
                tmp_ub_coefs = full_ub_coefs[..., h_start: h_end, w_start: w_end]
                tmp_ub_cnsts = full_ub_cnsts[..., h_start: h_end, w_start: w_end]
                tmp_newe = Ele(tmp_lb_coefs, tmp_lb_cnsts, tmp_ub_coefs, tmp_ub_cnsts, e.dlb, e.dub)
                tmp_newe = tmp_newe.contiguous().view(img_b, -1)  # Batch x FlatDim0 x -1
                tmp_newe = tmp_newe.contiguous().matmul(ws) + self.bias  # Batch x FlatDim0/1 x OutC
                row_newes.append(tmp_newe)
            full_newes.append(row_newes)

        # utils.pp_cuda_mem('Conv: After new conv matmul')

        def stack_all(full: List[List[Tensor]]) -> Tensor:
            res = []
            for ts in full:
                res.append(torch.stack(ts, dim=-1))  # each is Batch x FlatDim0 x OutC x FilterW
            return torch.stack(res, dim=-2)  # Batch x FlatDim0 x OutC x FilterH x FilterW

        full_lcoefs = [[newe._lcoef for newe in tmp_row] for tmp_row in full_newes]
        full_lcnsts = [[newe._lcnst for newe in tmp_row] for tmp_row in full_newes]
        full_ucoefs = [[newe._ucoef for newe in tmp_row] for tmp_row in full_newes]
        full_ucnsts = [[newe._ucnst for newe in tmp_row] for tmp_row in full_newes]

        # utils.pp_cuda_mem('Conv: After new gathering')

        full_lcoefs = stack_all(full_lcoefs)
        full_lcnsts = stack_all(full_lcnsts)
        full_ucoefs = stack_all(full_ucoefs)
        full_ucnsts = stack_all(full_ucnsts)

        # utils.pp_cuda_mem('Conv: After final stacking')
        return Ele(full_lcoefs, full_lcnsts, full_ucoefs, full_ucnsts, e.dlb, e.dub)
    pass


def clamp(e: Ele, ch_val_mins: Tensor, ch_val_maxs: Tensor) -> Ele:
    """ Basically a generalized version of ReLU, clamp() for an abstract element may also be over-approximated.
        Note that now there are two bars, to maintain the validity of LB<=UB, both may be over-approximated.
    """
    assert e._lcoef.shape[1] == 1  # only for brightness robustness for now, just 1 commons eps to depend on
    assert ch_val_mins.dim() == 1 and len(ch_val_mins) in [1, 3]
    assert ch_val_maxs.dim() == 1 and len(ch_val_maxs) in [1, 3]

    lbs, ubs = e.gamma()  # Batch x Dims...

    full_lb_coefs = e._lcoef
    full_lb_cnsts = e._lcnst
    full_ub_coefs = e._ucoef
    full_ub_cnsts = e._ucnst
    coef_zeros = torch.zeros_like(e._lcoef)
    cnst_zeros = torch.zeros_like(e._lcnst)

    # now that it may provide different min/max for different channel, I have to enumerate multiple times
    channels = e.size()[1]
    for c in range(channels):
        c_filter = torch.zeros_like(lbs, dtype=torch.uint8)
        c_filter[:, c, :, :] = 1  # B x C x H x W

        val_min = ch_val_mins[c]
        val_max = ch_val_maxs[c]

        both_le_min = c_filter & (ubs <= val_min)
        both_ge_max = c_filter & (lbs >= val_max)
        both_valid = c_filter & (lbs >= val_min) & (ubs <= val_max)
        the_rest = c_filter & (~ (both_le_min | both_ge_max | both_valid))

        lb_min_ub_max = c_filter & the_rest & ((lbs <= val_min) & (ubs <= val_max))
        min_lb_max_ub = c_filter & the_rest & ((lbs >= val_min) & (ubs >= val_max))
        lb_min_max_ub = c_filter & the_rest & ((lbs <= val_min) & (ubs >= val_max))

        assert (c_filter == both_le_min | both_ge_max | both_valid | lb_min_ub_max | min_lb_max_ub | lb_min_max_ub).all()

        # old index was for concrete outputs, need  to unsqueeze to fit both coefs and cnsts
        both_le_min = both_le_min.unsqueeze(dim=1)
        both_ge_max = both_ge_max.unsqueeze(dim=1)
        both_valid = both_valid.unsqueeze(dim=1)
        lb_min_ub_max = lb_min_ub_max.unsqueeze(dim=1)
        min_lb_max_ub = min_lb_max_ub.unsqueeze(dim=1)
        lb_min_max_ub = lb_min_max_ub.unsqueeze(dim=1)

        def _case_both_le_min() -> Tuple[Tensor, Tensor, Tensor, Tensor]:
            new_coef = coef_zeros
            new_cnst = cnst_zeros + val_min
            return new_coef, new_cnst, new_coef, new_cnst

        tmp_lb_coefs, tmp_lb_cnsts, tmp_ub_coefs, tmp_ub_cnsts = _case_both_le_min()
        full_lb_coefs = torch.where(both_le_min, tmp_lb_coefs, full_lb_coefs)
        full_lb_cnsts = torch.where(both_le_min, tmp_lb_cnsts, full_lb_cnsts)
        full_ub_coefs = torch.where(both_le_min, tmp_ub_coefs, full_ub_coefs)
        full_ub_cnsts = torch.where(both_le_min, tmp_ub_cnsts, full_ub_cnsts)

        def _case_both_ge_max() -> Tuple[Tensor, Tensor, Tensor, Tensor]:
            new_coef = coef_zeros
            new_cnst = cnst_zeros + val_max
            return new_coef, new_cnst, new_coef, new_cnst

        tmp_lb_coefs, tmp_lb_cnsts, tmp_ub_coefs, tmp_ub_cnsts = _case_both_ge_max()
        full_lb_coefs = torch.where(both_ge_max, tmp_lb_coefs, full_lb_coefs)
        full_lb_cnsts = torch.where(both_ge_max, tmp_lb_cnsts, full_lb_cnsts)
        full_ub_coefs = torch.where(both_ge_max, tmp_ub_coefs, full_ub_coefs)
        full_ub_cnsts = torch.where(both_ge_max, tmp_ub_cnsts, full_ub_cnsts)

        def _case_lb_min_ub_max() -> Tuple[Tensor, Tensor, Tensor, Tensor]:
            new_lb_coef = coef_zeros
            new_lb_cnst = cnst_zeros + val_min

            denom = ubs - lbs
            backup = torch.ones_like(denom)
            denom = torch.where(denom == 0., backup, denom)

            ub_k = (ubs - val_min) / denom
            ub_b = ubs * (val_min - lbs) / denom

            ub_k = ub_k.unsqueeze(dim=1)  # then it will fit both coefs and cnsts
            ub_b = ub_b.unsqueeze(dim=1)
            assert ub_k.shape == ub_b.shape == e._ucoef.shape == e._ucnst.shape
            new_ub_coef = e._ucoef * ub_k
            new_ub_cnst = e._ucnst * ub_k + ub_b
            return new_lb_coef, new_lb_cnst, new_ub_coef, new_ub_cnst

        tmp_lb_coefs, tmp_lb_cnsts, tmp_ub_coefs, tmp_ub_cnsts = _case_lb_min_ub_max()
        full_lb_coefs = torch.where(lb_min_ub_max, tmp_lb_coefs, full_lb_coefs)
        full_lb_cnsts = torch.where(lb_min_ub_max, tmp_lb_cnsts, full_lb_cnsts)
        full_ub_coefs = torch.where(lb_min_ub_max, tmp_ub_coefs, full_ub_coefs)
        full_ub_cnsts = torch.where(lb_min_ub_max, tmp_ub_cnsts, full_ub_cnsts)

        def _case_min_lb_max_ub() -> Tuple[Tensor, Tensor, Tensor, Tensor]:
            new_ub_coef = coef_zeros
            new_ub_cnst = cnst_zeros + val_max

            denom = ubs - lbs
            backup = torch.ones_like(denom)
            denom = torch.where(denom == 0., backup, denom)

            lb_k = (val_max - lbs) / denom
            lb_b = lbs * (ubs - val_max) / denom

            lb_k = lb_k.unsqueeze(dim=1)  # then it will fit both coefs and cnsts
            lb_b = lb_b.unsqueeze(dim=1)
            assert lb_k.shape == lb_b.shape == e._lcoef.shape == e._lcnst.shape
            new_lb_coef = e._lcoef * lb_k
            new_lb_cnst = e._lcnst * lb_k + lb_b
            return new_lb_coef, new_lb_cnst, new_ub_coef, new_ub_cnst

        tmp_lb_coefs, tmp_lb_cnsts, tmp_ub_coefs, tmp_ub_cnsts = _case_min_lb_max_ub()
        full_lb_coefs = torch.where(min_lb_max_ub, tmp_lb_coefs, full_lb_coefs)
        full_lb_cnsts = torch.where(min_lb_max_ub, tmp_lb_cnsts, full_lb_cnsts)
        full_ub_coefs = torch.where(min_lb_max_ub, tmp_ub_coefs, full_ub_coefs)
        full_ub_cnsts = torch.where(min_lb_max_ub, tmp_ub_cnsts, full_ub_cnsts)

        def _case_lb_min_max_ub() -> Tuple[Tensor, Tensor, Tensor, Tensor]:
            new_lb_coef = coef_zeros
            new_lb_cnst = cnst_zeros + val_min
            new_ub_coef = coef_zeros
            new_ub_cnst = cnst_zeros + val_max
            return new_lb_coef, new_lb_cnst, new_ub_coef, new_ub_cnst

        tmp_lb_coefs, tmp_lb_cnsts, tmp_ub_coefs, tmp_ub_cnsts = _case_lb_min_max_ub()
        full_lb_coefs = torch.where(lb_min_max_ub, tmp_lb_coefs, full_lb_coefs)
        full_lb_cnsts = torch.where(lb_min_max_ub, tmp_lb_cnsts, full_lb_cnsts)
        full_ub_coefs = torch.where(lb_min_max_ub, tmp_ub_coefs, full_ub_coefs)
        full_ub_cnsts = torch.where(lb_min_max_ub, tmp_ub_cnsts, full_ub_cnsts)

    return Ele(full_lb_coefs, full_lb_cnsts, full_ub_coefs, full_ub_cnsts, e.dlb, e.dub)


class ReLU(nn.ReLU):
    def __str__(self):
        return f'{Dom.name}.' + super().__str__()

    def export(self) -> nn.ReLU:
        return nn.ReLU()

    def forward(self, *ts: Union[Tensor, Ele]) -> Union[Tensor, Ele, Tuple[Tensor, ...]]:
        """ According to paper, it approximates E by either of the two cases, whichever has smaller areas.
            Mathematically, it can be proved that the (linear) approximation is optimal in terms of approximated areas.

            (1) Now that the approximation must contain the two vertices (li, 0) and (ui, ui). If only one single linear
                constraint is allowed, it has to be the one determined by (li, 0) and (ui, ui). So upper bound is fixed.
            (2) Now that the approximation must contain the third vertex (0, 0). The lower bound linear constraint has
                to be some y = kx function. To minimize the overall approximation area, it can be shown that the minimum
                is achieved either when k = 0 or k = 1. Which is the (b) and (c) in Fig. 4 in paper.
            (3) More specifically, when |li| > |ui|, choose k = 0 for minimum area; when |li| < |ui|, choose k = 1.
        """
        input_is_ele = True
        if len(ts) == 1:
            if isinstance(ts[0], Tensor):
                return super().forward(ts[0])  # plain tensor, no abstraction
            elif isinstance(ts[0], Ele):
                e = ts[0]  # abstract element
            else:
                raise ValueError(f'Not supported argument type {type(ts[0])}.')
        else:
            input_is_ele = False
            e = Ele(*ts)  # reconstruct abstract element

        size = e._lcoef.size()
        flat_size = size[1]  # FlatDim0

        # was flattening the dimensions, actually no need to do that
        lcoef = e._lcoef  # Batch x FlatDim0 x Dims...
        lcnst = e._lcnst
        ucoef = e._ucoef
        ucnst = e._ucnst

        # utils.pp_cuda_mem('ReLU: Before gamma()')

        lb, ub = e.gamma()  # Batch x Dims...
        coef_zeros = torch.zeros_like(lcoef)
        cnst_zeros = torch.zeros_like(lcnst)
        lbub_zeros = torch.zeros_like(lb)

        # utils.pp_cuda_mem('ReLU: After gamma()')

        ''' 4 distinct cases
            (a) All cleared to zero
            (b) All preserved
            (c) Need over-approximation, upper constraint is fixed by two points (lb, 0) and (ub, ub)
                (c1) Lower constraint over-approximated by y = 0 (k=0)
                (c2) Lower constraint over-approximated by y = x (k=1)

            There was a heuristic to choose LB'=0 or LB'=LB following that in DeepPoly paper. Basically, when |LB| is
            larger, setting k=0 lower bound induces smaller area, otherwise, simply preserve k=1 without change.

            However, this heuristic incurs some refinement soundness violation, namely, a refined smaller region may
            have larger safety loss. See deeppoly_soundness_issue.py for experiment details.

            Now, it just sets LB'=0 when necessary.
        '''
        all_clear = ub <= 0
        not_all_clear = ~ all_clear
        all_pres = (lb >= 0) & not_all_clear
        approxed = not_all_clear & (~ all_pres)

        def full_bits(bits: Tensor, is_coef: bool) -> Tensor:
            sizes = list(bits.size())
            bits = bits.unsqueeze(dim=1)  # right after Batch x ...
            if is_coef:
                sizes.insert(1, flat_size)  # Batch x FlatDim0 x ...
            else:
                sizes.insert(1, 1)  # Batch x 1 x ...
            return bits.expand(*sizes)

        # kzero = (lb.abs() >= ub.abs()) & approxed
        kzero = approxed  # setting all to be zero
        kone = approxed & (~ kzero)

        # Experiments show that torch.where() is 2~3 times faster than zeros[bits] = ucoef[bits] then sum.
        ''' Cases for new Upper Constraint:
            (1) Cleared -- all_clear
            (2) Preserved -- all_pres
            (3) Over-approximated -- approxed
        '''
        _ucoef2 = torch.where(full_bits(all_pres, True), ucoef, coef_zeros)
        _ucnst2 = torch.where(full_bits(all_pres, False), ucnst, cnst_zeros)

        # New upper constraint is determined due to (lb, 0) and (ub, ub): x' <= ub(x-lb)/(ub-lb)
        _ucoef3 = torch.where(full_bits(approxed, True), ucoef, coef_zeros)  # Batch x FlatDim0 x Dims
        _ucnst3 = torch.where(full_bits(approxed, False), ucnst, cnst_zeros)  # Batch x 1 x Dims
        _lb3 = torch.where(approxed, lb, lbub_zeros).unsqueeze(dim=1)  # Batch x 1 x Dims
        _ub3 = torch.where(approxed, ub, lbub_zeros).unsqueeze(dim=1)  # Batch x 1 x Dims

        # original deeppoly
        _ucnst3 = _ucnst3 - _lb3  # Batch x Dim x 1
        _denominator = torch.where(approxed.unsqueeze(dim=1), _ub3 - _lb3, torch.ones_like(lb).unsqueeze(dim=1))
        _ucoef3 = _ucoef3 * _ub3 / _denominator  # should then broadcast
        _ucnst3 = _ucnst3 * _ub3 / _denominator

        # 06/21: Quick experiment: now fix upper constraint to be y = x - L where L = min(gamma(g), 0)
        # _ucnst3 = _ucnst3 - _lb3  # Batch x Dim x 1

        new_ucoef = _ucoef2 + _ucoef3  # _ucoef1 is 0
        new_ucnst = _ucnst2 + _ucnst3

        ''' Cases for new Lower Constraint:
            (1) Cleared -- all_clear, kzero
            (2) Preserved -- all_pres, kone
        '''
        new_lcoef = torch.where(full_bits(all_pres | kone, True), lcoef, coef_zeros)
        new_lcnst = torch.where(full_bits(all_pres | kone, False), lcnst, cnst_zeros)

        # utils.pp_cuda_mem('ReLU: After everything')

        new_e = Ele(new_lcoef, new_lcnst, new_ucoef, new_ucnst, e.dlb, e.dub)
        return new_e if input_is_ele else tuple(new_e)
    pass


class Tanh(nn.Tanh):
    def __str__(self):
        return f'{Dom.name}.' + super().__str__()

    def export(self) -> nn.Tanh:
        return nn.Tanh()

    def forward(self, *ts: Union[Tensor, Ele]) -> Union[Tensor, Ele, Tuple[Tensor, ...]]:
        """ For both LB' and UB', it chooses the smaller slope between LB-UB and LB'/UB'. Specifically,
            when L > 0, LB' chooses LB-UB, otherwise LB';
            when U < 0, UB' chooses LB-UB, otherwise UB'.
        """
        input_is_ele = True
        if len(ts) == 1:
            if isinstance(ts[0], Tensor):
                return super().forward(ts[0])  # plain tensor, no abstraction
            elif isinstance(ts[0], Ele):
                e = ts[0]  # abstract element
            else:
                raise ValueError(f'Not supported argument type {type(ts[0])}.')
        else:
            input_is_ele = False
            e = Ele(*ts)  # reconstruct abstract element

        flat_size = e._lcoef.size()[1]  # FlatDim0

        # was flattening the dimensions, actually no need to do that
        lcoef = e._lcoef  # Batch x FlatDim0 x Dims...
        lcnst = e._lcnst
        ucoef = e._ucoef
        ucnst = e._ucnst

        # utils.pp_cuda_mem('Tanh: Before gamma()')

        lb, ub = e.gamma()  # Batch x Dims...
        coef_zeros = torch.zeros_like(lcoef)
        cnst_zeros = torch.zeros_like(lcnst)
        lbub_zeros = torch.zeros_like(lb)
        # utils.pp_cuda_mem('Tanh: After gamma()')

        tanh_lb, tanh_ub = torch.tanh(lb), torch.tanh(ub)
        lbub_same = tanh_lb >= tanh_ub  # perhaps the extra > would cover a bit of numerical error as well?

        def full_bits(bits: Tensor, is_coef: bool) -> Tensor:
            sizes = list(bits.size())
            bits = bits.unsqueeze(dim=1)  # right after Batch x ...
            if is_coef:
                sizes.insert(1, flat_size)  # Batch x FlatDim0 x ...
            else:
                sizes.insert(1, 1)  # Batch x 1 x ...
            return bits.expand(*sizes)

        # when L = U, just reset them to a constant value
        case_degen_lcoef = coef_zeros
        case_degen_lcnst = tanh_lb.unsqueeze(dim=1)  # Batch x 1 x Dims
        case_degen_ucoef = case_degen_lcoef
        case_degen_ucnst = case_degen_lcnst
        full_lcoef = torch.where(full_bits(lbub_same, True), case_degen_lcoef, coef_zeros)
        full_lcnst = torch.where(full_bits(lbub_same, False), case_degen_lcnst, cnst_zeros)
        full_ucoef = torch.where(full_bits(lbub_same, True), case_degen_ucoef, coef_zeros)
        full_ucnst = torch.where(full_bits(lbub_same, False), case_degen_ucnst, cnst_zeros)

        denom = torch.where(lbub_same, torch.ones_like(lb), ub - lb)
        # Batch x Dims...
        k_lbub = (tanh_ub - tanh_lb) / denom  # the slope for LB-UB
        k_lb = 1 - tanh_lb ** 2  # the slope for tangent on LB, tanh' = 1 - tanh^2
        k_ub = 1 - tanh_ub ** 2  # the slope for tangent on UB

        # Batch x Dims...
        b_lower_lbub_smaller = tanh_lb - k_lbub * lb  # the bias for LB', using k_lbub
        b_lower_lb_smaller = tanh_lb - k_lb * lb  # the bias for LB', using k_lb
        b_upper_lbub_smaller = tanh_ub - k_lbub * ub  # the bias for UB', using k_lbub
        b_upper_ub_smaller = tanh_ub - k_ub * ub  # the bias for UB', using k_ub

        # for LB'
        lbub_smaller = (k_lbub < k_lb) & (~lbub_same)
        full_lcoef = torch.where(full_bits(lbub_smaller, True), lcoef * k_lbub.unsqueeze(dim=1), full_lcoef)
        full_lcnst = torch.where(full_bits(lbub_smaller, False),
                                 lcnst * k_lbub.unsqueeze(dim=1) + b_lower_lbub_smaller.unsqueeze(dim=1), full_lcnst)

        lb_smaller = (~lbub_smaller) & (~lbub_same)
        full_lcoef = torch.where(full_bits(lb_smaller, True), lcoef * k_lb.unsqueeze(dim=1), full_lcoef)
        full_lcnst = torch.where(full_bits(lb_smaller, False),
                                 lcnst * k_lb.unsqueeze(dim=1) + b_lower_lb_smaller.unsqueeze(dim=1), full_lcnst)

        # for UB'
        lbub_smaller = (k_lbub < k_ub) & (~lbub_same)
        full_ucoef = torch.where(full_bits(lbub_smaller, True), ucoef * k_lbub.unsqueeze(dim=1), full_ucoef)
        full_ucnst = torch.where(full_bits(lbub_smaller, False),
                                 ucnst * k_lbub.unsqueeze(dim=1) + b_upper_lbub_smaller.unsqueeze(dim=1), full_ucnst)

        ub_smaller = (~lbub_smaller) & (~lbub_same)
        full_ucoef = torch.where(full_bits(ub_smaller, True), ucoef * k_ub.unsqueeze(dim=1), full_ucoef)
        full_ucnst = torch.where(full_bits(ub_smaller, False),
                                 ucnst * k_ub.unsqueeze(dim=1) + b_upper_ub_smaller.unsqueeze(dim=1), full_ucnst)

        # utils.pp_cuda_mem('Tanh: After everything')
        new_e = Ele(full_lcoef, full_lcnst, full_ucoef, full_ucnst, e.dlb, e.dub)
        return new_e if input_is_ele else tuple(new_e)
    pass


class MaxPool1d(nn.MaxPool1d):
    """ I have to implement the forward computation by myself, because F.max_pool1d() requires input to be Tensors.
        Note that the MaxPool1d takes input of shape Batch x InChannel(Planes) x Data.
    """
    def __str__(self):
        return f'{Dom.name}.' + super().__str__()

    def forward(self, *ts: Union[Tensor, Ele]) -> Union[Tensor, Ele, Tuple[Tensor, ...]]:
        input_is_ele = True
        if len(ts) == 1:
            if isinstance(ts[0], Tensor):
                return super().forward(ts[0])  # plain tensor, no abstraction
            elif isinstance(ts[0], Ele):
                e = ts[0]  # abstract element
            else:
                raise ValueError(f'Not supported argument type {type(ts[0])}.')
        else:
            input_is_ele = False
            e = Ele(*ts)  # reconstruct abstract element

        ''' See 'https://github.com/vdumoulin/conv_arithmetic' for animated illustrations.
            It's not hard to support them, but we just don't need that right now.
        '''
        if self.dilation not in [1, (1, 1)]:
            raise NotImplementedError(f'Unsupported dilation {self.dilation}')
        if self.padding not in [0, (0, 0)]:
            raise NotImplementedError(f'Unsupported padding {self.padding}')

        assert e.dim() == 3, 'Taking inputs of the shape Batch x InChannel(Planes) x Data'
        vec_b, flat_size, vec_c, vec_x = e._lcoef.size()  # Batch x FlatDim0 x InChannel(C) x InData(D)

        # formula: (W - F + 2P) / S + 1
        cnt = (vec_x - self.kernel_size) / self.stride + 1
        assert int(cnt) == cnt, "data and filter dimensions don't fit?"
        cnt = int(cnt)

        full_lb_coefs = e._lcoef
        full_lb_cnsts = e._lcnst
        full_ub_coefs = e._ucoef
        full_ub_cnsts = e._ucnst

        filtered_lb_coefs, filtered_lb_cnsts = [], []
        filtered_ub_coefs, filtered_ub_cnsts = [], []
        for i in range(cnt):
            start = i * self.stride
            end = start + self.kernel_size

            filtered_lb_coefs.append(full_lb_coefs[..., start: end])  # Batch x FlatDim0 x InC x FilterD
            filtered_lb_cnsts.append(full_lb_cnsts[..., start: end])
            filtered_ub_coefs.append(full_ub_coefs[..., start: end])
            filtered_ub_cnsts.append(full_ub_cnsts[..., start: end])

        filtered_lb_coefs = torch.stack(filtered_lb_coefs, dim=3)  # dim=3: right after Batch x FlatDim0 x InC x ...
        filtered_lb_cnsts = torch.stack(filtered_lb_cnsts, dim=3)  # Now Batch x FlatDim0 x InC x Out x FilterD
        filtered_ub_coefs = torch.stack(filtered_ub_coefs, dim=3)
        filtered_ub_cnsts = torch.stack(filtered_ub_cnsts, dim=3)

        filtered_e = Ele(filtered_lb_coefs, filtered_lb_cnsts, filtered_ub_coefs, filtered_ub_cnsts, e.dlb, e.dub)
        # filtered_e = filtered_e.view(vec_b, vec_c, cnt, -1)  # Batch x FlatDim0/1 x InC x Out x FilterD
        # above was for MaxPool2d, no need for 1d here, since it can gamma() directly
        filtered_lbs, filtered_ubs = filtered_e.gamma()  # Batch x InC x Out x FilterD

        lb_idxs = filtered_lbs.argmax(dim=-1, keepdim=True)  # Batch x InC x Out x 1
        ub_idxs = filtered_ubs.argmax(dim=-1, keepdim=True)
        lb_idxs = lb_idxs.unsqueeze(dim=1)  # Batch x 1 x InC x Out x 1
        ub_idxs = ub_idxs.unsqueeze(dim=1)

        # Batch x FlatDim0 x InC x Out
        newl_coefs = torch.gather(filtered_e._lcoef, -1, lb_idxs.expand(-1, flat_size, -1, -1, -1)).squeeze(dim=-1)
        newl_cnsts = torch.gather(filtered_e._lcnst, -1, lb_idxs).squeeze(dim=-1)  # Batch x 1 x InC x Out
        newu_coefs = torch.gather(filtered_e._ucoef, -1, ub_idxs.expand(-1, flat_size, -1, -1, -1)).squeeze(dim=-1)
        newu_cnsts = torch.gather(filtered_e._ucnst, -1, ub_idxs).squeeze(dim=-1)

        out = Ele(newl_coefs, newl_cnsts, newu_coefs, newu_cnsts, e.dlb, e.dub)
        return out if input_is_ele else tuple(out)
    pass


class MaxPool2d(nn.MaxPool2d):
    """ MaxPool2d layer with the ability to take in approximations rather than concrete inputs. """
    def __str__(self):
        return f'{Dom.name}.' + super().__str__()

    def forward(self, *ts: Union[Tensor, Ele]) -> Union[Tensor, Ele, Tuple[Tensor, ...]]:
        """ I have to implement the forward computation by myself, because F.max_pool2d() requires input to be Tensors. """
        input_is_ele = True
        if len(ts) == 1:
            if isinstance(ts[0], Tensor):
                return super().forward(ts[0])  # plain tensor, no abstraction
            elif isinstance(ts[0], Ele):
                e = ts[0]  # abstract element
            else:
                raise ValueError(f'Not supported argument type {type(ts[0])}.')
        else:
            input_is_ele = False
            e = Ele(*ts)  # reconstruct abstract element

        ''' See 'https://github.com/vdumoulin/conv_arithmetic' for animated illustrations.
            It's not hard to support them, but we just don't need that right now.
        '''
        if self.dilation not in [1, (1, 1)]:
            raise NotImplementedError(f'Unsupported dilation {self.dilation}')
        if self.padding not in [0, (0, 0)]:
            raise NotImplementedError(f'Unsupported padding {self.padding}')

        assert e.dim() == 4
        img_b, flat_size, img_c, img_h, img_w = e._lcoef.size()  # Batch x FlatDim0 x C x H x W

        def _load(v: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
            if isinstance(v, int):
                return v, v
            else:
                return v[0], v[1]

        fil_h, fil_w = _load(self.kernel_size)
        stride_h, stride_w = _load(self.stride)

        # formula: (W - F + 2P) / S + 1
        cnt_h = (img_h - fil_h) / stride_h + 1
        cnt_w = (img_w - fil_w) / stride_w + 1
        assert int(cnt_h) == cnt_h and int(cnt_w) == cnt_w, "img and filter dimensions don't fit?"
        cnt_h = int(cnt_h)
        cnt_w = int(cnt_w)

        full_lb_coefs = e._lcoef
        full_lb_cnsts = e._lcnst
        full_ub_coefs = e._ucoef
        full_ub_cnsts = e._ucnst

        # utils.pp_cuda_mem('Pool: Before enumeration')

        ''' The following code is faster that _mem_efficient_pool() (2.22s vs 3.06s)
            but allocates/caches more memory (max 196.3MB vs max 50.1MB).
        '''
        out = self._pool(e, img_b, flat_size, img_c, cnt_h, cnt_w, fil_h, fil_w, stride_h, stride_w,
                         full_lb_coefs, full_lb_cnsts, full_ub_coefs, full_ub_cnsts)
        return out if input_is_ele else tuple(out)

    def _pool(self, e, img_b, flat_size, img_c, cnt_h, cnt_w, fil_h, fil_w, stride_h, stride_w,
                   full_lb_coefs, full_lb_cnsts, full_ub_coefs, full_ub_cnsts) -> Ele:
        """ Collect all filtered sub-images in a large batch. """
        filtered_lb_coefs, filtered_lb_cnsts = [], []
        filtered_ub_coefs, filtered_ub_cnsts = [], []
        for i in range(cnt_h):
            row_lb_coefs, row_lb_cnsts = [], []
            row_ub_coefs, row_ub_cnsts = [], []
            for j in range(cnt_w):
                h_start = i * stride_h
                h_end = h_start + fil_h
                w_start = j * stride_w
                w_end = w_start + fil_w

                sub_lb_coefs = full_lb_coefs[..., h_start : h_end, w_start : w_end]  # Batch x FlatDim0 x InC x FilterH x FilterW
                sub_lb_cnsts = full_lb_cnsts[..., h_start : h_end, w_start : w_end]
                sub_ub_coefs = full_ub_coefs[..., h_start : h_end, w_start : w_end]
                sub_ub_cnsts = full_ub_cnsts[..., h_start : h_end, w_start : w_end]
                row_lb_coefs.append(sub_lb_coefs)
                row_lb_cnsts.append(sub_lb_cnsts)
                row_ub_coefs.append(sub_ub_coefs)
                row_ub_cnsts.append(sub_ub_cnsts)

            row_lb_coefs = torch.stack(row_lb_coefs, dim=3)  # dim=3: right after Batch x FlatDim0 x InC x ...
            row_lb_cnsts = torch.stack(row_lb_cnsts, dim=3)  # Now Batch x FlatDim0 x InC x OutW x FilterH x FilterW
            row_ub_coefs = torch.stack(row_ub_coefs, dim=3)
            row_ub_cnsts = torch.stack(row_ub_cnsts, dim=3)
            filtered_lb_coefs.append(row_lb_coefs)
            filtered_lb_cnsts.append(row_lb_cnsts)
            filtered_ub_coefs.append(row_ub_coefs)
            filtered_ub_cnsts.append(row_ub_cnsts)

        # utils.pp_cuda_mem('Pool: After fast enumeration')

        filtered_lb_coefs = torch.stack(filtered_lb_coefs, dim=3)  # dim=3: right after Batch x FlatDim0 x InC x ... again
        filtered_lb_cnsts = torch.stack(filtered_lb_cnsts, dim=3)  # Now Batch x FlatDim0 x InC x OutH x OutW x FilterH x FilterW
        filtered_ub_coefs = torch.stack(filtered_ub_coefs, dim=3)
        filtered_ub_cnsts = torch.stack(filtered_ub_cnsts, dim=3)

        # utils.pp_cuda_mem('Pool: After fast stacking')

        filtered_e = Ele(filtered_lb_coefs, filtered_lb_cnsts, filtered_ub_coefs, filtered_ub_cnsts, e.dlb, e.dub)
        filtered_e = filtered_e.view(img_b, img_c, cnt_h, cnt_w, -1)  # Batch x FlatDim0/1 x InC x OutH x OutW x (FilterH*FilterW)

        # utils.pp_cuda_mem('Pool: After fast reshaping')

        filtered_lbs, filtered_ubs = filtered_e.gamma()  # Batch x InC x OutH x OutW x (FilterH*FilterW)

        # utils.pp_cuda_mem('Pool: After fast gamma()')

        lb_idxs = filtered_lbs.argmax(dim=-1, keepdim=True)  # Batch x InC x OutH x OutW x 1
        ub_idxs = filtered_ubs.argmax(dim=-1, keepdim=True)
        lb_idxs = lb_idxs.unsqueeze(dim=1)  # Batch x 1 x InC x OutH x OutW x 1
        ub_idxs = ub_idxs.unsqueeze(dim=1)

        # utils.pp_cuda_mem('Pool: After fast selecting indices')

        # Batch x FlatDim0 x InC x OutH x OutW
        newl_coefs = torch.gather(filtered_e._lcoef, -1, lb_idxs.expand(-1, flat_size, -1, -1, -1, -1)).squeeze(dim=-1)
        newl_cnsts = torch.gather(filtered_e._lcnst, -1, lb_idxs).squeeze(dim=-1)  # Batch x 1 x InC x OutH x OutW
        newu_coefs = torch.gather(filtered_e._ucoef, -1, ub_idxs.expand(-1, flat_size, -1, -1, -1, -1)).squeeze(dim=-1)
        newu_cnsts = torch.gather(filtered_e._ucnst, -1, ub_idxs).squeeze(dim=-1)

        # utils.pp_cuda_mem('Pool: After fast index-based reconstruction')

        return Ele(newl_coefs, newl_cnsts, newu_coefs, newu_cnsts, e.dlb, e.dub)

    def _mem_efficient_pool(self, e, img_b, flat_size, img_c, cnt_h, cnt_w, fil_h, fil_w, stride_h, stride_w,
                            full_lb_coefs, full_lb_cnsts, full_ub_coefs, full_ub_cnsts) -> Ele:
        """ Directly compute the values on each location, to reduce memory usage. """
        full_newes = []
        for i in range(cnt_h):
            row_newes = []
            for j in range(cnt_w):
                h_start = i * stride_h
                h_end = h_start + fil_h
                w_start = j * stride_w
                w_end = w_start + fil_w

                tmp_lb_coefs = full_lb_coefs[..., h_start: h_end, w_start: w_end]  # Batch x FlatDim0 x InC x FilterH x FilterW
                tmp_lb_cnsts = full_lb_cnsts[..., h_start: h_end, w_start: w_end]
                tmp_ub_coefs = full_ub_coefs[..., h_start: h_end, w_start: w_end]
                tmp_ub_cnsts = full_ub_cnsts[..., h_start: h_end, w_start: w_end]

                tmp_newe = Ele(tmp_lb_coefs, tmp_lb_cnsts, tmp_ub_coefs, tmp_ub_cnsts, e.dlb, e.dub)
                tmp_newe = tmp_newe.contiguous().view(img_b, img_c, -1)  # Batch x FlatDim0 x InC x -1
                tmp_lbs, tmp_ubs = tmp_newe.gamma()

                lb_idxs = tmp_lbs.argmax(dim=-1, keepdim=True)  # Batch x InC x 1
                ub_idxs = tmp_ubs.argmax(dim=-1, keepdim=True)
                lb_idxs = lb_idxs.unsqueeze(dim=1)  # Batch x 1 x InC x 1
                ub_idxs = ub_idxs.unsqueeze(dim=1)

                # Batch x FlatDim0 x InC
                newl_coefs = torch.gather(tmp_newe._lcoef, -1, lb_idxs.expand(-1, flat_size, -1, -1)).squeeze(dim=-1)
                newl_cnsts = torch.gather(tmp_newe._lcnst, -1, lb_idxs).squeeze(dim=-1)  # Batch x 1 x InC
                newu_coefs = torch.gather(tmp_newe._ucoef, -1, ub_idxs.expand(-1, flat_size, -1, -1)).squeeze(dim=-1)
                newu_cnsts = torch.gather(tmp_newe._ucnst, -1, ub_idxs).squeeze(dim=-1)

                tmp_newe = Ele(newl_coefs, newl_cnsts, newu_coefs, newu_cnsts, e.dlb, e.dub)
                row_newes.append(tmp_newe)

            full_newes.append(row_newes)

        # utils.pp_cuda_mem('Pool: After mem friendly pooling')

        def stack_all(full: List[List[Tensor]]) -> Tensor:
            res = []
            for ts in full:
                res.append(torch.stack(ts, dim=-1))  # each is Batch x FlatDim0 x InC x FilterW
            return torch.stack(res, dim=-2)  # Batch x FlatDim0 x InC x FilterH x FilterW

        full_lcoefs = [[newe._lcoef for newe in tmp_row] for tmp_row in full_newes]
        full_lcnsts = [[newe._lcnst for newe in tmp_row] for tmp_row in full_newes]
        full_ucoefs = [[newe._ucoef for newe in tmp_row] for tmp_row in full_newes]
        full_ucnsts = [[newe._ucnst for newe in tmp_row] for tmp_row in full_newes]

        # utils.pp_cuda_mem('Pool: After mem friendly final gathering')

        full_lcoefs = stack_all(full_lcoefs)
        full_lcnsts = stack_all(full_lcnsts)
        full_ucoefs = stack_all(full_ucoefs)
        full_ucnsts = stack_all(full_ucnsts)

        # utils.pp_cuda_mem('Pool: After mem friendly final stacking')
        return Ele(full_lcoefs, full_lcnsts, full_ucoefs, full_ucnsts, e.dlb, e.dub)
    pass
