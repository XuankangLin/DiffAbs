""" Implement the Vanilla Interval domain based on PyTorch.
    Vanilla Interval: Simply propagates using interval arithmetic without any optimization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union, Iterator, Callable

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
    def __init__(self, lb: Tensor, ub: Tensor):
        """ In Vanilla Interval domain, only the Lower Bounds and Upper Bounds are maintained. """
        assert valid_lb_ub(lb, ub)

        self._lb = lb
        self._ub = ub
        return

    @classmethod
    def by_intvl(cls, lb: Tensor, ub: Tensor) -> Ele:
        return Ele(lb, ub)

    def __iter__(self) -> Iterator[Tensor]:
        return iter((self._lb, self._ub))

    def __getitem__(self, key):
        return Ele(self._lb[key], self._ub[key])

    def size(self):
        return self._lb.size()

    def dim(self):
        return self._lb.dim()

    def lb(self) -> Tensor:
        return self._lb

    def ub(self) -> Tensor:
        return self._ub

    def view(self, *shape) -> Ele:
        return Ele(self._lb.view(*shape), self._ub.view(*shape))

    def contiguous(self) -> Ele:
        return Ele(self._lb.contiguous(), self._ub.contiguous())

    def transpose(self, dim0, dim1) -> Ele:
        return Ele(self._lb.transpose(dim0, dim1), self._ub.transpose(dim0, dim1))

    def matmul(self, weights: Tensor) -> Ele:
        """ A much faster trick:
                L' = max(0, w) * L + min(0, w) * U
                U' = max(0, w) * U + min(0, w) * L
        """
        pos_ws, neg_ws = divide_pos_neg(weights)

        newl_pos = self._lb.matmul(pos_ws)
        newl_neg = self._ub.matmul(neg_ws)
        newl = newl_pos + newl_neg

        newu_pos = self._ub.matmul(pos_ws)
        newu_neg = self._lb.matmul(neg_ws)
        newu = newu_pos + newu_neg
        return Ele(newl, newu)

    def __add__(self, other) -> Ele:
        if isinstance(other, Ele):
            return Ele(self._lb + other._lb, self._ub + other._ub)
        else:
            return Ele(self._lb + other, self._ub + other)

    def __mul__(self, flt) -> Ele:
        if isinstance(flt, Tensor) and flt.dim() == 1 and flt.shape[0] == self.size()[-1]:
            # each output vector dimension has its own factor
            pos_ws, neg_ws = divide_pos_neg(flt)

            newl_pos = self._lb * (pos_ws)
            newl_neg = self._ub * (neg_ws)
            newl = newl_pos + newl_neg

            newu_pos = self._ub * (pos_ws)
            newu_neg = self._lb * (neg_ws)
            newu = newu_pos + newu_neg
            return Ele(newl, newu)
        elif not (isinstance(flt, float) or isinstance(flt, int)):
            raise ValueError('Unsupported multiplication with', str(flt), type(flt))

        flt = float(flt)
        if flt >= 0:
            return Ele(self._lb * flt, self._ub * flt)
        else:
            return Ele(self._ub * flt, self._lb * flt)

    def __rmul__(self, flt) -> Ele:
        return self.__mul__(flt)
    pass


class Dist(AbsDist):
    """ Vanilla interval domain is non-relational, thus the distances are purely based on LB/UB tensors. """
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
        others = self._idxs_not(e, *idxs)
        others = e.lb()[..., others]

        res = []
        for i in idxs:
            target = e.ub()[..., [i]]
            diff = target - others  # will broadcast
            diff = F.relu(diff + self.eps)
            mins, _ = torch.min(diff, dim=-1)
            res.append(mins)
        return sum(res)

    def cols_is_max(self, e: Ele, *idxs: int) -> Tensor:
        """ Intuitively, some-is-max => exists target . target > all_others is always true.
            Therefore, other_col.UB() - target_col.LB() should < 0, if not, that is the distance.
            All of the others should be accounted (i.e., max).
        """
        others = self._idxs_not(e, *idxs)
        others = e.ub()[..., others]

        res = []
        for i in idxs:
            target = e.lb()[..., [i]]
            diffs = others - target  # will broadcast
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
        others = self._idxs_not(e, *idxs)
        others = e.ub()[..., others]

        res = []
        for i in idxs:
            target = e.lb()[..., [i]]
            diffs = others - target  # will broadcast
            diffs = F.relu(diffs + self.eps)
            mins, _ = torch.min(diffs, dim=-1)
            res.append(mins)
        return sum(res)

    def cols_is_min(self, e: Ele, *idxs: int) -> Tensor:
        """ Intuitively, some-is-min => exists target . target < all_others is always true.
            Therefore, target_col.UB() - other_col.LB() should < 0, if not, that is the distance.
            All of the others should be accounted (i.e., max).
        """
        others = self._idxs_not(e, *idxs)
        others = e.lb()[..., others]

        res = []
        for i in idxs:
            target = e.ub()[..., [i]]
            diffs = target - others  # will broadcast
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
        full_lb = e.lb()
        full_ub = e.ub()
        res = []
        for i in range(len(labels)):
            cat = labels[i]
            piece_outs_lb = full_lb[[i]]
            piece_outs_ub = full_ub[[i]]

            # default lb-ub or ub-lb doesn't know that target domain has distance 0, so specify that explicitly
            lefts = piece_outs_ub[..., :cat]
            rights = piece_outs_ub[..., cat + 1:]
            target = piece_outs_lb[..., [cat]]

            full = torch.cat((lefts, target, rights), dim=-1)
            diffs = full - target  # will broadcast
            # no need to ReLU here, negative values are also useful
            res.append(diffs)

        res = torch.cat(res, dim=0)
        return res

    def labels_not_predicted(self, e: Ele, labels: Tensor) -> Tensor:
        """ Intuitively, this is specifying a label_not_max for every input abstraction.
        :param label: same number of batches as self
        """
        full_lb = e.lb()
        full_ub = e.ub()
        res = []
        for i in range(len(labels)):
            cat = labels[i]
            piece_outs_lb = full_lb[[i]]
            piece_outs_ub = full_ub[[i]]

            # default lb-ub or ub-lb doesn't know that target domain has distance 0, so specify that explicitly
            lefts = piece_outs_lb[..., :cat]
            rights = piece_outs_lb[..., cat+1:]
            target = piece_outs_ub[..., [cat]]

            full = torch.cat((lefts, target, rights), dim=-1)
            diffs = target - full  # will broadcast
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
    """ Linear layer with the ability to take approximations rather than concrete inputs. """
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
        """ Re-implement the forward computation by myself, because F.linear() may apply optimization using
            torch.addmm() which requires inputs to be tensor.
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
    """ Convolutional layer with the ability to take in approximations rather than concrete inputs. """
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
        img_b, img_c, img_h, img_w = e.size()  # Batch x C x H x W

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
            if pad_w > 0:
                zs = torch.zeros(img_b, img_c, img_h, pad_w, device=orig.device)
                orig = torch.cat((zs, orig, zs), dim=-1)
            if pad_h > 0:
                zs = torch.zeros(img_b, img_c, pad_h, img_w + 2 * pad_w, device=orig.device)  # width has increased
                orig = torch.cat((zs, orig, zs), dim=-2)
            return orig

        full_lb = _pad(e._lb)
        full_ub = _pad(e._ub)

        # collect all filtered sub-images in a large batch
        filtered_lb = []
        filtered_ub = []
        for i in range(cnt_h):
            row_lb = []
            row_ub = []
            for j in range(cnt_w):
                h_start = i * stride_h
                h_end = h_start + fil_h
                w_start = j * stride_w
                w_end = w_start + fil_w

                sub_lb = full_lb[:, :, h_start : h_end, w_start : w_end]  # Batch x InC x FilterH x FilterW
                sub_ub = full_ub[:, :, h_start : h_end, w_start : w_end]
                row_lb.append(sub_lb)
                row_ub.append(sub_ub)

            row_lb = torch.stack(row_lb, dim=1)  # dim=1: right after Batch x ...
            row_ub = torch.stack(row_ub, dim=1)  # Now Batch x OutW x InC x FilterH x FilterW
            filtered_lb.append(row_lb)
            filtered_ub.append(row_ub)

        filtered_lb = torch.stack(filtered_lb, dim=1)  # dim=1: right after Batch x ... again
        filtered_ub = torch.stack(filtered_ub, dim=1)  # Now Batch x OutH x OutW x InC x FilterH x FilterW

        # reshape everything to directly apply matmul
        filtered_lb = filtered_lb.view(img_b, cnt_h, cnt_w, -1)  # Batch x OutH x OutW x (InC * FilterH * Filter)
        filtered_ub = filtered_ub.view(img_b, cnt_h, cnt_w, -1)
        ws = self.weight.view(self.out_channels, -1).t()  # (InC * FilterH * FilterW) x OutC
        newe = Ele(filtered_lb, filtered_ub).matmul(ws) + self.bias  # Batch x OutH x OutW x OutC

        newl = newe._lb.permute(0, 3, 1, 2)  # Batch x OutC x OutH x OutW
        newu = newe._ub.permute(0, 3, 1, 2)
        out = Ele(newl, newu)
        return out if input_is_ele else tuple(out)
    pass


def _distribute_to_super(super_fn: Callable, *ts: Union[Tensor, Ele]) -> Union[Tensor, Ele, Tuple[Tensor, ...]]:
    """ Common pattern shared among different customized modules, applying original methods to the bounds. """
    input_is_ele = True
    if len(ts) == 1:
        if isinstance(ts[0], Tensor):
            return super_fn(ts[0])  # plain tensor, no abstraction
        elif isinstance(ts[0], Ele):
            e = ts[0]  # abstract element
        else:
            raise ValueError(f'Not supported argument type {type(ts[0])}.')
    else:
        input_is_ele = False
        e = Ele(*ts)  # reconstruct abstract element

    out_tuple = (super_fn(t) for t in iter(e))  # simply apply to both lower and upper bounds
    return Ele(*out_tuple) if input_is_ele else out_tuple


class ReLU(nn.ReLU):
    def __str__(self):
        return f'{Dom.name}.' + super().__str__()

    def export(self) -> nn.ReLU:
        return nn.ReLU()

    def forward(self, *ts: Union[Tensor, Ele]) -> Union[Tensor, Ele, Tuple[Tensor, ...]]:
        return _distribute_to_super(super().forward, *ts)
    pass


class Tanh(nn.Tanh):
    def __str__(self):
        return f'{Dom.name}.' + super().__str__()

    def export(self) -> nn.Tanh:
        return nn.Tanh()

    def forward(self, *ts: Union[Tensor, Ele]) -> Union[Tensor, Ele, Tuple[Tensor, ...]]:
        return _distribute_to_super(super().forward, *ts)
    pass


class MaxPool1d(nn.MaxPool1d):
    def __str__(self):
        return f'{Dom.name}.' + super().__str__()

    def forward(self, *ts: Union[Tensor, Ele]) -> Union[Tensor, Ele, Tuple[Tensor, ...]]:
        return _distribute_to_super(super().forward, *ts)
    pass


class MaxPool2d(nn.MaxPool2d):
    def __str__(self):
        return f'{Dom.name}.' + super().__str__()

    def forward(self, *ts: Union[Tensor, Ele]) -> Union[Tensor, Ele, Tuple[Tensor, ...]]:
        return _distribute_to_super(super().forward, *ts)
    pass
