""" Define some functions used in concrete domain. """

import torch
from torch import Tensor
from torch.nn import functional as F

from diffabs.abs import MetaFunc


class ConcDist(MetaFunc):
    """ Similar to AbsEle in abs.py, it needs the distance for concrete data points as well. Implementation similar to
        the non-relational interval domain. Note that no eps is set for ConcDist, it could, but it is fine since
        ConcDist is mainly used for validation but not training.
    """

    @classmethod
    def col_le_val(cls, outs: Tensor, idx: int, threshold: float, mean: float = 0., range: float = 1.) -> Tensor:
        t = outs[..., idx]
        threshold = (threshold - mean) / range
        d = t - threshold
        return F.relu(d)

    @classmethod
    def col_ge_val(cls, outs: Tensor, idx: int, threshold: float, mean: float = 0., range: float = 1.) -> Tensor:
        t = outs[..., idx]
        threshold = (threshold - mean) / range
        d = threshold - t
        return F.relu(d)

    @classmethod
    def cols_not_max(cls, outs: Tensor, *idxs: int) -> Tensor:
        others = cls._idxs_not(outs, *idxs)
        others = outs[..., others]

        res = []
        for i in idxs:
            target = outs[..., [i]]
            diff = target - others  # will broadcast
            diff = F.relu(diff)
            mins, _ = torch.min(diff, dim=-1)
            res.append(mins)
        return sum(res)

    @classmethod
    def cols_is_max(cls, outs: Tensor, *idxs: int) -> Tensor:
        others = cls._idxs_not(outs, *idxs)
        others = outs[..., others]

        res = []
        for i in idxs:
            target = outs[..., [i]]
            diffs = others - target  # will broadcast
            diffs = F.relu(diffs)
            res.append(diffs)

        if len(idxs) == 1:
            all_diffs = res[0]
        else:
            all_diffs = torch.stack(res, dim=-1)
            all_diffs, _ = torch.min(all_diffs, dim=-1)  # it's OK to have either one to be max, thus use torch.min()

        # then it needs to surpass everybody else, thus use torch.max() for maximum distance
        diffs, _ = torch.max(all_diffs, dim=-1)
        return diffs

    @classmethod
    def cols_not_min(cls, outs: Tensor, *idxs: int) -> Tensor:
        others = cls._idxs_not(outs, *idxs)
        others = outs[..., others]

        res = []
        for i in idxs:
            target = outs[..., [i]]
            diffs = others - target  # will broadcast
            diffs = F.relu(diffs)
            mins, _ = torch.min(diffs, dim=-1)
            res.append(mins)
        return sum(res)

    @classmethod
    def cols_is_min(cls, outs: Tensor, *idxs: int) -> Tensor:
        others = cls._idxs_not(outs, *idxs)
        others = outs[..., others]

        res = []
        for i in idxs:
            target = outs[..., [i]]
            diffs = target - others  # will broadcast
            diffs = F.relu(diffs)
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
