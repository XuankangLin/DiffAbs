from typing import Union, Tuple

from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset


def valid_lb_ub(lb: Union[float, Tensor], ub: Union[float, Tensor], eps: float = 1e-5) -> bool:
    """ Valid if (1) Size ==; (2) LB <= UB.
    :param eps: added for numerical instability.
    """
    if isinstance(lb, float) and isinstance(ub, float):
        return lb <= ub + eps

    if lb.size() != ub.size():
        return False

    # '<=' will return a uint8 tensor of 1 or 0 for each element, it should have all 1s.
    return (lb <= ub + eps).all().item()


def divide_pos_neg(ws: Tensor) -> Tuple[Tensor, Tensor]:
    """
    :return: positive part and negative part of the original tensor, 0 filled elsewhere.
    """
    pos_weights = F.relu(ws)
    neg_weights = F.relu(ws * -1) * -1
    return pos_weights, neg_weights


class AbsData(Dataset):
    """ Storing the split LB/UB boxes/abstractions. """
    def __init__(self, boxes_lb: Tensor, boxes_ub: Tensor, boxes_extra: Tensor = None):
        assert valid_lb_ub(boxes_lb, boxes_ub)
        self.boxes_lb = boxes_lb
        self.boxes_ub = boxes_ub
        self.boxes_extra = boxes_extra
        return

    def __len__(self):
        return len(self.boxes_lb)

    def __getitem__(self, idx):
        if self.boxes_extra is None:
            return self.boxes_lb[idx], self.boxes_ub[idx]
        else:
            return self.boxes_lb[idx], self.boxes_ub[idx], self.boxes_extra[idx]
    pass
