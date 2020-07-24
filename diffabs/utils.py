from typing import Union, Tuple

from torch import Tensor
from torch.nn import functional as F


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
