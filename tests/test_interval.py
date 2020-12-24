import torch

from diffabs.interval import Dom
from tests import common


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def test_maintain_lbub():
    return common.maintain_lbub(Dom())


def test_linear_degen():
    return common.linear_degen(Dom())


def test_linear_specific():
    """ Validate that my Linear implementation is correct given interval inputs. """
    dom = Dom()
    lin = dom.Linear(in_features=2, out_features=1).to(device)
    inputs = torch.tensor([
        [[-2, -1], [-1, 1]],
        [[-0.5, 0.5], [1.5, 3]]
    ], device=device)
    inputs_lb = inputs[:, :, 0]
    inputs_ub = inputs[:, :, 1]

    with torch.no_grad():
        lin.weight[0][0] = -0.5
        lin.weight[0][1] = 0.5
        lin.bias[0] = -1

    outs = lin(dom.Ele.by_intvl(inputs_lb, inputs_ub))
    answer = torch.tensor([
        [[-1, 0.5]],
        [[-0.5, 0.75]]
    ], device=device)
    answer_lb = answer[:, :, 0]
    answer_ub = answer[:, :, 1]
    assert torch.equal(answer_lb, outs.lb())
    assert torch.equal(answer_ub, outs.ub())
    return


def test_dot_prod(ntimes: int = 10):
    """ Validate that the dot product op in Interval domain is correct. """
    dom = Dom()
    for _ in range(ntimes):
        inputs = torch.randn(2, 2, 2, device=device)
        inputs_lb, _ = torch.min(inputs, dim=-1)
        inputs_ub, _ = torch.max(inputs, dim=-1)
        ins = dom.Ele.by_intvl(inputs_lb, inputs_ub)

        ws = torch.randn(2).to(device)
        outs = ins * ws
        outs_lb, outs_ub = outs.gamma()

        for i in range(2):
            for j in range(2):
                if ws[j] >= 0:
                    assert outs_lb[i][j] == inputs_lb[i][j] * ws[j]
                    assert outs_ub[i][j] == inputs_ub[i][j] * ws[j]
                else:
                    assert outs_lb[i][j] == inputs_ub[i][j] * ws[j]
                    assert outs_ub[i][j] == inputs_lb[i][j] * ws[j]
    return


def test_relu_by_ub():
    return common.relu_by_ub(Dom())


def test_tanh_by_lbub():
    return common.tanh_by_lbub(Dom())


def test_maxpool1d_by_lbub():
    return common.maxpool1d_by_lbub(Dom())


def test_maxpool2d_specific():
    return common.maxpool2d_specific(Dom())


def test_overapprox():
    dom = Dom()
    common.overapprox(dom, dom.ReLU)
    common.overapprox(dom, dom.Tanh)
    return


def test_optimizable():
    return common.optimizable(Dom())


def test_maxpool2d_degen():
    return common.maxpool2d_degen(Dom())


def test_conv_degen():
    return common.conv_degen(Dom())


def test_clamp():
    return common.clamp(Dom())
