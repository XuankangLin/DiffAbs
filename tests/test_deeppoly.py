import torch
from torch import Tensor

from diffabs.deeppoly import Dom
from tests import common


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def test_maintain_lbub():
    common.maintain_lbub(Dom())
    return


def test_linear_degen():
    common.linear_degen(Dom())
    return


def test_ex_fig3():
    """ Validate the implementation of DeepPoly domain (without ReLU) using concrete example of Fig. 3 in paper. """
    dom = Dom()

    input_bounds = torch.tensor([[
        [-1., 1.],
        [-1., 1.]
    ]], device=device)

    lb = input_bounds[:, :, 0]
    ub = input_bounds[:, :, 1]
    e = dom.Ele.by_intvl(lb, ub)

    lin = dom.Linear(2, 2)
    with torch.no_grad():
        lin.weight.data = torch.tensor([
            [1., 1.],
            [1., -1.]
        ])
        lin.bias.data.zero_()
        lin = lin.to(device)
        out = lin(e)

    answer_lb = torch.tensor([[-2., -2.]], device=device)
    answer_ub = torch.tensor([[2., 2.]], device=device)
    out_lb, out_ub = out.gamma()
    assert torch.equal(out_lb, answer_lb)
    assert torch.equal(out_ub, answer_ub)
    return


def test_ex_fig3_full():
    """ Validate the implementation of DeepPoly domain (with ReLU) using concrete example of Fig. 3 in paper. """
    dom = Dom()

    input_bounds = torch.tensor([[
        [-1., 1.],
        [-1., 1.]
    ]], device=device)

    lb = input_bounds[:, :, 0]
    ub = input_bounds[:, :, 1]
    input = dom.Ele.by_intvl(lb, ub)

    relu = dom.ReLU()
    lin1 = dom.Linear(2, 2)  # from [x1, x2] to [x3, x4]
    lin2 = dom.Linear(2, 2)  # from [x5, x6] to [x7, x8]
    lin3 = dom.Linear(2, 2)  # from [x9, x10] to [x11, x12]
    with torch.no_grad():
        lin1.weight.data = torch.tensor([
            [1., 1.],
            [1., -1.]
        ])
        lin1.bias.data.zero_()

        lin2.weight.data = torch.tensor([
            [1., 1.],
            [1., -1.]
        ])
        lin2.bias.data.zero_()

        lin3.weight.data = torch.tensor([
            [1., 1.],
            [0., 1.]
        ])
        lin3.bias.data = torch.tensor([1., 0.])

        lin1 = lin1.to(device)
        lin2 = lin2.to(device)
        lin3 = lin3.to(device)

    def _go(layer: torch.nn.Module, e: dom.Ele, ans_lb: Tensor, ans_ub: Tensor) -> dom.Ele:
        out = layer(e)
        assert torch.equal(ans_lb, out.lb())
        assert torch.equal(ans_ub, out.ub())
        return out

    out34 = _go(lin1, input,
                ans_lb=torch.tensor([[-2., -2.]], device=device),
                ans_ub=torch.tensor([[2., 2.]], device=device))
    out56 = _go(relu, out34,
                ans_lb=torch.tensor([[0., 0.]], device=device),
                ans_ub=torch.tensor([[2., 2.]], device=device))
    out78 = _go(lin2, out56,
                ans_lb=torch.tensor([[0., -2.]], device=device),
                ans_ub=torch.tensor([[3., 2.]], device=device))
    out910 = _go(relu, out78,
                 ans_lb=torch.tensor([[0., 0.]], device=device),
                 ans_ub=torch.tensor([[3., 2.]], device=device))

    # x11's UB is 5.5 on paper, but that should be wrong, 6.0 should be the correct answer
    out1112 = _go(lin3, out910,
                  ans_lb=torch.tensor([[1., 0.]], device=device),
                  ans_ub=torch.tensor([[6., 2.]], device=device))
    return


def test_relu_by_ub():
    common.relu_by_ub(Dom())
    return


def test_tanh_by_lbub():
    common.tanh_by_lbub(Dom())
    return


def test_maxpool1d_by_lbub():
    common.maxpool1d_by_lbub(Dom())
    return


def test_maxpool2d_specific():
    common.maxpool2d_specific(Dom())
    return


def test_overapprox():
    dom = Dom()
    common.overapprox(dom, dom.ReLU)
    common.overapprox(dom, dom.Tanh)
    return


def test_optimizable():
    return common.optimizable(Dom())


def test_maxpool2d_degen():
    common.maxpool2d_degen(Dom())
    return


def test_conv_degen():
    common.conv_degen(Dom())
    return


def test_clamp():
    return common.clamp(Dom())
