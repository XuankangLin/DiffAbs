""" Some tests should be implemented for each abstract domain. """

import random
from pathlib import Path

import torch
import torchvision
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

from diffabs.abs import AbsDom


IMG_DIR = Path(__file__).resolve().parent / 'imgs'
IMG_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def linear_degen(dom: AbsDom, ntimes: int = 10):
    """ Validate that my Linear implementation is correct given degenerated inputs. """
    lin = dom.Linear(in_features=2, out_features=2).to(device)
    for _ in range(ntimes):
        orig_inputs = torch.tensor([
            [random.random(), random.random()],
            [random.random(), random.random()]
        ], device=device)

        orig_outputs = lin(orig_inputs)
        outs = lin(dom.Ele.by_intvl(orig_inputs, orig_inputs))
        assert torch.equal(orig_outputs, outs.lb())
        assert torch.equal(orig_outputs, outs.ub())
    return


def maintain_lbub(dom: AbsDom, ntimes: int = 10):
    """ Validate that the inner data structure is maintaining LB/UB correctly. """
    for _ in range(ntimes):
        t1t2 = torch.stack((torch.randn(3, 3, device=device), torch.randn(3, 3, device=device)), dim=-1)
        lb, _ = torch.min(t1t2, dim=-1)
        ub, _ = torch.max(t1t2, dim=-1)

        e = dom.Ele.by_intvl(lb, ub)

        assert torch.allclose(e.lb(), lb)
        assert torch.allclose(e.ub(), ub)
    return


def relu_by_ub(dom: AbsDom, ntimes: int = 10):
    """ Validate that the ReLU approximation is correct by checking its afterward UB. """
    for _ in range(ntimes):
        t1t2 = torch.stack((torch.randn(3, 3, device=device), torch.randn(3, 3, device=device)), dim=-1)
        lb, _ = torch.min(t1t2, dim=-1)
        ub, _ = torch.max(t1t2, dim=-1)
        e = dom.Ele.by_intvl(lb, ub)

        lin = dom.Linear(3, 2).to(device)
        relu = dom.ReLU()

        out = lin(e)
        final = relu(out)
        threshold = 1e-6  # allow some numerical differences

        def _is_close(v1: Tensor, v2, filter_bits: Tensor, v2_alt=None):
            diff_bits1 = (v1 - v2).abs() > threshold
            diff_bits1 = diff_bits1 & filter_bits
            if v2_alt is None:
                return not diff_bits1.any()

            diff_bits2 = (v1 - v2_alt).abs() > threshold
            diff_bits2 = diff_bits2 & filter_bits
            diff_bits = diff_bits1 & diff_bits2  # different from both
            return not diff_bits.any()

        lbge0 = out.lb().ge(0)
        assert _is_close(final.lb(), out.lb(), lbge0)
        assert _is_close(final.ub(), out.ub(), lbge0)

        ublt0 = out.ub().le(0)
        assert _is_close(final.lb(), 0, ublt0)
        assert _is_close(final.ub(), 0, ublt0)

        ubge0 = out.ub().ge(0)
        assert _is_close(final.ub(), out.ub(), ubge0)

        # lastly, if ub >= 0 and lb <= 0, it should change lb to either 0 (k=0) or preserve (k=1)
        approx = out.ub().ge(0) & out.lb().le(0)
        assert _is_close(final.lb(), out.lb(), approx, v2_alt=0)
    return


def tanh_by_lbub(dom: AbsDom, ntimes: int = 10):
    """ Validate that the Tanh approximation is correct by checking its afterward LB/UB. """
    for _ in range(ntimes):
        t1t2 = torch.stack((torch.randn(3, 3, device=device), torch.randn(3, 3, device=device)), dim=-1)
        lb, _ = torch.min(t1t2, dim=-1)
        ub, _ = torch.max(t1t2, dim=-1)
        e = dom.Ele.by_intvl(lb, ub)

        lin = dom.Linear(3, 2).to(device)
        tanh = dom.Tanh()

        out = lin(e)

        out_lb, out_ub = out.gamma()  # DP for linear layer is already validated

        conc_final_lb = tanh(out_lb)
        conc_final_ub = tanh(out_ub)
        abs_final = tanh(out)
        abs_final_lb, abs_final_ub = abs_final.gamma()

        threshold = 1e-6  # allow some numerical error

        # sometimes error larger than torch.allclose() accepts
        # self.assertTrue(torch.allclose(conc_final_lb, abs_final_lb))
        # self.assertTrue(torch.allclose(conc_final_ub, abs_final_ub))
        diff_lb = (conc_final_lb - abs_final_lb).abs()
        diff_ub = (conc_final_ub - abs_final_ub).abs()
        assert diff_lb.max() < threshold
        assert diff_ub.max() < threshold
    return


def maxpool1d_by_lbub(dom: AbsDom, ntimes: int = 10):
    """ Validate that the MaxPool1d layer is correct by checking its afterward LB/UB. """
    for _ in range(ntimes):
        t1t2 = torch.stack((torch.randn(10, 1, 40, device=device), torch.randn(10, 1, 40, device=device)), dim=-1)
        lb, _ = torch.min(t1t2, dim=-1)
        ub, _ = torch.max(t1t2, dim=-1)
        e = dom.Ele.by_intvl(lb, ub)

        mp = dom.MaxPool1d(4, stride=2)

        out = mp(e)
        out_lb, out_ub = out.gamma()

        conc_mp_lb = mp(lb)
        conc_mp_ub = mp(ub)

        threshold = 1e-6  # allow some numerical error

        # sometimes error larger than torch.allclose() accepts
        # assert torch.allclose(out_lb, conc_mp_lb)
        # assert torch.allclose(out_ub, conc_mp_ub)

        diff_lb = (out_lb - conc_mp_lb).abs()
        diff_ub = (out_ub - conc_mp_ub).abs()
        assert diff_lb.max() < threshold
        assert diff_ub.max() < threshold
    return


def maxpool2d_specific(dom: AbsDom):
    """ Validate my MaxPool layer implementation using a hand-written example. """
    ins = torch.tensor([[[
        [1, 1, 2, 4],
        [5, 6, 7, 8],
        [3, 2, 1, 0],
        [1, 2, 3, 4]
    ]]]).float()

    goals = torch.tensor([[[
        [6, 8],
        [3, 4]
    ]]]).float()

    pool = dom.MaxPool2d(2, 2)
    e = dom.Ele.by_intvl(ins, ins)
    outs = pool(e)
    outs_lb, outs_ub = outs.gamma()

    assert torch.equal(outs_lb, goals)
    assert torch.equal(outs_ub, goals)
    return


def maxpool2d_degen(dom: AbsDom, ntimes: int = 10):
    """ Validate my MaxPool layer implementation by comparing the outputs of degenerated intervals. """
    ds = torchvision.datasets.CIFAR10(root=IMG_DIR, train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
    trainloader = data.DataLoader(ds, batch_size=4, shuffle=True)

    cnt = 0
    for imgs, _ in trainloader:
        if cnt >= ntimes:
            break
        cnt += 1

        pool = dom.MaxPool2d((2, 2)).to(device)
        imgs = imgs.to(device)
        with torch.no_grad():
            conc_outs = pool(imgs)

            ele = dom.Ele.by_intvl(lb=imgs, ub=imgs)
            outs = pool(ele)
            outs_lb, outs_ub = outs.gamma()

        assert torch.allclose(conc_outs, outs_lb)
        assert torch.allclose(conc_outs, outs_ub)
    return


def conv_degen(dom: AbsDom, ntimes: int = 10, eps: float = 1e-6):
    """ Validate my Convolutional layer implementation by comparing the outputs of degenerated intervals. """
    ds = torchvision.datasets.CIFAR10(root=IMG_DIR, train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
    trainloader = data.DataLoader(ds, batch_size=2, shuffle=True)

    cnt = 0
    for imgs, _ in trainloader:
        if cnt >= ntimes:
            break
        cnt += 1

        # conv = Conv2d(3, 6, 5)
        conv = dom.Conv2d(3, 6, 5, padding=2).to(device)
        imgs = imgs.to(device)
        with torch.no_grad():
            conc_outs = conv(imgs)

            ele = dom.Ele.by_intvl(lb=imgs, ub=imgs)
            outs = conv(ele)
            outs_lb, outs_ub = outs.gamma()

        # sometimes error larger than torch.allclose() accepts..
        # assert torch.allclose(conc_outs, outs_lb)
        # assert torch.allclose(conc_outs, outs_ub)
        diff1 = conc_outs - outs_lb
        diff2 = conc_outs - outs_ub
        assert (diff1.abs() < eps).all()
        assert (diff2.abs() < eps).all()
    return


def _sample_points(lb: Tensor, ub: Tensor, K: int) -> Tensor:
    """ Uniformly sample K points for each region.
    :param lb: Lower bounds, batched
    :param ub: Upper bounds, batched
    :param K: how many pieces to sample
    """
    assert (lb <= ub).all() and K >= 1

    repeat_dims = [1] * (len(lb.size()) - 1)
    base = lb.repeat(K, *repeat_dims)  # repeat K times in the batch, preserving the rest dimensions
    width = (ub - lb).repeat(K, *repeat_dims)

    coefs = torch.rand_like(base)
    pts = base + coefs * width
    return pts


def overapprox(dom: AbsDom, acti: nn.Module, in_features: int = 3, out_features: int = 3, hidden_neurons: int = 10,
               ntimes: int = 10, npts: int = 1000):
    """ Validate that the implementation correctly over-approximates given ranges of inputs, by sampling. """
    for _ in range(ntimes):
        t1t2 = torch.stack((torch.randn(1, in_features, device=device),
                            torch.randn(1, in_features, device=device)), dim=-1)
        lb, _ = torch.min(t1t2, dim=-1)
        ub, _ = torch.max(t1t2, dim=-1)

        e = dom.Ele.by_intvl(lb, ub)
        net = nn.Sequential(
            dom.Linear(in_features, hidden_neurons),
            acti(),
            dom.Linear(hidden_neurons, out_features)
        ).to(device)
        out = net(e)
        out_lb, out_ub = out.gamma()

        pts = _sample_points(lb, ub, npts)
        out_pts = net(pts)

        assert (out_pts >= out_lb).all()
        assert (out_pts <= out_ub).all()
    return


def optimizable(dom: AbsDom, batch_size: int = 4, in_features: int = 2, out_features: int = 2,
                hidden_neurons: int = 4):
    """ Validate that my Linear layer and/or other activation functions can be optimized. """
    inputs = torch.randn(batch_size, in_features, out_features, device=device)
    inputs_lb, _ = torch.min(inputs, dim=-1)
    inputs_ub, _ = torch.max(inputs, dim=-1)
    ins = dom.Ele.by_intvl(inputs_lb, inputs_ub)

    mse = nn.MSELoss()

    def _loss(outputs_lb):
        lows = outputs_lb[:, 0]
        distances = 0 - lows
        distances = F.relu(distances)
        prop = torch.zeros_like(distances)
        return mse(distances, prop)

    while True:
        net = nn.Sequential(
            dom.Linear(in_features=in_features, out_features=hidden_neurons),
            dom.ReLU(),
            dom.Linear(in_features=hidden_neurons, out_features=out_features)
        ).to(device)

        with torch.no_grad():
            outs = net(ins)

        if _loss(outs.lb()) > 0:
            break

    # Now the layer has something to optimize
    print('===== optimizable(): =====')
    print('Using inputs LB:', inputs_lb)
    print('Using inputs UB:', inputs_ub)
    print('Before any optimization, the approximated output is:')
    print('Outputs LB:', outs.lb())
    print('Outputs UB:', outs.ub())

    opti = torch.optim.Adam(net.parameters(), lr=0.1)
    retrained = 0
    while True:
        opti.zero_grad()
        outs = net(ins)
        loss = _loss(outs.lb())
        if loss <= 0:
            # until the final output's 1st element is >= 0
            break

        loss.backward()
        opti.step()
        retrained += 1
        print('Iter', retrained, '- loss', loss.item())
        pass

    with torch.no_grad():
        print(f'All optimized after {retrained} retrains. Now the final outputs 1st element should be >= 0.')
        outs = net(ins)
        print('Outputs LB:', outs.lb())
        print('Outputs UB:', outs.ub())
        assert (outs.lb()[:, 0] >= 0.).all()
    return retrained


def clamp(dom: AbsDom, ntimes: int = 10, batch_size: int = 10):
    min, max = -.5, .5
    c = dom.Clamp(min, max)

    for _ in range(ntimes):
        # test concrete cases
        x = torch.randn(batch_size, batch_size, device=device)
        assert torch.allclose(c(x), torch.clamp(x, min, max))

        # test abstract cases
        t1t2 = torch.stack((torch.randn(batch_size, batch_size, device=device),
                            torch.randn(batch_size, batch_size, device=device)), dim=-1)
        lb, _ = torch.min(t1t2, dim=-1)
        ub, _ = torch.max(t1t2, dim=-1)
        e = dom.Ele.by_intvl(lb, ub)
        outs_lb, outs_ub = c(e).gamma()

        assert (min <= outs_lb).all() and (outs_lb <= max).all()
        assert (min <= outs_ub).all() and (outs_ub <= max).all()
    return
