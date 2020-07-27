import random
from pathlib import Path

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

from diffabs.interval import Dom


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dom = Dom()

IMG_DIR = Path(__file__).resolve().parent / 'imgs'
IMG_DIR.mkdir(parents=True, exist_ok=True)


def test_linear_degen(ntimes: int = 10):
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


def test_linear_specific():
    """ Validate that my Linear implementation is correct given interval inputs. """
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


def test_linear_optimizable():
    """ Validate that my Linear layer can be optimized. """
    inputs = torch.randn(2, 2, 2, device=device)
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
        net = dom.Linear(in_features=2, out_features=2).to(device)
        with torch.no_grad():
            outs = net(ins)

        if _loss(outs.lb()) > 0:
            break

    # Now the layer has something to optimize
    print('===== test_linear_optimizable(): =====')
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


def test_dot_prod(ntimes: int = 10):
    """ Validate that the dot product op in Interval domain is correct. """
    for _ in range(ntimes):
        inputs = torch.randn(2, 2, 2, device=device)
        inputs_lb, _ = torch.min(inputs, dim=-1)
        inputs_ub, _ = torch.max(inputs, dim=-1)
        ins = dom.Ele.by_intvl(inputs_lb, inputs_ub)

        ws = torch.randn(2)
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


def test_conv_degen(ntimes: int = 10, eps: float = 1e-6):
    """ Validate my Convolutional layer implementation by comparing the outputs of degenerated intervals. """
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

        # conv = Conv2d(3, 6, 5)
        conv = dom.Conv2d(3, 6, 5, padding=2)

        with torch.no_grad():
            conc_outs = conv(imgs)

            ele = dom.Ele.by_intvl(lb=imgs, ub=imgs)
            outs = conv(ele)
            outs_lb, outs_ub = outs.gamma()

        # sometimes error larger than torch.allclose() accepts..
        # self.assertTrue(torch.allclose(conc_outs, outs_lb))
        # self.assertTrue(torch.allclose(conc_outs, outs_ub))
        diff1 = conc_outs - outs_lb
        diff2 = conc_outs - outs_ub
        assert (diff1.abs() < eps).all()
        assert (diff2.abs() < eps).all()
    return


def test_maxpool_specific():
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


def test_maxpool2d_degen(ntimes: int = 10):
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

        pool = dom.MaxPool2d((2, 2))
        with torch.no_grad():
            conc_outs = pool(imgs)

            ele = dom.Ele.by_intvl(lb=imgs, ub=imgs)
            outs = pool(ele)
            outs_lb, outs_ub = outs.gamma()

        assert torch.allclose(conc_outs, outs_lb)
        assert torch.allclose(conc_outs, outs_ub)
    return


def test_maxpool1d(ntimes: int = 10):
    """ Validate that the MaxPool1d layer is correct by checking its afterward LB/UB. """
    for _ in range(ntimes):
        t1t2 = torch.stack((torch.randn(10, 1, 40, device=device), torch.randn(10, 1, 40, device=device)), dim=-1)
        lb, _ = torch.min(t1t2, dim=-1)
        ub, _ = torch.max(t1t2, dim=-1)
        e = dom.Ele.by_intvl(lb, ub)

        mp = dom.MaxPool1d(4, stride=2)

        out = mp(e)
        out_lb, out_ub = out.gamma()  # DP for linear layer is already validated

        conc_mp_lb = mp(lb)
        conc_mp_ub = mp(ub)

        threshold = 1e-6  # allow some numerical error

        # sometimes error larger than torch.allclose() accepts
        # self.assertTrue(torch.allclose(out_lb, conc_mp_lb))
        # self.assertTrue(torch.allclose(out_ub, conc_mp_ub))

        diff_lb = (out_lb - conc_mp_lb).abs()
        diff_ub = (out_ub - conc_mp_ub).abs()
        assert diff_lb.max() < threshold
        assert diff_ub.max() < threshold
    return
