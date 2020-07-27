""" By default, the usage of abstract elements by passing in AbsEle conflicts with forward/backward hooks, as the
    PyTorch hook code deals with Tensors only. To work around, DiffAbs allows passing in a tuple of tensors that
    constitute an abstract element, the hook shall now binds correctly to each of these tensors.

    Tested on interval domain, the rest domains are implemented similarly.
"""

import torch
from torch import nn
from torch.nn import functional as F

from diffabs.interval import Dom


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dom = Dom()


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

    # register forward/backward hooks
    def forward_callback(module, input, output):
        print(f'In forward hook: input: {input}, output: {output}')
        return

    def backward_callback(module, grad_input, grad_output):
        print(f'In backward hook: grad-in: {grad_input}, grad-out: {grad_output}')
        return

    net.register_forward_hook(forward_callback)
    net.register_backward_hook(backward_callback)

    opti = torch.optim.Adam(net.parameters(), lr=0.1)
    retrained = 0
    ins_ts = tuple(ins)
    while True:
        opti.zero_grad()
        # now needs to pass in tensors instead of abstrct element
        outs_ts = net(*ins_ts)
        outs_e = dom.Ele(*outs_ts)
        loss = _loss(outs_e.lb())
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
        outs_ts = net(*ins_ts)
        outs_e = dom.Ele(*outs_ts)
        print('Outputs LB:', outs_e.lb())
        print('Outputs UB:', outs_e.ub())
        assert (outs_e.lb()[:, 0] >= 0.).all()
    return retrained
