import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.stateless import functional_call

from .common import unpack_batch


class QuadraticBezier(nn.Module):
    def __init__(self, endpoint_a, endpoint_b):
        super().__init__()
        self.template = copy.deepcopy(endpoint_a)
        state_a = endpoint_a.state_dict()
        state_b = endpoint_b.state_dict()
        self.names = []
        controls = []
        self.endpoint_a = {}
        self.endpoint_b = {}
        for name, parameter in endpoint_a.named_parameters():
            self.names.append(name)
            a = state_a[name].detach().clone()
            b = state_b[name].detach().clone()
            self.endpoint_a[name] = a
            self.endpoint_b[name] = b
            controls.append(nn.Parameter(0.5 * (a + b)))
        self.controls = nn.ParameterList(controls)
        self.buffers = {
            name: value.detach().clone() for name, value in endpoint_a.named_buffers()
        }

    def _state_at(self, t):
        state = {}
        for name, control in zip(self.names, self.controls):
            a = self.endpoint_a[name].to(control.device)
            b = self.endpoint_b[name].to(control.device)
            state[name] = (1.0 - t) ** 2 * a + 2.0 * t * (1.0 - t) * control + t ** 2 * b
        for name, value in self.buffers.items():
            state[name] = value.to(next(self.parameters()).device)
        return state

    def forward(self, x_i, x_t, len_i, len_t, t):
        return functional_call(self.template, self._state_at(float(t)), (x_i, x_t, len_i, len_t))

    def materialize(self, t):
        model = copy.deepcopy(self.template)
        state = model.state_dict()
        curve_state = self._state_at(float(t))
        for name in state:
            if name in curve_state:
                state[name] = curve_state[name].detach().to(state[name].device)
        model.load_state_dict(state, strict=True)
        return model


def train_bezier_curve(endpoint_a, endpoint_b, loader, device, epochs, lr, t, eval_interval=20, eval_fn=None, tag=""):
    curve = QuadraticBezier(endpoint_a.to(device), endpoint_b.to(device)).to(device)
    optimizer = torch.optim.SGD(curve.controls, lr=float(lr), momentum=0.9, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1))
    history = []
    for epoch in range(int(epochs)):
        curve.train()
        total, batches = 0.0, 0
        for batch in loader:
            x_i, x_t, len_i, len_t, y = unpack_batch(batch, device)
            optimizer.zero_grad()
            logits = curve(x_i, x_t, len_i, len_t, t)[0]
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        scheduler.step()
        row = {"epoch": epoch + 1, "loss": total / max(batches, 1), "lr": optimizer.param_groups[0]["lr"]}
        if eval_fn is not None and ((epoch + 1) % int(eval_interval) == 0 or epoch + 1 == int(epochs)):
            row["dev"] = eval_fn(curve.materialize(t).to(device))
        history.append(row)
        logging.info("[TSC:%s] curve epoch %d/%d loss %.4f", tag, epoch + 1, epochs, row["loss"])
    return curve.materialize(t).to(device), history
