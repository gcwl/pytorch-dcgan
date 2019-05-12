import torch
import torch.nn as nn


def to_timedict(unix_time, frac_secs=False):
    units = (("hours", 60 * 60), ("mins", 60), ("secs", 1))
    if frac_secs:
        units += (("msec", 1e-3), ("usec", 1e-6))
    res = {}
    for unit, value in units:
        t, unix_time = divmod(unix_time, value)
        res[unit] = int(t)
    return res


def compose(*funcs):
    def g(x):
        for f in funcs:
            if f is None:
                continue
            x = f(x)
        return x

    return g


def weights_init(m):
    if isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        m.weight.data.normal_(0.0, 0.02)
