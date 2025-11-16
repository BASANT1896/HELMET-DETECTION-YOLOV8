# spd.py
import torch
import torch.nn as nn


class SPDConv(nn.Module):
    """
    Spatially Parametrized Dynamic Convolution (SPDConv)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, bias=False, use_modulation=False):
        super().__init__()
        self.use_modulation = use_modulation

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size, stride, padding,
            bias=bias
        )

        if use_modulation:
            self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1))
            self.shift = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, x):
        out = self.conv(x)
        if self.use_modulation:
            out = out * self.scale + self.shift
        return out


def replace_conv_with_spd(model, use_modulation=True):
    """
    Recursively replace all nn.Conv2d layers with SPDConv layers.
    """
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Conv2d):
            new_conv = SPDConv(
                module.in_channels,
                module.out_channels,
                module.kernel_size[0],
                module.stride[0],
                module.padding[0],
                bias=(module.bias is not None),
                use_modulation=use_modulation
            )
            setattr(model, name, new_conv)
        else:
            replace_conv_with_spd(module, use_modulation)
