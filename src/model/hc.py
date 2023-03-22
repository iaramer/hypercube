import torch
from torch import nn
import math


def linear_2d(x, w, b=None):
    x = x.expand((w.shape[1],*x.shape)).movedim(0,-2) * w
    x = torch.sum(x, axis=-1) + b
    return x


class Linear3D(nn.Module):
    """
    This is a 3D linear layer. It represents `n_2d_layers` 
    2D linear layers and performs a dimension-wise multiplication.
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, n_2d_layers: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_2d_layers = n_2d_layers
        self.weight = nn.Parameter(torch.empty((in_features,n_2d_layers,out_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((n_2d_layers,out_features), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input must be of size (N, *, n_2d_layers, in_features),
        where N - is the number of samples in the batch, * - additional dimensions.
        """
        return linear_2d(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, n_2d_layers={}, out_features={}, bias={}'.format(
            self.in_features, self.n_2d_layers, self.out_features, self.bias is not None
        )