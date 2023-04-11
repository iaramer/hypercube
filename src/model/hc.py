import torch
from torch import nn
import numpy as np


# TODO: add more flexibility for unequal input-output shapes
class HyperCubeLayer(nn.Module):
    __constants__ = ['in_features', 'out_sqrt_features']
    in_features: int
    out_sqrt_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_sqrt_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        hc_input_size = np.sqrt(in_features)
        assert hc_input_size % 1 == 0
        self.hc_input_size = hc_input_size = int(hc_input_size)
        self.in_features = in_features
        self.out_sqrt_features = out_sqrt_features  # No. of output features = out_sqrt_features * sqrt(in_features)
        self.weight = nn.Parameter(torch.empty((out_sqrt_features, hc_input_size, hc_input_size), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((out_sqrt_features,), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        return 'in_features={}, hc_input_size={}, out_sqrt_features={}, bias={}'.format(
            self.in_features, self.hc_input_size, self.out_sqrt_features, self.bias is not None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view((*x.shape[:-1], self.hc_input_size, self.hc_input_size))
        x = (x.movedim(1,2) @ self.weight).movedim(2,1) + self.bias
        x = x.flatten(start_dim=-2)
        return x


# TODO: stack the tensors without redundant reshaping,
#  the 3D output from hc1 is provided directly to hc2
class HyperCubeBlock(nn.Module):
    def __init__(self, in_features: int, out_sqrt_features: int, bias: bool = True,
                 device=None, dtype=None):
        if out_sqrt_features is None:
            out_sqrt_features = in_features
        super(HyperCubeBlock, self).__init__()
        self.hc_layers_1 = HyperCubeLayer(
            in_features, 
            int(np.sqrt(in_features)),  # TODO: fix
            bias=bias,
            device=device, 
            dtype=dtype
        )
        self.hc_layers_2 = HyperCubeLayer(
            in_features, 
            out_sqrt_features,
            bias=bias,
            device=device, 
            dtype=dtype
        )
        # self.relu = nn.ReLU()
            
    def forward(self, x):
        x = self.hc_layers_1(x)
        # TODO: Check if needed
        # sq = int(np.sqrt(x.shape[-1]))
        # x = x.view((*x.shape[:-1], sq, sq))
        # x = x.transpose(-1,-2)
        # x = x.flatten(start_dim=-2)

        # x = self.relu(x)  # TODO: Experiment with intermediate activation
        
        x = self.hc_layers_2(x)
        return x
