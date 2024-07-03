import torch
import torch.nn as nn

from torch import Tensor

from .bound_ops import LowerBound


class NonNegativeParametrizer(nn.Module):  
    """
    Used for stability during training.
    """
    pedestal: Tensor

    def __init__(self, minimum: float = 0, reparam_offset: float = 2 ** -18):
        super().__init__()

        self.minimum = float(minimum)
        self.reparam_offset = float(reparam_offset)

        pedestal = self.reparam_offset ** 2
        self.register_buffer("pedestal", torch.Tensor([pedestal]))
        bound = (self.minimum + self.reparam_offset ** 2) ** 0.5
        self.lower_bound = LowerBound(bound)

    def init(self, x: Tensor) -> Tensor:
        return torch.sqrt(torch.max(x + self.pedestal, self.pedestal))

    def forward(self, x: Tensor) -> Tensor:
        out = self.lower_bound(x)
        out = out ** 2 - self.pedestal
        return out

if __name__ == "__main__":
    gamma_init = 0.1
    nonn = NonNegativeParametrizer()
    gamma = gamma_init * torch.eye(5)
    gamma = nonn.init(gamma)
    print(gamma)
    gamma = nonn(gamma)
    print(gamma)