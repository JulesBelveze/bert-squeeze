import torch


class Lr(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = torch.nn.Linear(10, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """"""
        return self.model(inputs)
