import torch
import torch.nn as nn
from typing import Iterable, Tuple, Union

class TitanicMLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_units: Union[Tuple[int, ...], Iterable[int]] = (64, 32),
        dropout_rate: float = 0.5,
        output_activation: str = "linear",
    ):
        super().__init__()

        if isinstance(hidden_units, Iterable) and not isinstance(hidden_units, tuple):
            hidden_units = tuple(hidden_units)

        layers = []
        prev = input_dim
        for h in hidden_units:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout_rate and dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev = h

        layers.append(nn.Linear(prev, 1))

        if output_activation.lower() == "sigmoid":
            layers.append(nn.Sigmoid())
        elif output_activation.lower() == "linear":
            pass
        else:
            raise ValueError("output_activation must be 'linear' or 'sigmoid'")

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, input_dim) float32
        returns: (N, 1) tensor of logits (or probs if you forced sigmoid)
        """
        return self.model(x)
