# source: https://github.com/rmokady/CLIP_prefix_caption
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from typing import Tuple, Optional, Union


class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(nn.Dropout(p=0.5))
                layers.append(act())
        self.model = nn.Sequential(*layers)