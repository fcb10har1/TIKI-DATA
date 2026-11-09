"""Small offline, noncausal residual TCN. No temporal batch normalization."""

import os
import random

import numpy as np
import torch
from torch import nn

from tiki_data.contracts import require


def seed_all(seed, threads=1):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Block(nn.Module):
    def __init__(self, width, kernel, dilation, dropout):
        super().__init__()
        padding = dilation * (kernel // 2)
        self.layers = nn.Sequential(
            nn.Conv1d(width, width, kernel, padding=padding, dilation=dilation),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(width, width, kernel, padding=padding, dilation=dilation),
            nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x):
        return torch.relu(x + self.layers(x))


class TCN(nn.Module):
    def __init__(self, n_features, width=32, kernel=3, dilations=(1, 2, 4, 8, 16), dropout=0.1):
        super().__init__()
        require(isinstance(width, int) and width > 0, "width must be positive")
        require(isinstance(kernel, int) and kernel >= 3 and kernel % 2 == 1,
                "kernel must be odd and >= 3")
        require(bool(dilations) and all(isinstance(d, int) and d > 0 for d in dilations),
                "dilations must be positive integers")
        require(0 <= dropout < 1, "dropout must be in [0,1)")
        self.radius = 2 * (kernel // 2) * sum(dilations)
        self.network = nn.Sequential(nn.Conv1d(2 * n_features, width, 1), nn.ReLU(),
                                     *(Block(width, kernel, d, dropout) for d in dilations),
                                     nn.Conv1d(width, 3, 1))

    def forward(self, x):
        # (B, 2F, T) -> (B, 3, T), explicit H,A,D logits.
        return self.network(x)
