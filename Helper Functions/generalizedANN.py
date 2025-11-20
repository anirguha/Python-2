
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class MLP(nn.Module):
    """
    sizes: list of layer sizes, e.g. [in, h1, h2, ..., out]
    activation: activation for hidden layers (callable)
    out_activation: activation for the final layer (callable or None)
    dropout: float in [0,1) (applied to hidden layers)
    batchnorm: bool (BatchNorm1d on hidden layers)
    """
    def __init__(self, sizes, activation=nn.ReLU(), out_activation=None,
                 dropout=0.0, batchnorm=False):
        super().__init__()
        assert len(sizes) >= 2, "sizes must include [in_features, out_features]"

        layers = []
        for i in range(len(sizes)-1):
            in_f, out_f = sizes[i], sizes[i+1]
            layers.append(nn.Linear(in_f, out_f))

        self.layers = nn.ModuleList(layers)
        self.activation = activation
        self.out_activation = out_activation
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else None
        self.bn = nn.ModuleList(
            [nn.BatchNorm1d(sizes[i+1]) for i in range(len(sizes)-2)]
        ) if batchnorm else None

        # Kaiming init for hidden layers (good for ReLU-like activations)
        for lin in self.layers[:-1]:
            nn.init.kaiming_normal_(lin.weight, nonlinearity='relu')
            nn.init.zeros_(lin.bias)

    def forward(self, x):
        # hidden layers
        for i, lin in enumerate(self.layers[:-1]):
            x = lin(x)
            if self.bn: x = self.bn[i](x)
            x = self.activation(x)
            if self.dropout: x = self.dropout(x)
        # output layer
        x = self.layers[-1](x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x
