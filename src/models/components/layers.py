import torch
from torch import nn


class BatchNormLayer(nn.Module):
    def __init__(self, neuron_cnt: int, activation_func: nn.Module):
        super().__init__()
        self.linear = nn.Linear(neuron_cnt, neuron_cnt)
        self.bn = nn.BatchNorm1d(neuron_cnt)
        self.activation = activation_func(inplace=True)
        self._weights_init()

    def _weights_init(self):
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.activation(self.bn(self.linear(x)))
    

class LinearLayer(nn.Module):
    def __init__(self, neuron_cnt: int, activation_func: nn.Module):
        super.__init__()
        self.linear = nn.Linear(neuron_cnt, neuron_cnt)
        self.activation = activation_func(inplace=True)
        self._weights_init()

    def _weights_init(self):
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.activation(self.linear(x))
    

