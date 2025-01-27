import torch
import torch.nn as nn


class NeuralNetworkPlain(nn.Module):
    """A plain neural network."""

    def __init__(
        self,
        layer_cnt: int,
        neuron_cnt: int,
        activation_func: str,
        input_cnt: int,
        output_cnt: int,
    ):
        """
        :param layer_cnt: The number of layers in the neural network.
        :param neuron_cnt: The number of neurons in each layer.
        :param activation_func: The activation function to use.
        :param input_cnt: The number of input features.
        :param output_cnt: The number of output features.
        """
        super().__init__()
        self.act_func = getattr(nn, activation_func)

        layers = [nn.Linear(input_cnt, neuron_cnt), self.act_func(inplace=True)]
        for i in range(layer_cnt - 2):
            layers.append(nn.Linear(neuron_cnt, neuron_cnt))
            layers.append(self.act_func(inplace=True))
        layers.append(nn.Linear(neuron_cnt, output_cnt))

        self.layers = nn.Sequential(*layers)
        self.layers.apply(self._weights_init)

    def _weights_init(self, layer):
        """Initialize the weights of the neural network."""
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.layers(x)


class NeuralNetworkBatchNorm(nn.Module):
    """A neural network with batch normalization layers."""

    def __init__(
        self,
        layer_cnt: int,
        neuron_cnt: int,
        activation_func: nn.Module,
        input_cnt: int,
        output_cnt: int,
    ):
        """
        :param layer_cnt: The number of layers in the neural network.
        :param neuron_cnt: The number of neurons in each layer.
        :param activation_func: The activation function to use.
        :param input_cnt: The number of input features.
        :param output_cnt: The number of output features.
        """
        super().__init__()
        self.act_func = activation_func

        layers = [nn.Linear(input_cnt, neuron_cnt), activation_func(inplace=True)]
        for i in range(layer_cnt - 2):
            layers.append(nn.BatchNorm1d(neuron_cnt))
            layers.append(nn.Linear(neuron_cnt, neuron_cnt))
            layers.append(activation_func(inplace=True))
        layers.append(nn.BatchNorm1d(neuron_cnt))
        layers.append(nn.Linear(neuron_cnt, output_cnt))

        self.layers = nn.Sequential(*layers)
        self.layers.apply(self._weights_init)

    def _weights_init(self, layer):
        """Initialize the weights of the neural network."""
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        """Forward pass of the neural network."""
        return self.layers(x)
