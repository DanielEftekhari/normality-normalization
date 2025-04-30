import itertools
import collections
import numpy as np
import torch.nn as nn

import activations
import layers as norm_layers


class FCNet(nn.Module):
    def __init__(self, dims, c_dim, units, standardize, cfg, device, *args, **kwargs):
        super(FCNet, self).__init__()

        c_in = dims[0]
        all_units = [np.prod(dims)] + list(itertools.chain.from_iterable(units))

        self.activation = getattr(activations, cfg.activation.lower())

        if cfg.norm == 'none': self.norm_layer = nn.Identity
        elif cfg.norm == 'normal': self.norm_layer = norm_layers.BatchNormalNorm1d
        elif cfg.norm == 'batch': self.norm_layer = norm_layers.BatchNorm1d

        self._aux = collections.defaultdict(dict)

        self.add_module('standardize', norm_layers.Standardize(mean=standardize[0], std=standardize[1], device=device))

        for i in range(len(all_units)-1):
            self.add_module('affine{:0=4d}'.format(i+1), nn.Linear(all_units[i], all_units[i+1], bias=True))
            self._aux['affine{:0=4d}'.format(i+1)] = {'layer_number': i+1, 'type': 'affine'}

            self.add_module('norm{:0=4d}'.format(i+1), self.norm_layer(all_units[i+1], device=device, **kwargs))
            self._aux['norm{:0=4d}'.format(i+1)] = {'layer_number': i+1, 'type': 'norm'}

            self.add_module('dropout{:0=4d}'.format(i+1), nn.Dropout(cfg.dropout))
            self._aux['dropout{:0=4d}'.format(i+1)] = {'layer_number': i+1, 'type': 'dropout'}

            self.add_module('activation{:0=4d}'.format(i+1), self.activation())
            self._aux['activation{:0=4d}'.format(i+1)] = {'layer_number': i+1, 'type': 'activation'}

        self.add_module('fc', nn.Linear(all_units[-1], c_dim, bias=True))
        self._aux['fc'] = {'layer_number': len(all_units), 'type': 'fc'}

        self._layers = {}
        for elem in self.named_modules():
            if len(elem[0]) > 0:
                self._layers[elem[0]] = elem[1]
        self._names = list(self._layers.keys())

    def forward(self, x):
        # input norm (if/when applicable)
        x = self._layers[self._names[0]](x)

        # flatten input
        x = x.view(x.shape[0], -1)

        # remaining model forward pass
        for i in range(1, len(self._names)):
            x = self._layers[self._names[i]](x)
        return x
