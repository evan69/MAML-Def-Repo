# task_net.py by hyf
from collections import OrderedDict

import torch
import torch.nn.functional as F

from maml.models.model import Model

def weight_init(module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0, std=0.01)
        module.bias.data.zero_()


class TaskNet(Model):
    def __init__(self, input_size, output_size, hidden_sizes=(),
                 nonlinearity=F.relu, disable_norm=False,
                 bias_transformation_size=0):
        super(TaskNet, self).__init__()
        hidden_sizes = list(hidden_sizes)
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.num_layers = len(hidden_sizes) + 1
        self.disable_norm = disable_norm
        self.bias_transformation_size = bias_transformation_size

        if bias_transformation_size > 0:
            input_size = input_size + bias_transformation_size
            self.bias_transformation = torch.nn.Parameter(
                torch.zeros(bias_transformation_size))

        # print ([input_size], hidden_sizes, [output_size])
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(1, self.num_layers):
            self.add_module(
                'layer{0}_linear'.format(i),
                torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            if not self.disable_norm:
                self.add_module(
                    'layer{0}_bn'.format(i),
                    torch.nn.BatchNorm1d(layer_sizes[i], momentum=0.001))
        self.add_module(
            'output_linear',
            torch.nn.Linear(layer_sizes[self.num_layers - 1],
                            layer_sizes[self.num_layers]))
        self.add_module(
            'output_softmax',
            torch.nn.Softmax())
        self.apply(weight_init)

    def forward(self, task_emb, params=None, training=True):
        if params is None:
            params = OrderedDict(self.named_parameters())
        x = task_emb.view(task_emb.size(0), -1)

        if self.bias_transformation_size > 0:
            x = torch.cat((x, params['bias_transformation'].expand(
                x.size(0), params['bias_transformation'].size(0))), dim=1)

        for key, module in self.named_modules():
            if 'linear' in key:
                x = F.linear(x, weight=params[key + '.weight'],
                             bias=params[key + '.bias'])
                if self.disable_norm and 'output' not in key:
                    x = self.nonlinearity(x)
            if 'bn' in key:
                x = F.batch_norm(x, weight=params[key + '.weight'],
                                 bias=params[key + '.bias'],
                                 running_mean=module.running_mean,
                                 running_var=module.running_var,
                                 training=training)
                x = self.nonlinearity(x)
            if 'softmax' in key:
                x = F.softmax(x)
        return x

