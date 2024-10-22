from collections import OrderedDict

import torch
import torch.nn.functional as F

from maml.models.model import Model


def weight_init(module):
    if (isinstance(module, torch.nn.Linear)
        or isinstance(module, torch.nn.Conv2d)):
        torch.nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()


class ConvModel(Model):
    """
    NOTE: difference to tf implementation: batch norm scaling is enabled here
    TODO: enable 'non-transductive' setting as per
          https://arxiv.org/abs/1803.02999
    """
    def __init__(self, input_channels, output_size, num_channels=64,
                 kernel_size=3, padding=1, nonlinearity=F.relu,
                 use_max_pool=False, img_side_len=28, verbose=False):
        super(ConvModel, self).__init__()
        self._input_channels = input_channels
        self._output_size = output_size
        self._num_channels = num_channels
        self._kernel_size = kernel_size
        self._nonlinearity = nonlinearity
        self._use_max_pool = use_max_pool
        self._padding = padding
        self._bn_affine = False
        self._reuse = False
        self._verbose = verbose

        self.update_tmp_params(None)

        if self._use_max_pool:
            self._conv_stride = 1
            self._features_size = 1
            self.features = torch.nn.Sequential(OrderedDict([
                ('layer1_conv', torch.nn.Conv2d(self._input_channels,
                                                self._num_channels,
                                                self._kernel_size,
                                                stride=self._conv_stride,
                                                padding=self._padding)),
                ('layer1_bn', torch.nn.BatchNorm2d(self._num_channels,
                                                   affine=self._bn_affine,
                                                   momentum=0.001)),
                ('layer1_max_pool', torch.nn.MaxPool2d(kernel_size=2,
                                                       stride=2)),
                ('layer1_relu', torch.nn.ReLU(inplace=True)),
                ('layer2_conv', torch.nn.Conv2d(self._num_channels,
                                                self._num_channels*2,
                                                self._kernel_size,
                                                stride=self._conv_stride,
                                                padding=self._padding)),
                ('layer2_bn', torch.nn.BatchNorm2d(self._num_channels*2,
                                                   affine=self._bn_affine,
                                                   momentum=0.001)),
                ('layer2_max_pool', torch.nn.MaxPool2d(kernel_size=2,
                                                       stride=2)),
                ('layer2_relu', torch.nn.ReLU(inplace=True)),
                ('layer3_conv', torch.nn.Conv2d(self._num_channels*2,
                                                self._num_channels*4,
                                                self._kernel_size,
                                                stride=self._conv_stride,
                                                padding=self._padding)),
                ('layer3_bn', torch.nn.BatchNorm2d(self._num_channels*4,
                                                   affine=self._bn_affine,
                                                   momentum=0.001)),
                ('layer3_max_pool', torch.nn.MaxPool2d(kernel_size=2,
                                                       stride=2)),
                ('layer3_relu', torch.nn.ReLU(inplace=True)),
                ('layer4_conv', torch.nn.Conv2d(self._num_channels*4,
                                                self._num_channels*8,
                                                self._kernel_size,
                                                stride=self._conv_stride,
                                                padding=self._padding)),
                ('layer4_bn', torch.nn.BatchNorm2d(self._num_channels*8,
                                                   affine=self._bn_affine,
                                                   momentum=0.001)),
                ('layer4_max_pool', torch.nn.MaxPool2d(kernel_size=2,
                                                       stride=2)),
                ('layer4_relu', torch.nn.ReLU(inplace=True)),
            ]))
        else:
            self._conv_stride = 2
            self._features_size = (img_side_len // 14)**2
            self.features = torch.nn.Sequential(OrderedDict([
                ('layer1_conv', torch.nn.Conv2d(self._input_channels,
                                                self._num_channels,
                                                self._kernel_size,
                                                stride=self._conv_stride,
                                                padding=self._padding)),
                ('layer1_bn', torch.nn.BatchNorm2d(self._num_channels,
                                                   affine=self._bn_affine,
                                                   momentum=0.001)),
                ('layer1_relu', torch.nn.ReLU(inplace=True)),
                ('layer2_conv', torch.nn.Conv2d(self._num_channels,
                                                self._num_channels*2,
                                                self._kernel_size,
                                                stride=self._conv_stride,
                                                padding=self._padding)),
                ('layer2_bn', torch.nn.BatchNorm2d(self._num_channels*2,
                                                   affine=self._bn_affine,
                                                   momentum=0.001)),
                ('layer2_relu', torch.nn.ReLU(inplace=True)),
                ('layer3_conv', torch.nn.Conv2d(self._num_channels*2,
                                                self._num_channels*4,
                                                self._kernel_size,
                                                stride=self._conv_stride,
                                                padding=self._padding)),
                ('layer3_bn', torch.nn.BatchNorm2d(self._num_channels*4,
                                                   affine=self._bn_affine,
                                                   momentum=0.001)),
                ('layer3_relu', torch.nn.ReLU(inplace=True)),
                ('layer4_conv', torch.nn.Conv2d(self._num_channels*4,
                                                self._num_channels*8,
                                                self._kernel_size,
                                                stride=self._conv_stride,
                                                padding=self._padding)),
                ('layer4_bn', torch.nn.BatchNorm2d(self._num_channels*8,
                                                   affine=self._bn_affine,
                                                   momentum=0.001)),
                ('layer4_relu', torch.nn.ReLU(inplace=True)),
            ]))

        self.classifier = torch.nn.Sequential(OrderedDict([
            ('fully_connected', torch.nn.Linear(self._num_channels*8,
                                                self._output_size))
        ]))
        self.apply(weight_init)


    def update_tmp_params(self, params):
        self._tmp_params = params


    def forward(self, task, params=None, embeddings=None):
        return self.forward_single(task.x, params, embeddings)


    def forward_single(self, x, params=None, embeddings=None):
        if not self._reuse and self._verbose: print('='*10 + ' Model ' + '='*10)
        if params is None:
            params = OrderedDict(self.named_parameters())
        if self._tmp_params != None:
            params = self._tmp_params

        # x = task.x
        if not self._reuse and self._verbose: print('input size: {}'.format(x.size()))
        for layer_name, layer in self.features.named_children():
            weight = params.get('features.' + layer_name + '.weight', None)
            bias = params.get('features.' + layer_name + '.bias', None)
            if 'conv' in layer_name:
                x = F.conv2d(x, weight=weight, bias=bias,
                             stride=self._conv_stride, padding=self._padding)
            elif 'bn' in layer_name:
                x = F.batch_norm(x, weight=weight, bias=bias,
                                 running_mean=layer.running_mean,
                                 running_var=layer.running_var,
                                 training=True)
            elif 'max_pool' in layer_name:
                x = F.max_pool2d(x, kernel_size=2, stride=2)
            elif 'relu' in layer_name:
                x = F.relu(x)
            elif 'fully_connected' in layer_name:
                break
            else:
                raise ValueError('Unrecognized layer {}'.format(layer_name))
            if not self._reuse and self._verbose: print('{}: {}'.format(layer_name, x.size()))

        # in maml network the conv maps are average pooled
        x = x.view(x.size(0), self._num_channels*8, self._features_size)
        if not self._reuse and self._verbose: print('reshape to: {}'.format(x.size()))
        x = torch.mean(x, dim=2)
        if not self._reuse and self._verbose: print('reduce mean: {}'.format(x.size()))
        logits = F.linear(
            x, weight=params['classifier.fully_connected.weight'],
            bias=params['classifier.fully_connected.bias'])
        if not self._reuse and self._verbose: print('logits size: {}'.format(logits.size()))
        if not self._reuse and self._verbose: print('='*27)
        self._reuse = True
        return logits
