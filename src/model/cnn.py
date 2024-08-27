from src.model import *
from torch.nn import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Conv2d(torch.nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)




class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        # 입력 특성과 출력 특성의 수를 초기화합니다.
        self.in_features = in_features
        self.out_features = out_features
        # 가중치와 편향을 파라미터로 등록합니다.
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)  # 편향이 없는 경우입니다.
        self.reset_parameters()

    def reset_parameters(self):
        # 가중치를 적절히 초기화합니다. 여기서는 He 초기화 방법을 사용합니다.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            # 편향을 초기화합니다.
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # 가중치에서 입력 특성 차원에 대한 평균을 계산합니다.
        weight_mean = self.weight.mean(dim=1, keepdim=True)
        # 평균을 가중치에서 빼서 가중치의 평균을 0으로 만듭니다.
        weight = self.weight - weight_mean
        # 표준 편차를 계산하고 1e-5를 더해 0으로 나누는 것을 방지합니다.
        std = weight.view(self.out_features, -1).std(dim=1).view(-1, 1) + 1e-5
        # 가중치를 표준 편차로 나누어 표준화합니다.
        weight = weight / std.expand_as(weight)
        # 표준화된 가중치와 입력 x를 사용하여 선형 변환을 수행합니다.
        return F.linear(x, weight, self.bias)

class WSConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(WSConv, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias, padding_mode)
        nn.init.xavier_normal_(self.weight)
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        _eps = torch.tensor(1e-4, requires_grad=False)
        _fan_in = torch.tensor(self.weight.shape[1:].numel(), requires_grad=False).type_as(self.weight)
        self.register_buffer('eps', _eps, persistent=False)
        self.register_buffer('fan_in', _fan_in, persistent=False)

    def standardized_weights(self):
        mean = torch.mean(self.weight, axis=[1,2,3], keepdims=True)
        var = torch.var(self.weight, axis=[1,2,3], keepdims=True)
        scale = torch.rsqrt(torch.maximum(var * self.fan_in, self.eps))
        return (self.weight - mean) * scale * self.gain

    def forward(self, x):
        return F.conv2d(
            input=x,
            weight=self.standardized_weights(),
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )


class SimpleCNN(Module):
    def __init__(self, num_classes: int = 10, **kwargs):
        super(SimpleCNN, self).__init__()

        self.features = Sequential(
            Conv2d(3, 6, (5, 5)),
            ReLU(),
            MaxPool2d((2, 2)),
            Conv2d(6, 16, (5, 5)),
            ReLU(),
            MaxPool2d((2, 2))
        )
        self.fc_1 = Linear(16 * 5 * 5, 120)
        self.fc_2 = Linear(120, 84)
        self.logit = Linear(84, num_classes)

        self.fc_list = [self.fc_1, self.fc_2]

        if 'features' in kwargs:
            self.output_feature_map = kwargs['features']
        else:
            self.output_feature_map = False

    def forward(self, x):
        x = self.features(x)
        features = torch.flatten(x, 1)

        for i, layer in enumerate(self.fc_list):
            features = F.relu(layer(features), inplace=True)

        logit = self.logit(features)

        if self.output_feature_map:
            return logit, features
        else:
            return logit

class SimpleWNCNN(Module):
    def __init__(self, num_classes: int = 10, **kwargs):
        super(SimpleWNCNN, self).__init__()

        self.features = Sequential(
            WSConv(3, 6, (5, 5), bias=False),
            ReLU(),
            MaxPool2d((2, 2)),
            WSConv(6, 16, (5, 5), bias=False),
            ReLU(),
            MaxPool2d((2, 2))
        )
        self.fc_1 = Linear(16 * 5 * 5, 120, bias=False)
        self.fc_2 = Linear(120, 84, bias=False)
        self.logit = Linear(84, num_classes, bias=False)

        self.fc_list = [self.fc_1, self.fc_2]

        if 'features' in kwargs:
            self.output_feature_map = kwargs['features']
        else:
            self.output_feature_map = False

    def forward(self, x):
        x = self.features(x)
        features = torch.flatten(x, 1)

        for i, layer in enumerate(self.fc_list):
            features = F.relu(layer(features), inplace=True)

        logit = self.logit(features)

        if self.output_feature_map:
            return logit, features
        else:
            return logit
