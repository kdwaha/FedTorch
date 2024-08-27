from src import *
from src.model import *
from torch.nn import *
import torch.nn as nn
import math

def init_weights(layer):
    if isinstance(layer, Conv2d):
        init.xavier_uniform(layer.weight)
        layer.bias.data.fill_(0.01)
    elif isinstance(layer, Linear):
        init.xavier_uniform(layer.weight)
        layer.bias.data.fill_(0.01)
#
# class Conv2d(torch.nn.Conv2d):
#
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True):
#         super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
#                  padding, dilation, groups, bias)
#
#     def forward(self, x):
#         weight = self.weight
#         weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
#                                   keepdim=True).mean(dim=3, keepdim=True)
#         weight = weight - weight_mean
#         std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
#         weight = weight / std.expand_as(weight)
#         return F.conv2d(x, weight, self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)
#
#
#
#
# class Linear(torch.nn.Module):
#     def __init__(self, in_features, out_features, bias=True):
#         super(Linear, self).__init__()
#         # 입력 특성과 출력 특성의 수를 초기화합니다.
#         self.in_features = in_features
#         self.out_features = out_features
#         # 가중치와 편향을 파라미터로 등록합니다.
#         self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_features))
#         else:
#             self.register_parameter('bias', None)  # 편향이 없는 경우입니다.
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         # 가중치를 적절히 초기화합니다. 여기서는 He 초기화 방법을 사용합니다.
#         nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             # 편향을 초기화합니다.
#             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             nn.init.uniform_(self.bias, -bound, bound)
#
#     def forward(self, x):
#         # 가중치에서 입력 특성 차원에 대한 평균을 계산합니다.
#         weight_mean = self.weight.mean(dim=1, keepdim=True)
#         # 평균을 가중치에서 빼서 가중치의 평균을 0으로 만듭니다.
#         weight = self.weight - weight_mean
#         # 표준 편차를 계산하고 1e-5를 더해 0으로 나누는 것을 방지합니다.
#         std = weight.view(self.out_features, -1).std(dim=1).view(-1, 1) + 1e-5
#         # 가중치를 표준 편차로 나누어 표준화합니다.
#         weight = weight / std.expand_as(weight)
#         # 표준화된 가중치와 입력 x를 사용하여 선형 변환을 수행합니다.
#         return F.linear(x, weight, self.bias)
#

## dyn 구현체로 확인해 볼 것//
class CustomCNN(Module):
    def __init__(self, num_of_classes: int = 10, b_global: bool = False, **kwargs):
        super(CustomCNN, self).__init__()
        if b_global:
            self.features = Sequential(
                Conv2d(3, 6, (5, 5), bias=False),
                ReLU(),
                MaxPool2d((2, 2)),
                Conv2d(6, 16 * kwargs['n_of_clients'], (5, 5), bias=False),
                ReLU(),
                MaxPool2d((2, 2))
            )
            self.fc = Sequential(
                Linear(16 * kwargs['n_of_clients'] * 5 * 5, 120, bias=False),
                ReLU(),
                Linear(120, 84, bias=False),
                ReLU(),
                Linear(84, num_of_classes, bias=False)
            )
        else:
            self.features = Sequential(
                Conv2d(3, 6, (5, 5), bias=False),# bias true
                ReLU(),
                MaxPool2d((2, 2)),
                Conv2d(6, 16, (5, 5), bias=False), # bias true
                ReLU(),
                MaxPool2d((2, 2))
            )
            self.fc = Sequential(
                Linear(16 * 5 * 5, 120, bias=False), # bias true
                ReLU(),
                Linear(120, 84, bias=False), # bias true
                ReLU(),
                Linear(84, num_of_classes, bias=False) # bias true
            )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def feature_maps(self, x):
        with torch.no_grad():
            x = self.features(x)
        return x


class MiniCustomCNN(Module):
    def __init__(self):
        super(MiniCustomCNN, self).__init__()
        self.features = Sequential(
            Conv2d(3, 6, (5, 5)),
            ReLU(),
            MaxPool2d((2, 2)),
            Conv2d(6, 8, (5, 5)),
            ReLU(),
            MaxPool2d((2, 2))
        )
        self.fc = Sequential(
            Linear(200, 40),
            ReLU(),
            Linear(40, 20),
            ReLU(),
            Linear(20, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class FeatureBlock(Module):
    def __init__(self):
        super(FeatureBlock, self).__init__()
        self.features = Sequential(
            Conv2d(3, 6, (5, 5)),
            ReLU(),
            MaxPool2d((2, 2)),
            Conv2d(6, 16, (5, 5)),
            ReLU(),
            MaxPool2d((2, 2))
        )

    def forward(self, x):
        return self.features(x)


class MixedModel(Module):
    def __init__(self, models):
        super(MixedModel, self).__init__()
        feature_block = Sequential(
            Conv2d(3, 6, (5, 5)),
            ReLU(),
            MaxPool2d((2, 2)),
            Conv2d(6, 16, (5, 5)),
            ReLU(),
            MaxPool2d((2, 2))
        )
        self.features = []

        for k, v in models.items():
            _w = OrderedDict({k: val.clone().detach().cpu() for k, val in v.features.state_dict().items()})
            _fb = copy.deepcopy(feature_block)
            _fb.load_state_dict(_w)
            _fb.requires_grad_(False)
            self.features.append(_fb)

        self.fc = Sequential(
            Linear(16 * 5 * 5 * len(models), 120 * len(models)),
            ReLU(),
            Linear(120 * len(models), 84 * len(models)),
            ReLU(),
            Linear(84 * len(models), 10 * len(models)),
            ReLU(),
            Linear(10 * len(models), 10)
        )

    def freeze_feature_layer(self):
        for feature in self.features:
            feature.requires_grad_(False)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        _out = []
        for model in self.features:
            _y = model(x)
            _y = torch.flatten(_y, start_dim=1)
            _out.append(_y)

        _out = torch.cat(_out, dim=1)
        out = self.fc(_out)
        return out


class ModelFedCon(Module):
    """
        classifier part return, source, label , representation
    """

    def __init__(self, out_dim, n_classes, net_configs=None):
        super(ModelFedCon, self).__init__()

        self.features = SimpleCNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)
        num_ftrs = 84

        self.l1 = Linear(num_ftrs, num_ftrs)
        self.l2 = Linear(num_ftrs, out_dim)

        # last layer
        self.l3 = Linear(out_dim, n_classes)

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()  # representation tensor
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)

        y = self.l3(x)
        return y


class SimpleCNN_header(Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN_header, self).__init__()
        self.conv1 = Conv2d(3, 6, 5)
        self.relu = ReLU()
        self.pool = MaxPool2d(2, 2)
        self.conv2 = Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = Linear(input_dim, hidden_dims[0])
        self.fc2 = Linear(hidden_dims[0], hidden_dims[1])

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


# class SimpleCNN(Module):
#     def __init__(self, num_classes: int = 10, **kwargs):
#         super(SimpleCNN, self).__init__()
#
#         self.features = Sequential(
#             Conv2d(3, 6, (5, 5)),
#             ReLU(),
#             MaxPool2d((2, 2)),
#             Conv2d(6, 16, (5, 5)),
#             ReLU(),
#             MaxPool2d((2, 2))
#         )
#         self.fc_1 = Linear(16 * 5 * 5, 120)
#         self.fc_2 = Linear(120, 84)
#         self.logit = Linear(84, num_classes)
#
#         self.fc_list = [self.fc_1, self.fc_2]
#
#         if 'threshold' in kwargs:
#             self.threshold = kwargs['threshold']
#         else:
#             self.threshold = 0
#
#     def forward(self, x):
#         x = self.features(x)
#         features = torch.flatten(x, 1)
#
#         for i, layer in enumerate(self.fc_list):
#             features = F.relu(layer(features))
#
#         logit = self.logit(features)
#         return logit, features
