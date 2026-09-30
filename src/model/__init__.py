import torch
import copy
import torch.nn.functional as F

from .custom_cnn import CustomCNN, ModelFedCon
from .cnn import SimpleCNN, SimpleWNCNN
from .resnet import ResNet50_cifar10,ResNet18_cifar10
from torchvision.models import *
from torch.nn import Sequential, Linear, ReLU
from .vgg11 import VGG

# ``wonresnet.py`` is not part of the public master branch.  Keep its optional
# model entry point from preventing the CNN / ResNet-18 baselines from loading.
try:
    from .wonresnet import nf_resnet18
except ImportError:
    nf_resnet18 = None

def model_call(model_name: str, num_of_classes: int,bn = True, **kwargs):
    if model_name.lower() == 'custom_cnn':
        return CustomCNN(num_of_classes=num_of_classes)
    elif model_name.lower() == 'moon_cnn':
        return ModelFedCon(10, n_classes=num_of_classes)
    elif model_name.lower() == "resnet-50":
        _model = ResNet50_cifar10(num_classes=num_of_classes)
        return _model
    elif model_name.lower() =="vgg-16":
        return VGG(num_classes=num_of_classes)
    elif model_name.lower() == "resnet-18":
        # _model = resnet18()
        # fc = Sequential(
        #     Linear(in_features=512, out_features=256, bias=False),
        #     ReLU(inplace=True),
        #     Linear(in_features=256, out_features=num_of_classes, bias=False)
        # )
        # _model.fc = fc
        _model = ResNet18_cifar10(num_classes=num_of_classes,bn = bn)
        return _model
    elif model_name.lower() == "resnetnb-18":
        _model = resnet18()
        fc = Sequential(
            Linear(in_features=512, out_features=256, bias=False),
            ReLU(inplace=True),
            Linear(in_features=256, out_features=num_of_classes, bias=False)
        )
        _model.fc = fc
        return _model
    elif model_name.lower() == 'simple_cnn':
        return SimpleCNN(num_classes=num_of_classes, **kwargs)
    elif model_name.lower() == 'ws_cnn':
        return SimpleWNCNN(num_classes=num_of_classes, **kwargs)
    elif model_name.lower() =='ws_resnet':
        if nf_resnet18 is None:
            raise NotImplementedError("ws_resnet requires src/model/wonresnet.py, which is not included in this branch.")
        return nf_resnet18(num_classes=num_of_classes, **kwargs)
    else:
        raise NotImplementedError("Not implemented yet.")


NUMBER_OF_CLASSES = {
        'cifar-10': 10,
        'cifar-100': 100,
        'mnist': 10
}

__all__ = [
    'CustomCNN',
    'SimpleCNN',
    'model_call',
    'F',
    'NUMBER_OF_CLASSES',
    'torch',
    'copy',
]
