import re
import torch
import torch.nn as nn

from torch.utils import model_zoo

from models.backbones.mobilenet import MobileNet

from models.backbones.resnet import ResNet
from models.backbones.resnet import BasicBlock
from models.backbones.resnet import Bottleneck

from models.backbones.senet import SENet
from models.backbones.senet import SEBottleneck
from models.backbones.senet import SEResNetBottleneck
from models.backbones.senet import SEResNeXtBottleneck

from models.backbones.densenet import DenseNet

from models.blocks import build_layers


def mobilenet(device='cpu', *argv, **kwargs):
    model = MobileNet(*argv, **kwargs)
    return model.to(device)


def resnet18(device='cpu', out_features=1, pretrained=False, *argv, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth'))
        
    model.fc = nn.Linear(model.fc.in_features, out_features)
    return model.to(device)


def resnet34(device='cpu', out_features=1, pretrained=False, *argv, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet34-333f7ec4.pth'))

    model.fc = nn.Linear(model.fc.in_features, out_features)
    return model.to(device)


def resnet50(device='cpu', out_features=1, pretrained=False, *argv, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))

    model.fc = nn.Linear(model.fc.in_features, out_features)
    return model.to(device)


def resnet101(device='cpu', out_features=1, pretrained=False, *argv, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'))

    model.fc = nn.Linear(model.fc.in_features, out_features)
    return model.to(device)


def resnet152(device='cpu', out_features=1, pretrained=False, *argv, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet152-b121ed2d.pth'))

    model.fc = nn.Linear(model.fc.in_features, out_features)
    return model.to(device)


def resnext50(device='cpu', out_features=1, pretrained=False, *argv, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth'))

    model.last_linear = nn.Linear(model.last_linear.in_features, out_features)
    return model.to(device)


def resnext101(device='cpu', out_features=1, pretrained=False, *argv, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth'))

    model.last_linear = nn.Linear(model.last_linear.in_features, out_features)
    return model.to(device)


def senet154(device='cpu', *argv, **kwargs):
    model = SENet(SEBottleneck, [3, 8, 36, 3], groups=64, reduction=16,
                  **kwargs)
    return model.to(device)


def se_resnet50(device='cpu', out_features=1, pretrained=False, *argv, **kwargs):
    model = SENet(SEResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16,
                  inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth'))

    model.last_linear = nn.Linear(model.last_linear.in_features, out_features)
    return model.to(device)


def se_resnet101(device='cpu', out_features=1, pretrained=False, *argv, **kwargs):
    model = SENet(SEResNetBottleneck, [3, 4, 23, 3], groups=1, reduction=16,
                  inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  **kwargs)
    return model.to(device)


def se_resnet152(device='cpu', *argv, **kwargs):
    model = SENet(SEResNetBottleneck, [3, 8, 36, 3], groups=1, reduction=16,
                  inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  **kwargs)
    return model.to(device)


def se_resnext50(device='cpu', out_features=1, pretrained=False, *argv, **kwargs):
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth'))
    
    model.last_linear = nn.Linear(model.last_linear.in_features, out_features)
    return model.to(device)


def se_resnext101(device='cpu', out_features=1, pretrained=False, *argv, **kwargs):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16,
                  inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth'))
        
    model.last_linear = nn.Linear(model.last_linear.in_features, out_features)
    return model.to(device)


def densenet121(device='cpu', out_features=1, pretrained=False, *argv, **kwargs):
    model = DenseNet(32, (6, 12, 24, 16), 64, **kwargs)
    if pretrained:
        _load_densenet(model, 'https://download.pytorch.org/models/densenet121-a639ec97.pth')
        
    #model.features.add_module('final', build_layers(1024))
    model.classifier = nn.Linear(model.classifier.in_features, out_features)
    return model.to(device)


def densenet201(device='cpu', out_features=1, pretrained=False, *argv, **kwargs):
    model = DenseNet(32, (6, 12, 48, 32), 64, **kwargs)
    if pretrained:
        _load_densenet(model, 'https://download.pytorch.org/models/densenet201-c1103571.pth')
        
    model.classifier = nn.Linear(model.classifier.in_features, out_features)
    return model.to(device)


def _load_densenet(model, model_url):
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = model_zoo.load_url(model_url)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)