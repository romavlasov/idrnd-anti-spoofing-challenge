import torch
import torch.nn as nn


class SELayer(nn.Module):
    def __init__(self, inplanes, squeeze_ratio=8, activation=nn.PReLU, size=None):
        super(SELayer, self).__init__()
        if size is not None:
            self.global_avgpool = nn.AvgPool2d(size)
        else:
            self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(inplanes, int(inplanes / squeeze_ratio), kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(int(inplanes / squeeze_ratio), inplanes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)
        return x * out
    
    
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, outp_size=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        self.inv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channels * expand_ratio),
            nn.ReLU(),

            nn.Conv2d(in_channels * expand_ratio, in_channels * expand_ratio, 3, stride, 1,
                      groups=in_channels * expand_ratio, bias=False),
            nn.BatchNorm2d(in_channels * expand_ratio),
            nn.ReLU(),

            nn.Conv2d(in_channels * expand_ratio, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            SELayer(out_channels, 8, nn.ReLU, outp_size)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.inv_block(x)

        return self.inv_block(x)
    
    
def build_layers(in_channel):
    setting = [
        # t, c, n, s
        [2, in_channel, 2, 2],
        [2, in_channel, 2, 2],
    ]
    layers = []
    for t, c, n, s in setting:
        out_channel = c
        for i in range(n):
            if i == 0:
                layers.append(InvertedResidual(in_channel, out_channel, s, t))
            else:
                layers.append(InvertedResidual(in_channel, out_channel, 1, t))
            in_channel = out_channel
            
    return nn.Sequential(*layers)