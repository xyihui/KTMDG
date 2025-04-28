import math
import torch
import torch.nn as nn


def Conv3x3BNReLU(in_channels,out_channels,stride,groups=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

def transform_tensor(tensor):
    mean_values = torch.mean(tensor, dim=(2, 3))
    return mean_values

def transform_tensor_variance(tensor):
    variance_values = torch.var(tensor, dim=(2, 3))
    result = variance_values.unsqueeze(1).unsqueeze(1)
    return result

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            Conv3x3BNReLU(in_channels, out_channels,stride=1),
            Conv3x3BNReLU(out_channels, out_channels, stride=1)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels,stride=2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=stride)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.pool(self.double_conv(x))


class Sencode(nn.Module):
    def __init__(self, i):
        super(Sencode, self).__init__()
        self.conv = DoubleConv(3, 32)
        self.down1 = DownConv(32, 64)
        self.down2 = DownConv(64, 128)
        self.i = i

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        if self.i == 1:
            x3 = transform_tensor(x3)
        return x3


def build_Sencode(i):
    return Sencode(i)

if __name__ =='__main__':
    model = build_Sencode(1)
    # print(model)

    input = torch.randn(3, 3, 512, 512)
    out = model(input)
    # summary(model, (3, 512, 512))
    print(out.shape)