import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)
    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))

class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
    def forward(self, x):
        return self.bn(self.conv(x))

class Depth_Wise(Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
    def forward(self, x):
        if self.residual:
            short_cut = x
            x = self.project(self.conv_dw(self.conv(x)))
            return short_cut + x
        else:
            return self.project(self.conv_dw(self.conv(x)))

class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)
    def forward(self, x):
        return self.model(x)

# 核心结构：匹配官方权重的 input_layer, body 和 output_layer
class MobileFaceNet(Module):
    def __init__(self, embedding_size=512):
        super(MobileFaceNet, self).__init__()
        
        self.input_layer = Sequential(Conv_block(3, 64, (3, 3), 2, 1),
                                      Conv_block(64, 64, (3, 3), 1, 1, 64))
        
        self.body = Sequential(
            Depth_Wise(64, 64, False, (3, 3), 2, 1, 128),
            Residual(64, 4, 128, (3, 3), 1, 1),
            Depth_Wise(64, 128, False, (3, 3), 2, 1, 256),
            Residual(128, 6, 256, (3, 3), 1, 1),
            Depth_Wise(128, 128, False, (3, 3), 2, 1, 512),
            Residual(128, 2, 256, (3, 3), 1, 1)
        )
        
        self.output_layer = Sequential(
            Conv_block(128, 512, (1, 1), 1, 0),
            Linear_block(512, 512, (7, 7), 1, 0, 512),
            Flatten(),
            Linear(512, embedding_size, bias=False),
            BatchNorm1d(embedding_size)
        )

    def forward(self, x):
        out = self.input_layer(x)
        out = self.body(out)
        out = self.output_layer(out)
        return out