import torch.nn as nn
import math


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidualtransposed(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidualtransposed, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        if stride == 1:
            if expand_ratio == 1:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
        else:
            if expand_ratio == 1:
                self.conv = nn.Sequential(
                    # dw
                   # nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
                    nn.ConvTranspose2d(hidden_dim, hidden_dim, 3, padding=1, stride=2, output_padding=1,groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # dw
                    nn.ConvTranspose2d(hidden_dim, hidden_dim, 3, padding=1, stride=2, output_padding=1, groups=hidden_dim, bias=False),
                    #nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, Normelise=True):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.Normelise=Normelise
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        if self.Normelise:
            if expand_ratio == 1:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
        else:
            if expand_ratio == 1:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )
            else:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        blockRestore=InvertedResidualtransposed
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        interverted_residual_settingRestore=[
            [6, 24, 2, 2, 16],
            [1, 16, 1, 1, 32],
        ]
        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        self.features2=[]
        self.reluu6= nn.ReLU6()
        self.NormeliseLayer= nn.BatchNorm2d(3)
        # building inverted residual blocks
        part=3
        ind=0
        self.SplitLayer = block(24, 48, 1, expand_ratio=6,Normelise=False)
        self.ExpandLayer = conv_bn(3, 24, 1)
        addrestorelayer=False
        if addrestorelayer:
            self.RestoreLayer=[conv_bn(3, 24, 1)]
            input_channelRes = 24
            for t, c, n, s, o in interverted_residual_settingRestore:
                output_channel = int(c * width_mult)
                for i in range(n):
                    if i == n-1:
                        self.RestoreLayer.append(blockRestore(input_channelRes,o,s, expand_ratio=t))
                        input_channelRes = o
                    else:
                        self.RestoreLayer.append(blockRestore(input_channelRes,output_channel,1,expand_ratio=t))
                        input_channelRes=output_channel
            self.RestoreLayer.append(nn.ConvTranspose2d(input_channelRes, 3, 3, padding=1, stride=2, output_padding =1))
        else:
            self.RestoreLayer=[]#[nn.ConvTranspose2d(3, 3, 3, padding=1, stride=2, output_padding =1)]
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            ind+=1
            if ind<part:
                for i in range(n):
                    if i == 0:
                        self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                    else:
                        self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                    input_channel = output_channel
            else:
                for i in range(n):
                    if i == 0:
                        self.features2.append(block(input_channel, output_channel, s, expand_ratio=t))
                    else:
                        self.features2.append(block(input_channel, output_channel, 1, expand_ratio=t))
                    input_channel = output_channel
        # building last several layers
        self.features2.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        self.features2 = nn.Sequential(*self.features2)
        if addrestorelayer:
            self.RestoreLayer = nn.Sequential(*self.RestoreLayer)
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x, stam):
        if stam==0:
            return x
        if stam == 1:
            x = self.features(x)
            x = self.features2(x)
            x = x.mean(3).mean(2)
            x = self.classifier(x)
            return x
        if stam == 2:
            x = self.features(x)
            x = self.SplitLayer(x)

            x = self.ExpandLayer(x)
            x = self.reluu6(x)
            x = self.features2(x)
            x = x.mean(3).mean(2)
            x = self.classifier(x)
            return x
        if stam == 3:
            x = self.features(x)
            x = self.SplitLayer(x)
            return x
        if stam == 4:
            x = self.NormeliseLayer(x)
            x = self.ExpandLayer(x)
            x = self.reluu6(x)
            x = self.features2(x)
            x = x.mean(3).mean(2)
            x = self.classifier(x)
            return x
        if stam == 5:
            x = self.NormeliseLayer(x)
            x = self.RestoreLayer(x)
            return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
