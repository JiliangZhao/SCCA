#_*_coding:utf8_*_
#_*_coding:utf8_*_
#_*_coding:utf8_*_
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter

class fusion_layer(nn.Module):
    def __init__(self, kernel_size = 3, k_size = 3):
        super(fusion_layer, self).__init__()
        ###sa
        self.conv1_2 = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size-1) //  2, bias=False)
        self.sigmoid1 = nn.Sigmoid()
        ###ca
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1_1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid2 = nn.Sigmoid()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
    def forward(self, x1, x2 ):
        ###sa
        avg_out = torch.mean(x2, dim=1, keepdim=True)
        #print(avg_out.size())
        max_out, _ = torch.max(x2, dim=1, keepdim=True)
        y2 = torch.cat([avg_out, max_out], dim=1)
        y1_2 = self.conv1_2(y2)
        #print(y.size())
        y2 = self.sigmoid2(y1_2)
        ###ca
        y1 = self.avg_pool(x1)
        # Two different branches of ECA module
        y1 = self.conv1_1(y1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y1 = self.sigmoid1(y1)
        y1_2 = self.maxpool1(y1_2)
        y1_2 = self.sigmoid1(y1_2)
        #print((x * y.expand_as(x)).size())
        x2 = x2 * y2.expand_as(x2)
        x1 = (x1 * y1.expand_as(x1)) + (x1 * y1_2.expand_as(x1))
        return x1, x2

class ca_layer(nn.Module):
    def __init__(self, k_size=3):
        super(ca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class sa_layer(nn.Module):
    def __init__(self, k_size=3):
        super(sa_layer, self).__init__()
        #assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        #padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size = k_size, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class SCCANet(nn.Module):
    def __init__(self, num_classes=12):
        super(SCCANet,self).__init__()
        ###  input:   ms = [32,32,4]  pan = [128,128,1]
        ###  prejust
        ###  ms
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(4, 6, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
        )
        ###  pan
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
        #self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ### output:   ms = [32,32,32]  pan = [64,64,8]


        ###  input:   ms = [32,32,32]  pan = [64,64,8]
        ###  layers1  two_blocks
        ###  ms_branch_layer1
        self.layers1_1_1 = nn.Sequential(
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(6),
        )
        self.ca1_1 = ca_layer(k_size=3)
        self.layers1_1_2 = nn.Sequential(
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(6),
        )
        self.ca1_2 = ca_layer(k_size=3)


        ###  pan_branch_layer1
        self.layers1_2_1 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(3),
        )
        self.sa1_1 = sa_layer(k_size=3)
        self.layers1_2_2 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(3),
        )
        self.sa1_2 = sa_layer(k_size=3)
        #self.fusion1 = fusion_layer(kernel_size=5,k_size=3)
        ###  output:  ms = [32,32,32]  pan = [64,64,8]

        ###  input:   ms = [32,32,32]  pan = [64,64,8]
        ###  layers2  two_blocks

        ###  ms_branch_layer2
        self.layers2_1_1 = nn.Sequential(
            nn.Conv2d(6, 12, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(12),
        )
        self.ca2_1 = ca_layer(k_size=3)
        self.downsample2_1 = nn.Sequential(
            nn.Conv2d(6, 12, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(12),
        )
        self.layers2_1_2 = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 12,  kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(12),
        )
        self.ca2_2 = ca_layer(k_size=3)

        ###  pan_branch_layer2
        self.layers2_2_1 = nn.Sequential(
            nn.Conv2d(3, 6,  kernel_size=5, stride=2, padding=2,bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 6, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(6),
        )
        self.sa2_1 = sa_layer(k_size=3)
        self.downsample2_2 = nn.Sequential(
            nn.Conv2d(3, 6,  kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(6),
        )
        self.layers2_2_2 = nn.Sequential(
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(6),
        )
        self.sa2_2 = sa_layer(k_size=3)
        ###  output:  ms = [16,16,64]  pan = [32,32,16]


        ###  input:   ms = [16,16,64]  pan = [32,32,16]
        ###  layers2  two_blocks
        ###  ms_branch_layer3
        self.layers3_1_1 = nn.Sequential(
            nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
        )
        self.ca3_1 = ca_layer(k_size=3)
        self.downsample3_1 = nn.Sequential(
            nn.Conv2d(12, 24, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(24),
        )
        self.layers3_1_2 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
        )
        self.ca3_2 = ca_layer(k_size=3)


        ###  pan_branch_layer1
        self.layers3_2_1 = nn.Sequential(
            nn.Conv2d(6, 12, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(12),
        )
        self.sa3_1 = sa_layer(k_size=3)
        self.downsample3_2 = nn.Sequential(
            nn.Conv2d(6, 12, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(12),
        )
        self.layers3_2_2 = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(12),
        )
        self.sa3_2 = sa_layer(k_size=3)


        ###   output: ms = [8,8,128]  pan = [16,16,32]
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        ###  ms = [8,8,128]  pan = [8,8,32]
        ###  fus = [8,8,160]
        ###  input: [8,8,160]
        ###  fuslayers4  two_blocks
        self.layers4_1 = nn.Sequential(
            nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(36),
            nn.ReLU(inplace=True),
            nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(36),
        )
        self.ca4_1 = ca_layer(k_size=3)
        self.layers4_2 = nn.Sequential(
            nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(36),
            nn.ReLU(inplace=True),
            nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(36),
        )
        self.ca4_2 = ca_layer(k_size=3)
        ###  output: [8,8,160]


        ###  input: [8,8,160]
        ###  fuslayers5  two_blocks
        self.layers5_1 = nn.Sequential(
            nn.Conv2d(36, 72, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(72),
            nn.ReLU(inplace=True),
            nn.Conv2d(72, 72, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(72),
        )
        self.ca5_1 = ca_layer(k_size=3)
        self.downsample5 = nn.Sequential(
            nn.Conv2d(36, 72, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(72),
        )
        self.layers5_2 = nn.Sequential(
            nn.Conv2d(72, 72, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(72),
            nn.ReLU(inplace=True),
            nn.Conv2d(72, 72, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(72),
        )
        self.ca5_2 = ca_layer(k_size=3)
        ###  output: [4,4,320]

        ###  input: [4,4,320]
        ###  fuslayers6  fully connect
        self.avgpool = nn.AvgPool2d(4, stride=1)
        ###  output: [1,1,320]

        ###  input: [1,1,320]
        self.fc = nn.Sequential(nn.Linear(72 , 36),
                                nn.Linear(36, 18),
                                nn.Linear(18, num_classes))


    def forward(self, x1, x2):
        #####  prejust
        out1 = self.conv1_1(x1)
        out2 = self.conv1_2(x2)
        #####  phase1
        ### block1
        residual1 = out1
        residual2 = out2
        out1 = self.layers1_1_1(out1)
        out2 = self.layers1_2_1(out2)
        out1 += residual1
        out2 += residual2
        ### block2
        residual1 = out1
        residual2 = out2
        out1 = self.layers1_1_2(out1)
        out2 = self.layers1_2_2(out2)
        out1 += residual1
        out2 += residual2
        #####  plase2
        ### block1
        residual1 = out1
        residual2 = out2
        out1 = self.layers2_1_1(out1)
        out2 = self.layers2_2_1(out2)
        residual1 = self.downsample2_1(residual1)
        residual2 = self.downsample2_2(residual2)
        out1 += residual1
        out2 += residual2
        ### block2
        residual1 = out1
        residual2 = out2
        out1 = self.layers2_1_2(out1)
        out2 = self.layers2_2_2(out2)
        out1 += residual1
        out2 += residual2
        #####  plase3
        ### block1
        residual1 = out1
        residual2 = out2
        out1 = self.layers3_1_1(out1)
        out2 = self.layers3_2_1(out2)
        residual1 = self.downsample3_1(residual1)
        residual2 = self.downsample3_2(residual2)
        out1 += residual1
        out2 += residual2
        ### block2
        residual1 = out1
        residual2 = out2
        out1 = self.layers3_1_2(out1)
        out2 = self.layers3_2_2(out2)
        out1 += residual1
        out2 += residual2
        #####  conc
        out2 = self.maxpool(out2)
        out = torch.cat([out1, out2], dim=1)
        #####  plase4
        ###block1
        residual = out
        out = self.layers4_1(out)
        out += residual
        ###block2
        residual = out
        out = self.layers4_2(out)
        out += residual
        #####  plase5
        ###block1
        residual = out
        out = self.layers5_1(out)
        residual = self.downsample5(residual)
        out += residual
        ###block2
        residual = out
        out = self.layers5_2(out)
        out += residual
        ##### fully connect
        ##### phase6
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

scca = SCCANet()
print(scca)