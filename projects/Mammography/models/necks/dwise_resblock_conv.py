# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmpretrain.registry import MODELS

class Bottleneck(BaseModule):
    def __init__(self,in_channels, expansion_ratio=8):
        super(Bottleneck, self).__init__()
        self.mid_channels=int(in_channels/expansion_ratio)

        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels,self.mid_channels,kernel_size=1,bias=False),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(),
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=2, padding=1,
                      groups=self.mid_channels,bias=False),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(),
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(self.mid_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.relu=nn.ReLU()
        self.downsamdple=nn.AvgPool2d(2,stride=2,ceil_mode=True,count_include_pad=False)
        self.dropout=nn.Dropout(0.2)


    def forward(self,x):
        out=self.dropout(self.conv3(self.conv2(self.conv1(x))))
        identity=self.downsamdple(x)
        out+=identity
        out=self.relu(out)
        return out

@MODELS.register_module()
class DwiseConvResBlockPooling(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_block=2,
                 expansion_ratio=8,
                 frozen=False):
        super(DwiseConvResBlockPooling, self).__init__()

        self.globalpooling = nn.AdaptiveAvgPool2d((1, 1))
        res_blocks=[]
        for i in range(num_block):
            res_blocks.append(Bottleneck(in_channels,expansion_ratio))
        self.res_blocks=nn.Sequential(*res_blocks)
        self.fc_out=False
        if in_channels!=out_channels:
            # self.hidden=nn.Sequential(nn.Linear(in_channels,21),
            #                           nn.ReLU(),
            #                           nn.Dropout(0.2))
            self.fc=nn.Linear(in_channels,out_channels)
            self.fc_out=True
        if frozen:
            for m in [self.globalpooling,self.res_blocks,self.hidden,self.fc]:
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([self.globalpooling(self.res_blocks(x)) for x in inputs])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
            if self.fc_out:
                # outs=tuple([self.fc(self.hidden(out)) for out in outs])
                outs=tuple([self.fc(out) for out in outs])
        elif isinstance(inputs, torch.Tensor):
            outs = self.globalpooling(self.res_blocks(inputs))
            outs = outs.view(inputs.size(0), -1)
            if self.fc_out:
                outs = self.fc(outs)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs
