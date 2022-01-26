# Copyright (c) SenseTime. All Rights Reserved.
import torch
import torch.nn as nn

class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=7):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            )
        self.center_size = center_size

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:
            l = (x.size(3) - self.center_size) // 2
            r = l + self.center_size
            x = x[:, :, l:r, l:r]
        return x


class AdjustAllLayer(nn.Module):
    def __init__(self, in_channels=[512, 1024, 2048], out_channels=[128, 256, 512], center_size=7):
        super(AdjustAllLayer, self).__init__()
        self.num = len(out_channels)

        for i in range(self.num):
            self.add_module('downsample'+str(i+2),
                            AdjustLayer(in_channels[i],
                                        out_channels[i],
                                        center_size))

    def forward(self, features):
        out = []
        for i in range(self.num):
            adj_layer = getattr(self, 'downsample'+str(i+2))
            out.append(adj_layer(features[i]))
        return out


if __name__ == '__main__':
    net = AdjustAllLayer()
    print(net)
    net = net.cuda()

    var1 = torch.FloatTensor(1, 512, 15, 15).cuda()
    var2 = torch.FloatTensor(1,1024, 15, 15).cuda()
    var3 = torch.FloatTensor(1,2048, 15, 15).cuda()
    var = [var1, var2, var3]
    result = net(var)
    for i in range(len(result)):
        print(result[i].shape)




