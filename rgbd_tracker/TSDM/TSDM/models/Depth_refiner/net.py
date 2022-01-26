# -*- coding: utf-8 -*-
# @Author: Zhao Pengyao
# @Date:   2019-12-31 12:31:24
import torch.nn as nn
import torch.nn.functional as F
import torch
class FuseNet(nn.Module):
    def __init__(self):
        super(FuseNet, self).__init__()

        #branch1
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=2),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            )#out:[1,96,22,22]
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            )#out:[1,256,8,8]
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            )#out:[1,384,6,6]
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            )#out:[1,384,4,4]

        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            )#out:[1,384,2,2]

        #branch2
        self.layer1d = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=2),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            )#out:[1,96,22,22]
        self.layer2d = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            )#out:[1,256,8,8]
        self.layer3d = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            )#out:[1,384,6,6]
        self.layer4d = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            )#out:[1,384,4,4]

        self.layer5d = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            )#out:[1,384,2,2]

        #pool
        self.poolNet = nn.Sequential(
            nn.AvgPool2d(kernel_size=4,stride=4)
        )

        #fuse
        self.secondNet = nn.Sequential(
            nn.Conv2d(256*4, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        #regress
        self.denseNet = nn.Sequential(
            nn.Linear(128*2*2, 16*2*2),
            nn.Sigmoid(),
            nn.Linear(16*2*2, 4),
        )

    def forward(self, rgb, ddd):
        result1 = self.layer1(rgb)
        result2 = self.layer1d(ddd)
        result1 = self.layer2(result1)
        result2 = self.layer2d(result2)

        left1 = self.poolNet(result1)
        left2 = self.poolNet(result2)

        right1 = self.layer3(result1)
        right2 = self.layer3d(result2)
        right1 = self.layer4(right1)
        right2 = self.layer4d(right2)
        right1 = self.layer5(right1)
        right2 = self.layer5d(right2)

        result = torch.cat((left1, left2, right1, right2), 1)
        result = self.secondNet(result)
        result = result.view(1,-1)

        result = self.denseNet(result)
        result = result.squeeze()

        return result

if __name__ == '__main__':
    model = FuseNet()
    rgb = torch.ones((1, 3, 100, 100))
    d3  = torch.ones((1, 3, 100, 100))
    y = model(rgb, d3)
    print(y.shape) 

