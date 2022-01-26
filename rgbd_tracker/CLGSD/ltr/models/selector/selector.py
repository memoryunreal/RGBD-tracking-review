import torch.nn as nn
from ltr import model_constructor
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

class Branch_Selector(nn.Module):
    """ BBox Predictor module"""
    def __init__(self, inplanes=64,num_branch=3):
        super(Branch_Selector, self).__init__()
        self.conv1 = conv(inplanes,inplanes//2,kernel_size=3,padding=1)
        self.conv2 = conv(inplanes//2,inplanes//4,kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Sequential(nn.Linear(1024, 512),nn.BatchNorm1d(512),nn.ReLU())
        self.fc2 = nn.Linear(512,num_branch)

    def forward(self, x):
        """ Forward pass with input x. """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
@model_constructor
def branch_selector():
    return Branch_Selector()
@model_constructor
def branch_selector_bc():
    return Branch_Selector(num_branch=2)