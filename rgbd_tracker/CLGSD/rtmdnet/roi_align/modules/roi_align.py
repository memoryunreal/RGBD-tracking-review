from torch.nn.modules.module import Module
from torch.nn.functional import avg_pool2d, max_pool2d
from ..functions.roi_align import RoIAlignFunction, RoIAlignAdaFunction, RoIAlignDenseAdaFunction

import torch


class RoIAlign(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlign, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RoIAlignFunction.apply(features, rois,
                                self.aligned_height, self.aligned_width,
                                self.spatial_scale)

class RoIAlignAvg(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlignAvg, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x =  RoIAlignFunction.apply(features, rois,
                              self.aligned_height+1, self.aligned_width+1,
                                self.spatial_scale)
        return avg_pool2d(x, kernel_size=2, stride=1)

class RoIAlignMax(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlignMax, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x =  RoIAlignFunction.apply(features, rois,
                              self.aligned_height+4, self.aligned_width+4,
                                self.spatial_scale)
        return max_pool2d(x, kernel_size=3, stride=2)


class RoIAlignAdaMax(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlignAdaMax, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x =  RoIAlignAdaFunction.apply(features, rois,
                                 self.aligned_height+4, self.aligned_width+4,
                                self.spatial_scale)
        return max_pool2d(x, kernel_size=3, stride=2)


class RoIAlignDenseAdaMax(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlignDenseAdaMax, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x =  RoIAlignDenseAdaFunction.apply(self.aligned_height+4, self.aligned_width+4,
                                self.spatial_scale)(features, rois)
        # x_relu = torch.nn.ReLU()(x)
        return max_pool2d(x, kernel_size=3, stride=2)