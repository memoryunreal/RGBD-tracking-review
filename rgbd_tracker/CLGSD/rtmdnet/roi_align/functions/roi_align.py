import torch
from torch.autograd import Function
from .._ext import roi_align


# TODO use save_for_backward instead
class RoIAlignFunction(Function):

    @staticmethod
    def forward(ctx, features, rois,
                aligned_height, aligned_width, spatial_scale):
        ctx.aligned_width = int(aligned_width)
        ctx.aligned_height = int(aligned_height)
        ctx.spatial_scale = float(spatial_scale)
        ctx.rois = None
        ctx.feature_size = None

        ctx.rois = rois
        ctx.feature_size = features.size()

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)

        output = features.new(num_rois, num_channels, ctx.aligned_height, ctx.aligned_width).zero_()
        if features.is_cuda:
            success = roi_align.roi_align_forward_cuda(ctx.aligned_height,
                                             ctx.aligned_width,
                                             ctx.spatial_scale, features,
                                             rois, output)
        else:
            raise NotImplementedError

        return output

    @staticmethod
    def backward(ctx, grad_output,
                 aligned_height, aligned_width, spatial_scale):
        ctx.aligned_width = int(aligned_width)
        ctx.aligned_height = int(aligned_height)
        ctx.spatial_scale = float(spatial_scale)
        ctx.rois = None
        ctx.feature_size = None

        assert(ctx.feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = ctx.feature_size

        grad_input = ctx.rois.new(batch_size, num_channels, data_height,
                                  data_width).zero_()
        roi_align.roi_align_backward_cuda(ctx.aligned_height,
                                          ctx.aligned_width,
                                          ctx.spatial_scale, grad_output,
                                          ctx.rois, grad_input)

        # print grad_input

        return grad_input, None


# TODO use save_for_backward instead
class RoIAlignAdaFunction(Function):


    @staticmethod
    def forward(ctx, features, rois,
                aligned_height, aligned_width, spatial_scale):
        ctx.aligned_width = int(aligned_width)
        ctx.aligned_height = int(aligned_height)
        ctx.spatial_scale = float(spatial_scale)
        ctx.rois = None
        ctx.feature_size = None

        ctx.rois = rois
        ctx.feature_size = features.size()

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)

        output = features.new(num_rois, num_channels, ctx.aligned_height, ctx.aligned_width).zero_()
        if features.is_cuda:
            success = roi_align.roi_align_ada_forward_cuda(ctx.aligned_height,
                                             ctx.aligned_width,
                                             ctx.spatial_scale, features,
                                             rois, output)
        else:
            raise NotImplementedError

        return output

    @staticmethod
    def backward(ctx, grad_output,
                 aligned_height, aligned_width, spatial_scale):
        ctx.aligned_width = int(aligned_width)
        ctx.aligned_height = int(aligned_height)
        ctx.spatial_scale = float(spatial_scale)
        ctx.rois = None
        ctx.feature_size = None
        assert(ctx.feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = ctx.feature_size

        grad_input = ctx.rois.new(batch_size, num_channels, data_height,
                                  data_width).zero_()
        roi_align.roi_align_ada_backward_cuda(ctx.aligned_height,
                                          ctx.aligned_width,
                                          ctx.spatial_scale, grad_output,
                                          ctx.rois, grad_input)

        # print grad_input

        return grad_input, None


# TODO use save_for_backward instead
class RoIAlignDenseAdaFunction(Function):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)
        self.rois = None
        self.feature_size = None

    @staticmethod
    def forward(ctx, features, rois,
                aligned_height, aligned_width, spatial_scale):
        ctx.aligned_width = int(aligned_width)
        ctx.aligned_height = int(aligned_height)
        ctx.spatial_scale = float(spatial_scale)
        ctx.rois = None
        ctx.feature_size = None

        ctx.rois = rois
        ctx.feature_size = features.size()

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)

        output = features.new(num_rois, num_channels, ctx.aligned_height, ctx.aligned_width).zero_()
        if features.is_cuda:
            success = roi_align.roi_align_dense_ada_forward_cuda(ctx.aligned_height,
                                             ctx.aligned_width,
                                             ctx.spatial_scale, features,
                                             rois, output)
        else:
            raise NotImplementedError

        return output

    @staticmethod
    def backward(ctx, grad_output,
                 aligned_height, aligned_width, spatial_scale):
        ctx.aligned_width = int(aligned_width)
        ctx.aligned_height = int(aligned_height)
        ctx.spatial_scale = float(spatial_scale)
        ctx.rois = None
        ctx.feature_size = None

        assert(ctx.feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = ctx.feature_size

        grad_input = ctx.rois.new(batch_size, num_channels, data_height,
                                  data_width).zero_()
        roi_align.roi_align_dense_ada_backward_cuda(ctx.aligned_height,
                                          ctx.aligned_width,
                                          ctx.spatial_scale, grad_output,
                                          ctx.rois, grad_input)

        # print grad_input

        return grad_input, None
