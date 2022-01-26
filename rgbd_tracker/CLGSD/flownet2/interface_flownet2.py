import torch
import numpy as np
import argparse

from flownet2.models import FlowNet2CS  # the path is depended on where you create this module
from flownet2.utils.frame_utils import read_gen  # the path is depended on where you create this module
import matplotlib.pyplot as plt

class Flow2CS:
    def __init__(self, model):
        parser = argparse.ArgumentParser()
        parser.add_argument('--fp16', action='store_true',
                            help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
        parser.add_argument("--rgb_max", type=float, default=255.)

        args, unknown = parser.parse_known_args()
        # initial a Net
        self.net = FlowNet2CS(args).cuda()
        # load the state_dict
        dict = torch.load(model)
        self.net.load_state_dict(dict["state_dict"])

    def flow(self, im1, im2):
        imh, imw = im1.shape[:2]
        im1 = torch.from_numpy(im1.astype(np.float32).transpose(2, 0, 1)).unsqueeze(0).cuda()
        im1 = torch.nn.functional.interpolate(im1, [320, 640])

        im2 = torch.from_numpy(im2.astype(np.float32).transpose(2, 0, 1)).unsqueeze(0).cuda()
        im2 = torch.nn.functional.interpolate(im2, [320, 640])

        im = torch.cat([im1, im2], dim=0).unsqueeze(0).permute(0, 2, 1, 3, 4)

        # process the image pair to obtian the flow
        result = self.net(im)
        result = torch.nn.functional.interpolate(result, [imh, imw]).squeeze()

        data = result.data
        data = data.cpu().numpy()

        return data

    def filter_flow(self, im1, im2, boxes, thres=1):
        H,W = im1.shape[:2]
        scale_h = 320. / H
        scale_w = 640. / W

        tmp_box = boxes.copy().reshape(-1, 4)  # [x y x y]
        tmp_box[:, 0::2] = tmp_box[:, 0::2] * scale_w
        tmp_box[:, 1::2] = tmp_box[:, 1::2] * scale_h
        tmp_box = tmp_box.astype(int)

        with torch.no_grad():
            flow = self.flow(im1, im2)

        flow = np.abs(flow).max(0)
        flow[flow < thres] = 0
        flow[flow >= thres] = 1

        out = []
        for idx, b in enumerate(tmp_box):
            tmp_flow = flow[b[1]:b[3], b[0]:b[2]].sum()
            if tmp_flow / ((b[3]-b[1])*(b[2]-b[0])) > 2/3:
                out.append(boxes[idx].tolist())

        return np.array(out).reshape(-1, 4)

    def post_process(self, map, thres=1):
        flow = np.abs(map).max(0)
        flow = flow >= thres
        return flow


if __name__ == '__main__':
    # obtain the necessary args for construct the flownet framework
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)

    args, unknown = parser.parse_known_args()
    # args.fp16 = True

    # initial a Net
    net = FlowNet2CS(args).cuda()
    # load the state_dict
    dict = torch.load("/home/space/Documents/code/flownet2-pytorch/NVIDIA-flownetv2/FlowNet2-CS_checkpoint.pth.tar")
    net.load_state_dict(dict["state_dict"])

    # load the image pair, you can find this operation in dataset.py
    pim1 = read_gen("/home/space/Documents/code/flownet2-pytorch/00000001.jpg")
    pim2 = read_gen("/home/space/Documents/code/flownet2-pytorch/00000002.jpg")

    im1 = torch.from_numpy(pim1.astype(np.float32).transpose(2, 0, 1)).unsqueeze(0).cuda()
    im1 = torch.nn.functional.interpolate(im1, [320, 640])

    im2 = torch.from_numpy(pim2.astype(np.float32).transpose(2, 0, 1)).unsqueeze(0).cuda()
    im2 = torch.nn.functional.interpolate(im2, [320, 640])

    im = torch.cat([im1, im2], dim=0).unsqueeze(0).permute(0, 2, 1, 3, 4)

    # process the image pair to obtian the flow
    result = net(im).squeeze()
    data = result.data
    data = data.cpu().numpy().transpose(1, 2, 0)
    data = np.abs(data).max(2)
    data[data < 1] = 0
    data[data > 1] = 1

    plt.imshow(data)
    plt.show()
