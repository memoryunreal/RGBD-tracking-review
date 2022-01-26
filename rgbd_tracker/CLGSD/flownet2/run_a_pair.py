import torch
import numpy as np
import argparse

from models import FlowNet2CS  # the path is depended on where you create this module
from utils.frame_utils import read_gen  # the path is depended on where you create this module

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


    images = [pim1, pim2]
    images = np.array(images).transpose(3, 0, 1, 2)

    im1 = torch.from_numpy(pim1.astype(np.float32).transpose(2, 0, 1)).unsqueeze(0).cuda()
    im1 = torch.nn.functional.interpolate(im1, [320, 640])

    im2 = torch.from_numpy(pim2.astype(np.float32).transpose(2, 0, 1)).unsqueeze(0).cuda()
    im2 = torch.nn.functional.interpolate(im2, [320, 640])

    im = torch.cat([im1, im2], dim=0).unsqueeze(0).permute(0, 2, 1, 3, 4)

    # process the image pair to obtian the flow
    result = net(im).squeeze()


    # save flow, I reference the code in scripts/run-flownet.py in flownet2-caffe project
    def writeFlow(name, flow):
        f = open(name, 'wb')
        f.write('PIEH'.encode('utf-8'))
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow = flow.astype(np.float32)
        flow.tofile(f)
        f.flush()
        f.close()


    data = result.data.cpu().numpy().transpose(1, 2, 0)
    writeFlow("/home/hjj/flownet2-master/data/FlyingChairs_examples/0000007-img.flo", data)
