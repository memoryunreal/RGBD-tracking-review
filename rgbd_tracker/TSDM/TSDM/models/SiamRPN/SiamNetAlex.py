import torch
import torch.nn as nn
import sys
sys.path.append("/home/guo/zpy/Mypysot")

from models.SiamRPN.backbone.alexnet import AlexNetLegacy
from models.SiamRPN.rpn import DepthwiseRPN

class MySiamRPNAlex(nn.Module):
    def __init__(self):
        super(MySiamRPNAlex, self).__init__()
        self.backbone = AlexNetLegacy()
        self.rpn_head = DepthwiseRPN()

    def template(self, z):
        self.zf = self.backbone(z)

    def track(self, x):
        xf = self.backbone(x)
        cls, loc = self.rpn_head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc
               }

if __name__ == '__main__':
    model = MySiamRPNAlex().cuda().eval()
    model_load_path = '/home/guo/zpy/Mypysot/mypysot/data_and_result/weight/modelAlex.pth'
    model.load_state_dict(torch.load(model_load_path))
    template = torch.ones((1, 3, 127, 127))
    detection= torch.ones((1, 3, 287, 287))
    model.template(template.cuda())
    state = model.track(detection.cuda())
    print(state['cls'].shape) #[1, 10, 21, 21]
    print(state['loc'].shape) #[1, 20, 21, 21]
