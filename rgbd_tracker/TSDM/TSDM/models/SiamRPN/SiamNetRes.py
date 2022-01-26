import torch
import torch.nn as nn
import sys

from TSDM.models.SiamRPN.backbone.resnet_atrous import ResNet50
from TSDM.models.SiamRPN.neck import AdjustAllLayer
from TSDM.models.SiamRPN.rpn import MultiRPN

class MySiamRPNRes(nn.Module):
    def __init__(self):
        super(MySiamRPNRes, self).__init__()
        self.backbone = ResNet50()
        self.neck = AdjustAllLayer()
        self.rpn_head = MultiRPN()

    def template(self, z):
        zf = self.backbone(z)
        zf_s = self.neck(zf)
        self.zf_s = zf_s

    def track(self, x):
        xf = self.backbone(x)
        xf_s = self.neck(xf)
        cls, loc = self.rpn_head(self.zf_s, xf_s)
        return {
                'cls': cls,
                'loc': loc
               }

if __name__ == '__main__':
    model = MySiamRPNRes()
    model_load_path = '/home/guo/zpy/Mypysot/mypysot/data_and_result/weight/modelRes.pth'
    model.load_state_dict(torch.load(model_load_path))
    template = torch.ones((1, 3, 127, 127))
    detection= torch.ones((1, 3, 831, 831))
    model.template(template)
    state = model.track(detection)
    print(state['cls'].shape) #[1, 10, 25, 25]
    print(state['loc'].shape) #[1, 20, 25, 25]
