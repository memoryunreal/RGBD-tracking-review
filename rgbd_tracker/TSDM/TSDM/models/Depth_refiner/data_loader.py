import torch
from torchvision import transforms
import numpy as np
import cv2

# transform image to tensor in PyTorch
def get_transform_for_train():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=(0.5,0.5,0.5,0.5),std=(0.5,0.5,0.5,0.5)))
    return transforms.Compose(transform_list)

# combinate RGB(3 channels) and depth(1 channel) into RGBD(4 channel)
def combinate(rgb, d, region):
    rgbn = np.array(rgb) 
    dn = np.array(d)
    dn_crop = dn[int(region[1]):int(region[1]+region[3]),int(region[0]):int(region[0]+region[2])]
    dn_mean = np.mean(dn_crop)
    dn[dn>dn_mean*3] = dn_mean*3
    dn = (dn/np.max(dn)*255)#.astype(np.int16)
    dn = np.expand_dims(dn,2)
    rgbdn = np.concatenate((rgbn,dn), axis = 2)
    return rgbdn.astype(np.uint8)


class RGBDDataLoader(object):
    def __init__(self):
        self.ret = {}
        self.count = 0
        self.out_feature = 100

    def crop_and_resize(self, image_rgb, image_depth, region):
        self.ret['image'] = combinate(image_rgb, image_depth, region.copy())
        self.ret['target_cxcywh']= [int(region[0]+region[2]/2),
                                    int(region[1]+region[3]/2),
                                    int(region[2]),
                                    int(region[3]),]
        cx, cy, w, h = self.ret['target_cxcywh'].copy()

        # crop
        self.ret['img_cropped'] = self.ret['image'][cy-h//2:cy+h//2 , cx-w//2:cx+w//2]
        # resize
        self.ret['img_cropped_resized'] = cv2.resize(self.ret['img_cropped'], 
                                                    (self.out_feature, self.out_feature))
        self.ret['w_resized_ratio'] = np.round( self.out_feature/w, 2)
        self.ret['h_resized_ratio'] = np.round(self.out_feature/h, 2)

    def _tranform(self):
        transform = get_transform_for_train()
        img = self.ret['img_cropped_resized'].copy()
        img_tensor = transform(img)
        self.ret['img_tensor'] = img_tensor.unsqueeze(0)
        self.ret['img_rgb_tensor'] = self.ret['img_tensor'][:,[2,1,0],:,:]
        self.ret['img_ddd_tensor'] = torch.cat((self.ret['img_tensor'][:,[3],:,:], self.ret['img_tensor'][:,[3],:,:], self.ret['img_tensor'][:,[3],:,:]), 1)

    def __get__(self, image_rgb, image_depth, region):
        self.crop_and_resize(image_rgb, image_depth, region)
        self._tranform()
        self.count += 1
        return self.ret

