import os
import cv2
import numpy as np
import math
import time

def hsv2bgr(color_hsv):
    h = float(color_hsv[0])
    s = float(color_hsv[1])
    v = float(color_hsv[2])
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return [b, g, r]

def bgr2hsv(color_bgr):
    b, g, r = color_bgr[0]/255.0, color_bgr[1]/255.0, color_bgr[2]/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    return [h, s, v]

class Depth_mask():
    def __init__(self):
        self.region = [1,1,1,1]
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        self.depth = 1000
        self.gama = 5000
        self.isWork = True
        self.double_color = True
        self.mask_color1 = [0,0,0]
        self.mask_color2 = [0,0,0]

    def general_mask(self, image_rgb, image_depth):
        depth = self.depth
        h, w = image_depth.shape[0], image_depth.shape[1]
        obw, obh = self.region[2],  self.region[3]
        cx, cy = self.region[0] + self.region[2]/2, self.region[1] + self.region[3]/2
        x1, y1 = cx - 0.75*obw, cy - 0.75*obh
        x2, y2 = cx + 0.75*obw, cy + 0.75*obh
        x1 = np.clip(int(x1), 0, w)
        y1 = np.clip(int(y1), 0, h)
        x2 = np.clip(int(x2), 0, w)
        y2 = np.clip(int(y2), 0, h)

        # get M
        mask = np.array(image_depth)
        mask[mask < depth/2] = 0
        mask[mask > depth*2] = 0
        mask[mask > 0] = 255
        mask[y1:y2, x1:x2] = 255
        mask =mask.astype(np.uint8)
        mask = cv2.dilate(mask, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        # get Mc and Xm
        image_rgb = np.array(image_rgb)
        if self.double_color:
            mask[mask != 255] = np.random.randint(0,2,mask[mask != 255].shape)
            image_rgb[:,:,0][mask == 1] = self.mask_color1[0]
            image_rgb[:,:,0][mask == 0] = self.mask_color2[0]
            image_rgb[:,:,1][mask == 1] = self.mask_color1[1]
            image_rgb[:,:,1][mask == 0] = self.mask_color2[1]
            image_rgb[:,:,2][mask == 1] = self.mask_color1[2]
            image_rgb[:,:,2][mask == 0] = self.mask_color2[2]
        else:
            image_rgb[:,:,0][mask != 255] = self.mask_color1[0]
            image_rgb[:,:,1][mask != 255] = self.mask_color1[1]
            image_rgb[:,:,2][mask != 255] = self.mask_color1[2]

        return  image_rgb

    def get_depth(self, image_depth, region, confidence):
        self.region = region
        #crop object
        h, w = image_depth.shape[0], image_depth.shape[1]
        x1, y1 = np.clip(int(region[0]), 0, w-1), np.clip(int(region[1]), 0, h-1)
        x2, y2 = np.clip(int(region[0]+region[2]), x1+1, w), np.clip(int(region[1]+region[3]), y1+1, h)
        self.region = region.copy()
        object_depth = image_depth[y1:y2, x1:x2]
        object_depth = np.array(object_depth)
        
        # get mean value of object
        sequence_depth = object_depth.copy().reshape(-1)
        len1 = len(sequence_depth[sequence_depth < 10])
        len2 = len(sequence_depth)
        if  len1 > len2*0.7:
            self.isWork = False
            return
        sequence_depth = sequence_depth[sequence_depth > 10]
        depth_max, depth_min = np.max(sequence_depth), np.min(sequence_depth)
        sequence_depth = np.uint8((sequence_depth - depth_min)/(depth_max - depth_min)*255)
        thresh, _ = cv2.threshold(sequence_depth,0,255,cv2.THRESH_OTSU)
        current_depth = np.mean(sequence_depth[sequence_depth < thresh])
        current_depth = current_depth*(depth_max - depth_min)/255 + depth_min
        current_erro = abs(self.depth - current_depth)
        
        # whether to stop M-g
        if (current_erro > self.gama and confidence < 0.65) or (confidence < 0.55):
            self.isWork = False
            return
        else:
            self.depth = current_depth

    def start_mask(self, image_rgb, image_depth, region):
        self.region = region
        h, w = image_depth.shape[0], image_depth.shape[1]
        x1, y1 = np.clip(int(region[0]), 0, w-1), np.clip(int(region[1]), 0, h-1)
        x2, y2 = np.clip(int(region[0]+region[2]), x1+1, w), np.clip(int(region[1]+region[3]), y1+1, h)
        object_depth = image_depth[y1:y2, x1:x2]
        object_depth = np.array(object_depth)
        object_rgb = image_rgb[y1:y2, x1:x2, :]
        object_rgb = np.array(object_rgb)
        self.depth = np.mean(object_depth)
        self.isWork = True

        # get limit between two frams
        self.gama = (np.max(image_depth) - np.min(image_depth))/100

        # get color of Mc
        color = [0,0,0]
        color[0] = np.mean(image_rgb[:,:,0]).astype('uint8')
        color[1] = np.mean(image_rgb[:,:,1]).astype('uint8')
        color[2] = np.mean(image_rgb[:,:,2]).astype('uint8')
        color = bgr2hsv(color)
        if self.double_color:
            self.mask_color1[0] = int(120 + color[0])%360
            self.mask_color2[0] = int(240 + color[0])%360
            self.mask_color1[1] = 1 #1-color[1]
            self.mask_color2[1] = 1 #1-color[1]
            self.mask_color1[2] = 0.7 #max(1 - color[2], color[1])
            self.mask_color2[2] = 0.7 #max(1 - color[2], color[1])
            self.mask_color1 = hsv2bgr(self.mask_color1)
            self.mask_color2 = hsv2bgr(self.mask_color2)    
        else:
            self.mask_color1[0] = int(180 + color[0])%360
            self.mask_color1[1] = color[1]
            self.mask_color1[2] = color[2]
            self.mask_color1 = hsv2bgr(self.mask_color1)


if __name__ == '__main__':
    Masker = Depth_mask()
    region = [338.23,164.54,98.909,54.001]
    image_file_rgb= '/home/guo/zpy/vot-toolkit-master/sequences/XMG_outside/color/00000001.jpg'
    image_file_depth='/home/guo/zpy/vot-toolkit-master/sequences/XMG_outside/depth/00000001.png'

    img_rgb = cv2.imread(image_file_rgb)
    img_depth = cv2.imread(image_file_depth, -1)

    Masker.start_mask(img_rgb, img_depth, region)
    Masker.get_depth(img_depth, region, 1)
    new = Masker.general_mask(img_rgb, img_depth)

    save_path = 'new.jpg'
    cv2.imwrite(save_path, new)


