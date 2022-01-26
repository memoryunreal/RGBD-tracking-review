import cv2
import os
import argparse

img = cv2.imread('1.jpg')
text = 'target score =' + '0.5'
img = cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
cv2.imwrite('10.jpg', img)




'''
parser = argparse.ArgumentParser(description='comparing')
parser.add_argument('--sequence', default='robot_human_corridor_noocc_1_B')
parser.add_argument('--indexmax', default=500, type=int)

args = parser.parse_args()
name = args.sequence
indexmax = args.indexmax

gt1 = []
rootdir = '/home/guo/zpy/vot-toolkit-master/results/TSDM/rgbd-unsupervised'
with open(os.path.join(rootdir, name, name+'_001.txt'), 'r') as f:
    lines = f.readlines()
    for index in range(1,indexmax):
        region = [int(float(i)) for i in lines[index].strip('\n').split(',')]
        region = [region[0], region[1], region[0]+region[2], region[1]+region[3]]
        gt1.append(region)

gt2 = []
rootdir = '/home/guo/zpy/vot-toolkit-master/results/OTR/rgbd-unsupervised'
with open(os.path.join(rootdir, name, name+'_001.txt'), 'r') as f:
    lines = f.readlines()
    for index in range(1,indexmax):
        region = [int(float(i)) for i in lines[index].strip('\n').split(',')]
        region = [region[0], region[1], region[0]+region[2], region[1]+region[3]]
        gt2.append(region)

gt3 = []
rootdir = '/home/guo/zpy/vot-toolkit-master/results/MBMD/rgbd-unsupervised'
with open(os.path.join(rootdir, name, name+'_001.txt'), 'r') as f:
    lines = f.readlines()
    for index in range(1,indexmax):
        region = [int(float(i)) for i in lines[index].strip('\n').split(',')]
        region = [region[0], region[1], region[0]+region[2], region[1]+region[3]]
        gt3.append(region)

path_root = '/home/guo/zpy/vot-toolkit-master/sequences/'
for i in range(2, indexmax):
    img = cv2.imread(os.path.join(path_root, name, 'color', str(i).zfill(8)+'.jpg'))
    #MBMD
    x1, y1, x2, y2 = gt3[i-2][0], gt3[i-2][1], gt3[i-2][2], gt3[i-2][3]
    cv2.line(img,(x1,y1),(x2,y1),(0,255,0),3)
    cv2.line(img,(x1,y2),(x2,y2),(0,255,0),3)
    cv2.line(img,(x1,y1),(x1,y2),(0,255,0),3)
    cv2.line(img,(x2,y1),(x2,y2),(0,255,0),3)

    #OTR
    x1, y1, x2, y2 = gt2[i-2][0], gt2[i-2][1], gt2[i-2][2], gt2[i-2][3]
    cv2.line(img,(x1,y1),(x2,y1),(0,255,255),3)
    cv2.line(img,(x1,y2),(x2,y2),(0,255,255),3)
    cv2.line(img,(x1,y1),(x1,y2),(0,255,255),3)
    cv2.line(img,(x2,y1),(x2,y2),(0,255,255),3)

    #TSDM
    x1, y1, x2, y2 = gt1[i-2][0], gt1[i-2][1], gt1[i-2][2], gt1[i-2][3]
    cv2.line(img,(x1,y1),(x2,y1),(0,0,255),3)
    cv2.line(img,(x1,y2),(x2,y2),(0,0,255),3)
    cv2.line(img,(x1,y1),(x1,y2),(0,0,255),3)
    cv2.line(img,(x2,y1),(x2,y2),(0,0,255),3)

    save_path = '/home/guo/zpy/Mypysot/mypysot/result/result7/comparison_' + str(i).zfill(8) + '.jpg'
    cv2.imwrite(save_path, img)
'''

