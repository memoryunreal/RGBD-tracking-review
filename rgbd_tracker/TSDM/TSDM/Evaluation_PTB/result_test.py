import cv2
import argparse
import os


parser = argparse.ArgumentParser(description='result test')
parser.add_argument('--sequence', default=1, type=int)

with open('/media/guo/DataUbuntu/PTB_dataset/all_init_bbox.txt', 'r') as f:
    lines = f.readlines()
sequence_name = lines[0:190:2]
args = parser.parse_args()
name = sequence_name[args.sequence].strip('\n')

boxes_estimated = []
result_dir = '/home/guo/zpy/Mypysot/mypysot/result/result4/' + name + '.txt'
with open(result_dir, 'r') as f:
    lines = f.readlines()
    for k in range(0, len(lines)):
        if lines[k].find('a') == -1:
            box_estimated = [int(float(i)) for i in lines[k].strip('\n').split(',')]
            boxes_estimated.append(box_estimated)
        else:
            boxes_estimated.append([-1,-1,-1,-1])

for i in range(1, 5000):
    image_file = '/media/guo/DataUbuntu/PTB_dataset/' + name + '/rgb/' + str(i).zfill(8) + '.png'
    if not os.path.exists(image_file):
        break
    img = cv2.imread(image_file)
    if boxes_estimated[i-1][0] != -1:
        x1, y1 = boxes_estimated[i-1][0], boxes_estimated[i-1][1]
        x2, y2 = boxes_estimated[i-1][2], boxes_estimated[i-1][3]
        cv2.line(img,(x1,y1),(x2,y1),(255,0,0),5)
        cv2.line(img,(x1,y2),(x2,y2),(255,0,0),5)
        cv2.line(img,(x1,y1),(x1,y2),(255,0,0),5)
        cv2.line(img,(x2,y1),(x2,y2),(255,0,0),5)
    else:
        img = cv2.putText(img, 'Target is lost', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite('/home/guo/zpy/Mypysot/mypysot/result/result5/' + str(i).zfill(8) + '.jpg', img)
    print(i)


