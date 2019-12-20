# encoding: UTF-8
import glob as gb
import cv2
img_path = gb.glob("./Result/FCNS/um/*.png") 
img_path.sort()
img_gt = gb.glob("./Result/FCNS/um_gt/*.png") 
img_gt.sort()
videoWriter = cv2.VideoWriter('./Result/FCNS/um_gt.avi',cv2.VideoWriter_fourcc(*'MJPG'), 15, (1242,375))
for path ,gt_path in zip(img_path , img_gt):
    print(path)
    img  = cv2.imread(path) 
    img = cv2.resize(img,(1242,375))
    img_gt = cv2.imread(gt_path)
    img_gt = cv2.resize(img_gt,(1242,375))
    alpha = 0.75
    beta = 1-alpha
    gamma = 0
    img_add = cv2.addWeighted(img_gt, alpha, img, beta, gamma)
    videoWriter.write(img_add) 