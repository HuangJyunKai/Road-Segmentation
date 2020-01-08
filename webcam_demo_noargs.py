import argparse

import cv2
import torch

from mmdet.apis import inference_detector, init_detector, show_result
from VisualizeResults import evaluateModel
import os
import torchvision
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch import nn, optim
from torchvision.transforms import transforms
from torch.optim import lr_scheduler
from fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs
from ESPNET import ESPNet,ESPNet_Encoder
from dataset import RoadDataset
import numpy as np
import PIL.Image as Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def Decode_image(img_n):
    pallete = [[0,0,0],[255,0,255],[255, 255, 0]]
    img_n = img_n.cpu()
    img_ans=np.zeros((img_n.shape[0],img_n.shape[1],3), dtype=np.int) #class 2
    for idx in range(len(pallete)):
        [r, g, b] = pallete[idx]
        img_ans[img_n == idx] = [b, g, r] 
    im_ans = Image.fromarray(np.uint8(img_ans)).convert('RGB') 
    im_ans = cv2.cvtColor(np.array(im_ans),cv2.COLOR_RGB2BGR)         
    #im_ans.save("./result.png")
    return im_ans
def RoadSeg(img_gt,model):
    x_transforms = transforms.Compose([
    transforms.Resize((720,1280)), #for bdd
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    img = Image.fromarray(np.uint8(img_gt)).convert('RGB')
    img = x_transforms(img)
    img = img.view(1,3,720,1280) #for bbd espnet
    img = img.to(device)
    with torch.no_grad():
        output= model(img)
        output = torch.softmax(output,dim=1)
        N, _, h, w = output.shape
        pred = output.transpose(0, 2).transpose(3, 1).reshape(-1, 3).argmax(axis=1).reshape(N, h, w) #class 3
        pred = pred.squeeze(0)
        result = Decode_image(pred)
        #result = cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
        #result  = cv2.imread("./result.png")
        result = cv2.resize(result ,(img_gt.shape[1],img_gt.shape[0]))
        alpha = 0.75
        beta = 1-alpha
        gamma = 0
        img_add = cv2.addWeighted(img_gt, alpha, result, beta, gamma)
        return img_add
def main():
    #args = parse_args()
    '''model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))
    '''
    model = init_detector(
        '../configs/cityscapes/mask_rcnn_r50_fpn_1x_cityscapes.py', '../checkpoints/mask_rcnn_r50_fpn_1x_city_20190727-9b3c56a5.pth', device=torch.device('cuda', 0))
    modelseg = ESPNet_Encoder(3, p=2, q=3).to(device)
    modelseg.load_state_dict(torch.load('bdd_weights_20_ESPNET_road.pth',map_location='cpu'))
    modelseg.eval()
    #camera = cv2.VideoCapture(args.camera_id)
    camera = cv2.VideoCapture('umgt.avi')
    print('Press "Esc", "q" or "Q" to exit.')
    if camera.isOpened():
        while True:
            ret_val, img = camera.read()
            imgroad = RoadSeg(img,modelseg)
            #imgroad = evaluateModel(img,modelseg)
            result = inference_detector(model, img)

            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord('q') or ch == ord('Q'):
                break
        
            #show_result(img, result, model.CLASSES, score_thr=args.score_thr, wait_time=1)
            show_result(imgroad, result, model.CLASSES, score_thr=0.5, wait_time=1)
    cv2.destroyAllWindows()  

if __name__ == '__main__':
    main()
