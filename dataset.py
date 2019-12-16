import torch
from torch.utils.data import Dataset , DataLoader
import PIL.Image as Image
import os
from torchvision import utils
from torchvision.transforms import transforms
import numpy as np
def convert_grayscale(img):
    pix = img.load()
    width = img.size[0]
    height = img.size[1]
    img=img.convert('RGB')
    array=[]
    for x in range(width):
        tmp=[]
        for y in range(height):
            r, g, b = img.getpixel((x,y))
            rgb = (r, g, b)
            tmp.append(rgb)
        array.append(tmp)
    img_t=np.zeros((width,height), dtype=np.int)
    for x in range(width):
        for y in range(height):
            if array[x][y]==(0,0,0): #black
                img_t[x][y]=0
            elif array[x][y]==(255,0,255): #purple road
                img_t[x][y]=1    
            elif array[x][y]==(255,0,0): #red background
                img_t[x][y]=2
    img_t = np.transpose(img_t)
    im = Image.fromarray((img_t).astype(np.uint8))
    return im.convert('L')
def make_dataset(root):
    imgs=[]
    masks=[]
    for filename in os.listdir(root):
        for filename2 in os.listdir(root+filename):
            if filename2 == '.ipynb_checkpoints':
                break
            if filename == 'label':
                mask=os.path.join(root+'label_grayscale/'+filename2)
                masks.append(mask)
            if filename == 'image':
                img=os.path.join(root+'image/'+filename2)
                imgs.append(img)
    masks.sort()
    imgs.sort()  
    conbine=list(zip(imgs,masks))
    #print(conbine)
    return conbine


class RoadDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path).convert('RGB')
        img_y = Image.open(y_path).convert('L')
        
        #img_y = convert_grayscale(img_y)
        #img_x = Image.open(x_path).convert('RGB')
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        img_n = np.array(img_y)
        target = torch.zeros(3, img_n.shape[0], img_n.shape[1]) #3 class
        for c in range(3):
            target[c][img_n == c] = 1
        #print(img_x.shape,target.shape)
        
        return img_x , target

    def __len__(self):
        return len(self.imgs)
