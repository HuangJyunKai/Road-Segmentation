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
    return im
def make_dataset(root):
    count=1
    for filename in os.listdir(root):
        for filename2 in os.listdir(root+filename):
            if filename == 'label':
                img_path=os.path.join(root+'label/'+filename2)
                img=Image.open(img_path)
                img=convert_grayscale(img)
                img.save(root+'label_grayscale/'+filename2)
                print(count)
                count+=1
if __name__ == '__main__':
    root='./data_road/training/'
    make_dataset(root)