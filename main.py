#For road segmentation
import torch
import argparse
import os
import torchvision
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch import nn, optim
from torchvision.transforms import transforms
from torch.optim import lr_scheduler
from unet_model import Unet
from unet_resnet18 import ResNetUNet
from fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs
from ESPNET import ESPNet,ESPNet_Encoder
from dataset import RoadDataset
from model import FeatureResNet, SegResNet
import numpy as np

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
	#transforms.Resize((375,1242)),
    transforms.Resize((352,1216)), #for fcn
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    #transforms.Normalize(mean=[0.5],std=[0.5])
])

# mask只需要转换为tensor

#y_transforms = transforms.ToTensor()
y_transforms = transforms.Compose([
	#transforms.Resize((375,1242)),
    #transforms.Resize((352,1216)),#for fcn
    transforms.Resize((44,152)),#for ESPnet
    #transforms.ToTensor()
])



def train_model(model, criterion, optimizer, dataload, scheduler,num_epochs=200):
    for epoch in range(num_epochs):
        #scheduler.step()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0

        #num_correct = 0
        print(dt_size)
        for x, y in dataload:
            num_correct = 0
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            #print(inputs.shape,labels.shape)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            #print(outputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            print("%d/%d,train_loss:%0.5f " % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        
        print("epoch %d loss:%0.5f " % (epoch, epoch_loss/step))
        #val_loss = val(epoch,model, criterion, optimizer, dataload)
        fp = open("Road_%d_ESPNET.txt" % num_epochs, "a")
        fp.write("epoch %d loss:%0.5f \n" % (epoch, epoch_loss/step))
        fp.close()
        
    torch.save(model.state_dict(), 'weights_%d_ESPNET_road.pth' % num_epochs)
    return model

#训练模型
def train(args):
    step_size  = 50
    gamma      = 0.5
    #vgg_model = VGGNet(requires_grad=True, remove_fc=True)
    #model = FCNs(pretrained_net=vgg_model, n_class=3).to(device)
    model = ESPNet_Encoder(3, p=2, q=3).to(device)
    #model=torchvision.models.segmentation.fcn_resnet101(pretrained=False, progress=True, num_classes=3).to(device)
    #model = Unet(3, 3).to(device)
    batch_size = args.batch_size
    #criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs
    criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.CrossEntropyLoss()
    #optimizer = optim.RMSprop(model.parameters(), lr=1e-4, momentum=0, weight_decay=1e-5)
    #optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    road_dataset = RoadDataset("./data_road/training/",transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(road_dataset, batch_size=batch_size, shuffle=True)
    train_model(model, criterion, optimizer, dataloaders,scheduler)

#显示模型的输出结果
def check(args):
    #model = Unet(3, 3)
    vgg_model = VGGNet(requires_grad=True, remove_fc=True)
    model = FCN8s(pretrained_net=vgg_model, n_class=3)
    model.load_state_dict(torch.load(args.ckpt,map_location='cpu'))
    model.eval()
             
    import PIL.Image as Image
    img = Image.open('./data_road/training/image/um_000005.png')
    #img = Image.open('/home/cvlab04/Desktop/Code/Medical/u_net_liver/A001-23230277-27.jpeg').convert('RGB')
    img = x_transforms(img)
    #img = img.view(1,3,375,1242)
    img = img.view(1,3,352,1216)
    import matplotlib.pyplot as plt
    with torch.no_grad():
        output= model(img)
        #print(output.shape)
        output = torch.softmax(output,dim=1)
        N, _, h, w = output.shape
        #print(output)
        pred = output.transpose(0, 2).transpose(3, 1).reshape(-1, 3).argmax(axis=1).reshape(N, h, w) #class 3
        pred = pred.squeeze(0)
        print(pred)
        Decode_image(pred)
    
def Generate(args):
    import PIL.Image as Image
    import matplotlib.pyplot as plt
    #model = Unet(3, 3)
    vgg_model = VGGNet(requires_grad=True, remove_fc=True)
    model = FCN8s(pretrained_net=vgg_model, n_class=3)
    model.load_state_dict(torch.load(args.ckpt,map_location='cpu'))
    model.eval()
    count=0
    root = './data_road/training/'
    print("Generate...")
    for filename in os.listdir(root):
        if filename == 'image':
            for filename2 in os.listdir(root+filename):
                if filename2 == '.ipynb_checkpoints':
                    break
                imgroot=os.path.join(root+'image/'+filename2)
                name = filename2[:-4]
                print("Image Processing: ",name)
                img = Image.open(imgroot)
                img = x_transforms(img)
                img = img.view(1,3,352,1216) #for Fcn 
                with torch.no_grad():
                    output= model(img)
                    output = torch.softmax(output,dim=1)
                    N, _, h, w = output.shape
                    pred = output.transpose(0, 2).transpose(3, 1).reshape(-1, 3).argmax(axis=1).reshape(N, h, w) #class 3
                    pred = pred.squeeze(0)
                    Decode_image(pred,name)
def Model_visualization(args):
    from torchsummary import summary
    #model = Unet(3, 3).to(device)
    summary(model, input_size=(1,512,512)) 
def Decode_image(img_n,name):
    import PIL.Image as Image
    img_ans=np.zeros((img_n.shape[0],img_n.shape[1],3), dtype=np.int) #class 2
    for i in range(img_n.shape[0]):
        for j in range(img_n.shape[1]):
            if img_n[i][j] == 0: #black to red background
                img_ans[i][j][0] = 255
                img_ans[i][j][1] = 0
                img_ans[i][j][2] = 0
            elif img_n[i][j] == 1: #purple
                img_ans[i][j][0] = 255
                img_ans[i][j][1] = 0
                img_ans[i][j][2] = 255
            elif img_n[i][j] == 2: #red background
                img_ans[i][j][0] = 255
                img_ans[i][j][1] = 0
                img_ans[i][j][2] = 0
    im_ans = Image.fromarray(np.uint8(img_ans)).convert('RGB')           
    im_ans.save("./Result/FCNS/Train/"+name+"_fcn_pred200_class3.png")

if __name__ == '__main__':
    #参数解析
    parse=argparse.ArgumentParser()
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=12)
    parse.add_argument("--ckpt", type=str, help="the path of model weight file")
    args = parse.parse_args()

    if args.action=="train":
        train(args)
    elif args.action=="test":
        test(args)
    elif args.action=="check":
        check(args)
    elif args.action=="model":
        Model_visualization(args)
    elif args.action=="generate":
        Generate(args)
