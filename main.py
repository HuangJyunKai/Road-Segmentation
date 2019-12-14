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
from network import R2U_Net,AttU_Net,R2AttU_Net,U_Net
from unet_resnet18 import ResNetUNet
from fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs
from dataset import RoadDataset
from model import FeatureResNet, SegResNet
import numpy as np

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
	transforms.Resize((375,1242)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    #transforms.Normalize(mean=[0.5],std=[0.5])
])

# mask只需要转换为tensor

#y_transforms = transforms.ToTensor()
y_transforms = transforms.Compose([
	transforms.Resize((375,1242)),
    #transforms.ToTensor()
])

# mask只需要转换为tensor
def val(epoch,model, criterion, optimizer, dataload):
    model.eval()
    road_dataset = RoadDataset("/home/cvlab04/Desktop/Code/Medical/u_net_liver/data/val/", transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(road_dataset, batch_size=6)
    step=0
    epoch_loss = 0
    print("Validation...")
    with torch.no_grad():
        for x, y in dataloaders:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
        print("epoch %d Val_loss:%0.5f " % (epoch, epoch_loss/step))
    return epoch_loss/step

def train_model(model, criterion, optimizer, dataload, scheduler,num_epochs=50):
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
        fp = open("Road_%d_unet.txt" % num_epochs, "a")
        fp.write("epoch %d loss:%0.5f " % (epoch, epoch_loss/step))
        fp.close()
        
    torch.save(model.state_dict(), 'weights_%d_unet_road.pth' % num_epochs)
    return model

#训练模型
def train(args):
    step_size  = 50
    gamma      = 0.5
    #vgg_model = VGGNet(requires_grad=True, remove_fc=True)
    #model = FCNs(pretrained_net=vgg_model, n_class=3).to(device)
    #model=torchvision.models.segmentation.fcn_resnet101(pretrained=False, progress=True, num_classes=3).to(device)
    model = Unet(3, 3).to(device)
    #pretrained_net = FeatureResNet()
    #pretrained_net.load_state_dict(torchvision.models.resnet34(pretrained=True).state_dict())
    #model = SegResNet(3, pretrained_net).to(device)
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
def test(args):
    model = Unet(1, 1)
    #model = R2Att_Net()
    #model = U_Net()
    model.load_state_dict(torch.load(args.ckpt,map_location='cpu'))
    road_dataset = RoadDataset("./data_road/testing/", transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(road_dataset, batch_size=6)
    model.eval()
    import matplotlib.pyplot as plt
    #plt.ion()
    count=0
    count_sum=0.
    dice_loss=0.
    with torch.no_grad():
        for x, labels in dataloaders:
            count+=1
            print("batch:",count)
            y=model(x)
            img_y=torch.squeeze(y).numpy()
            img_y = (img_y> 0.3).astype(np.uint8)
            img_y = img_y.flatten()
            count_predict=np.count_nonzero(img_y > 0)
            #print("predict pixel:   ",count_predict)
            true =torch.squeeze(labels).numpy()
            true  = true .flatten()
            count_true=np.count_nonzero(true > 0)
            #print("true pixel:   ",count_true)
            ans=0
            ans = np.count_nonzero(img_y*true>0)
            dice_loss = (2*ans+0.0001)/(count_predict+count_true + 0.0001)
            print("dice_loss:",dice_loss)
            
            count_sum += (dice_loss)
        print("Final_Dice_Loss:",count_sum/count)
train_to_full = {0:0,1:1,2:2}
full_to_colour = {0: (255, 0, 0), 1: (255, 255, 255),2: (0,0, 255)}

def check(args):
    model = Unet(3, 3)
    #vgg_model = VGGNet(requires_grad=True, remove_fc=True)
    #model = FCN8s(pretrained_net=vgg_model, n_class=1)
    model.load_state_dict(torch.load(args.ckpt,map_location='cpu'))
    model.eval()
    import PIL.Image as Image
    img = Image.open('./data_road/training/image/um_000004.png')
    #img = Image.open('/home/cvlab04/Desktop/Code/Medical/u_net_liver/A001-23230277-27.jpeg').convert('RGB')
    img = x_transforms(img)
    img = img.view(1,3,375,1242)
    import matplotlib.pyplot as plt
    plt.ion()
    with torch.no_grad():
        output= model(img)
        #print(output.shape)
        output = torch.softmax(output,dim=1)
        N, _, h, w = output.shape
        #print(output)
        pred = output.transpose(0, 2).transpose(3, 1).reshape(-1, 3).argmax(axis=1).reshape(N, h, w)
        pred = pred.squeeze(0)
        print(pred)
        Decode_image(pred)


def Model_visualization(args):
    from torchsummary import summary
    model = Unet(1, 1).to(device)
    summary(model, input_size=(1,512,512)) 
def Decode_image(img_n):
    import PIL.Image as Image
    img_ans=np.zeros((img_n.shape[0],img_n.shape[1],3), dtype=np.int)
    for i in range(img_n.shape[0]):
        for j in range(img_n.shape[1]):
            if img_n[i][j] == 0: #black
                img_ans[i][j][0] = 0
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
    im_ans.save("./Result/um_000004_pred.png")

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
