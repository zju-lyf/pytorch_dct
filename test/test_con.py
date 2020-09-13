import torch
import torch.nn as nn
import torchvision
#from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import cv2
from torchvision import datasets, models, transforms
from network.resnet_con import *
from network.xception import *
from network.transform import mesonet_data_transforms
from PIL import Image
from PIL import ImageFile
import numpy
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

loader = transforms.Compose([
    transforms.ToTensor()])

unloader = transforms.ToPILImage()
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

root = '/home/liangyf/env/DCTAnalysis/ws/DCTAnalysis/data/FF/df/c23_df/image_dct/'


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        fh = open(txt, 'r')
        lines = fh.readlines()
        #print (len(lines))
        imgs = []
        for line in lines:
            line = line.strip('\n')
            line = line.rstrip('\n')
            words = line.split()
            imgs.append((words[0], words[1], int(words[2])))
        #print (words[0])
        #print(words[1])
        #print(words[2])
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __len__(self):
        #print (len(self.imgs))
        return len(self.imgs)

    def __getitem__(self, index):
        fn, fn1, label = self.imgs[index]
        img = self.loader(fn)
        img = img.resize((224, 224))
        dct = self.loader(fn1)
        dct = dct.resize((224, 224))
        if self.transform is not None:
            img = self.transform(img)
            dct = self.transform(dct)
        img_dct = torch.cat([img,dct],dim = 0)
        return img, dct, label
def model_concat(input1,input2,model,model1,model2):
    _,layer_1 = model1(input1)
    _,layer_2 = model2(input2)
    layer_con = torch.cat([layer_1,layer_2],dim = 1)
    #print (layer_1.shape)
    #print (layer_2.shape)
    #print (layer_con.shape)
    out,_ = model(layer_con)
    return out
def main():
    args = parse.parse_args()
    batch_size = args.batch_size
    model_path = args.model_path
    torch.backends.cudnn.benchmark = True
    test_data = MyDataset(txt=root + 'test.txt', transform=transforms.ToTensor())
    #val_data = MyDataset(txt=root + 'val.txt', transform=transforms.ToTensor())
    # print (len(train_data))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True,
                                               num_workers=8)

    test_dataset_size = len(test_data)
    corrects = 0
    acc = 0
    # model = MobileNetV3_Small()
    model1 = resnet50(num_classes=2)
    model2 = resnet50(num_classes=2)
    #model = resnet50_1(num_classes=2)
    model = resnet50_1(num_classes=2)
    # model = Xception()
    # model = Resnet()
    model.load_state_dict(torch.load(model_path))
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model1 = model1.cuda()
    model2 = model2.cuda()
    model = model.cuda()
    model.eval()
    #print (test_dataset.imgs)
    with torch.no_grad():
        #Pred = []
        for (image, dct, labels) in test_loader:
            image = image.cuda()
            dct = dct.cuda()
            labels = labels.cuda()
            outputs = model_concat(image,dct,model,model1,model2)
            _, preds = torch.max(outputs.data, 1)

            corrects += torch.sum(preds == labels.data).to(torch.float32)
            print('Iteration Acc {:.4f}'.format(torch.sum(preds == labels.data).to(torch.float32) / batch_size))
        acc = corrects / test_dataset_size
        print('Test Acc: {:.4f}'.format(acc))


if __name__ == '__main__':
    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--batch_size', '-bz', type=int, default=32)
    #parse.add_argument('--test_path', '-tp', type=str, default='./test')
    parse.add_argument('--model_path', '-mp', type=str, default='./dct/Resnet_con1/best.pkl')
    #parse.add_argument('--model1_path', '-mp1', type=str, default='./output/MesoInception_FF_eye/best.pkl')
    #parse.add_argument('--model2_path', '-mp2', type=str, default='./output/MesoInception_deepwild_eye/best.pkl')
    main()
