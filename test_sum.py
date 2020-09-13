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
from network.resnet_sum1 import *
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


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        fh = open(txt, 'r')
        lines = fh.readlines()
        imgs = []
        for line in lines:
            line = line.strip('\n')
            line = line.rstrip('\n')
            words = line.split()
            imgs.append((words[0], words[1], int(words[2])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __len__(self):
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
def model_sum(input1,input2,model,model1):
    _,layer_1 = model1(input1)
    _,layer_2 = model(input2)
    layer_sum = layer_1+layer_2
    out,_ = model(layer_sum)
    return out
def main():
    args = parse.parse_args()
    batch_size = args.batch_size
    model_path = args.model_path
    root = args.root
    model1_path = args.model1_path
    torch.backends.cudnn.benchmark = True
    test_data = MyDataset(txt=root + 'test.txt', transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True,
                                               num_workers=8)

    test_dataset_size = len(test_data)
    corrects = 0
    acc = 0
    model1 = resnet50(num_classes=2)
    model = resnet50(num_classes=2)
    model.load_state_dict(torch.load(model_path))
    model1.load_state_dict(torch.load(model1_path))
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if isinstance(model1, torch.nn.DataParallel):
        model1 = model1.module
    model1 = model1.cuda()
    model = model.cuda()
    model.eval()
    model1.eval()
    with torch.no_grad():
        for (image, dct, labels) in test_loader:
            image = image.cuda()
            dct = dct.cuda()
            labels = labels.cuda()
            outputs = model_sum(image,dct,model,model1)
            _, preds = torch.max(outputs.data, 1)

            corrects += torch.sum(preds == labels.data).to(torch.float32)
            print('Iteration Acc {:.4f}'.format(torch.sum(preds == labels.data).to(torch.float32) / batch_size))
        acc = corrects / test_dataset_size
        print('Test Acc: {:.4f}'.format(acc))


if __name__ == '__main__':
    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--batch_size', '-bz', type=int, default=32)
    #parse.add_argument('--test_path', '-tp', type=str, default='./test')
    parse.add_argument('--model_path', '-mp', type=str, default='./dct_/Resnet_sum_1/10_resnet.pkl')
    parse.add_argument('--model1_path', '-mp1', type=str, default='./dct_/Resnet_sum_1/10_resnet1.pkl')
    parse.add_argument('--root', '-rt', type=str, default='/home/liangyf/env/DCTAnalysis/ws/DCTAnalysis/data/FF/df/c23_df/image_dct/')
    main()
