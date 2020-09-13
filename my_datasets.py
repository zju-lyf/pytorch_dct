import torch.nn.functional as F
import torch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os

learning_rate = 0.0001

root = '/home/liangyf/env/DCTAnalysis/ws/DCTAnalysis/data/FF/df/c23_df/'


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
        print (words[0])
        print(words[1])
        print(words[2])
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
        img = img.resize((256,256))
        #print(type(img))
        dct = self.loader(fn1)
        dct = dct.resize((256,256))
        if self.transform is not None:
            img = self.transform(img)
            dct = self.transform(dct)
        img_dct = torch.cat([img,dct],dim = 0)
        return img_dct, label



train_data = MyDataset(txt=root + 'image_dct.txt', transform=transforms.ToTensor())
#test_data = MyDataset(txt=root + 'text.txt', transform=transforms.ToTensor())
print (len(train_data))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, drop_last=False, num_workers=8)
i = 1
for (image, labels) in train_loader:
    if i ==1:
        print(image.shape)
        print(labels)
    i = i +1
