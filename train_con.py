import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import argparse
import os
import cv2
from PIL import Image
from torchvision import datasets, models, transforms
from network.resnet_concat3 import *
from network.transform import mesonet_data_transforms
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


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
        #img_dct = torch.cat([img,dct],dim = 0)
        return img, dct, label
def model_concat(input1,input2,model,model1,model2):
    _,layer_1 = model1(input1)
    _,layer_2 = model2(input2)
    layer_con = torch.cat([layer_1,layer_2],dim = 1)
    out,_ = model(layer_con)
    return out
def main():
    args = parse.parse_args()
    name = args.name
    continue_train = args.continue_train
    epoches = args.epoches
    batch_size = args.batch_size
    model_name = args.model_name
    model_path = args.model_path
    root = args.root
    output_path = os.path.join('./dct', name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    torch.backends.cudnn.benchmark=True
    train_dataset = MyDataset(txt=root + 'train.txt', transform=mesonet_data_transforms['train'])
    val_dataset = MyDataset(txt=root + 'val.txt', transform=mesonet_data_transforms['val'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    train_dataset_size = len(train_dataset)
    val_dataset_size = len(val_dataset)
    model1 = resnet50(num_classes=2)
    model2 = resnet50(num_classes=2)
    model = resnet50_1(num_classes=2)
    if continue_train:
        model.load_state_dict(torch.load(model_path))
    model1 = model1.cuda()
    model2 = model2.cuda()
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    best_model_wts = model.state_dict()
    best_acc = 0.0
    iteration = 0
    for epoch in range(epoches):
        print('Epoch {}/{}'.format(epoch+1, epoches))
        print('-'*10)
        model1=model1.train()
        model2=model2.train()
        model=model.train()
        train_loss = 0.0
        train_corrects = 0.0
        val_loss = 0.0
        val_corrects = 0.0
        for (image, dct ,labels) in train_loader:
            iter_loss = 0.0
            iter_corrects = 0.0
            image = image.cuda()
            dct = dct.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model_concat(image, dct, model, model1, model2)
            #optimizer.zero_grad()
			#print (layer.shape)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            iter_loss = loss.data.item()
            train_loss += iter_loss
            iter_corrects = torch.sum(preds == labels.data).to(torch.float32)
            train_corrects += iter_corrects
            iteration += 1
            if not (iteration % 20):
                print('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(iteration, iter_loss / batch_size, iter_corrects / batch_size))
        epoch_loss = train_loss / train_dataset_size
        epoch_acc = train_corrects / train_dataset_size
        print('epoch train loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        model.eval()
        with torch.no_grad():
            for (image, dct, labels) in val_loader:
                image = image.cuda()
                dct = dct.cuda()
                labels = labels.cuda()
                outputs = model_concat(image, dct, model, model1, model2)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                val_loss += loss.data.item()
                val_corrects += torch.sum(preds == labels.data).to(torch.float32)
            epoch_loss = val_loss / val_dataset_size
            epoch_acc = val_corrects / val_dataset_size
            print('epoch val loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        scheduler.step()
        if not (epoch % 10):
            torch.save(model.state_dict(), os.path.join(output_path, str(epoch) + '_' + model_name))
    print('Best val Acc: {:.4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    #torch.save(model.module.state_dict(), os.path.join(output_path, "best.pkl"))
    torch.save(model.state_dict(), os.path.join(output_path, "best.pkl"))



if __name__ == '__main__':
    parse = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--name', '-n', type=str, default='Resnet_con')
    parse.add_argument('--root', '-rt' , type=str, default = '/home/liangyf/env/DCTAnalysis/ws/DCTAnalysis/data/FF/df/c23_df/image_dct/')
    parse.add_argument('--batch_size', '-bz', type=int, default=32)
    parse.add_argument('--epoches', '-e', type=int, default='50')
    parse.add_argument('--model_name', '-mn', type=str, default='resnet.pkl')
    parse.add_argument('--continue_train', type=bool, default=False)
    parse.add_argument('--model_path', '-mp', type=str, default='./dct/Resnet_con/best.pkl')
    main()
