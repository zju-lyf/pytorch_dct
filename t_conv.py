import torch.nn as nn
import torch
import math

x = torch.randn(32,256,56,56)
x1 = torch.randn(32,256,56,56)
x2 = torch.cat([x,x1],dim = 1)
print (x2.shape)

conv1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0,
                               bias=False)
a = torch.randn(3,1,1)
bn1 = nn.BatchNorm2d(256)
relu = nn.ReLU(inplace=True)
y = conv1(x2)
y = bn1(y)
y = relu(y)
print (y.shape)
#print(y)
