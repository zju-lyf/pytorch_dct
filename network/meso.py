import os
import argparse


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision

class Meso4(nn.Module):
	"""
	Pytorch Implemention of Meso4
	Autor: Honggu Liu
	Date: July 4, 2019
	"""
	def __init__(self, num_classes=2):
		super(Meso4, self).__init__()
		self.num_classes = num_classes
		#self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
		#dct_image 6 chanel
		self.conv1 = nn.Conv2d(6, 8, 3, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(8)
		self.relu = nn.ReLU(inplace=True)
		self.leakyrelu = nn.LeakyReLU(0.1)

		self.conv2 = nn.Conv2d(8, 8, 5, padding=2, bias=False)
		self.bn2 = nn.BatchNorm2d(16)
		self.conv3 = nn.Conv2d(8, 16, 5, padding=2, bias=False)
		self.conv4 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
		self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2))
		self.maxpooling2 = nn.MaxPool2d(kernel_size=(4, 4))
		#flatten: x = x.view(x.size(0), -1)
		self.dropout = nn.Dropout2d(0.5)
		self.fc1 = nn.Linear(16*4*4, 16)
		self.fc2 = nn.Linear(16, num_classes)

	def forward(self, input):
		#print(x.shape)
		x = self.conv1(input) #(8, 128, 128)
		#print(x.shape)
		x = self.relu(x)
		#print(x.shape)
		x = self.bn1(x)
		#print(x.shape)
		x = self.maxpooling1(x) #(8, 64, 64)
		#print(x.shape)

		x = self.conv2(x) #(8, 64, 64)
		#print(x.shape)
		x = self.relu(x)
		#print(x.shape)
		x = self.bn1(x)
		#print(x.shape)
		x = self.maxpooling1(x) #(8, 32, 32)
		#print(x.shape)

		x = self.conv3(x) #(16, 32, 32)
		#print(x.shape)
		x = self.relu(x)
		#print(x.shape)
		x = self.bn2(x)
		#print(x.shape)
		x = self.maxpooling1(x) #(16, 16, 16)
		#print(x.shape)

		x = self.conv4(x) #(16, 32, 32)
		#print(x.shape)
		x = self.relu(x)
		#print(x.shape)
		x = self.bn2(x)
		#print(x.shape)
		x = self.maxpooling2(x) #(16, 4, 4)
		#print(x.shape)

		x = x.view(x.size(0), -1) #(Batch, 16*8*8)
		#print(x.shape)
		x = self.dropout(x)
		#print(x.shape)
		x = self.fc1(x) #(Batch, 16)
		x = self.leakyrelu(x)
		x = self.dropout(x)
		x = self.fc2(x)

		return x


class MesoInception4(nn.Module):
	"""
	Pytorch Implemention of MesoInception4
	Author: Honggu Liu
	Date: July 7, 2019
	"""
	def __init__(self, num_classes=2):
		super(MesoInception4, self).__init__()
		self.num_classes = num_classes
		#InceptionLayer1
		self.Incption1_conv1 = nn.Conv2d(3, 1, 1, padding=0, bias=False)
		self.Incption1_conv2_1 = nn.Conv2d(3, 4, 1, padding=0, bias=False)
		self.Incption1_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
		self.Incption1_conv3_1 = nn.Conv2d(3, 4, 1, padding=0, bias=False)
		self.Incption1_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
		self.Incption1_conv4_1 = nn.Conv2d(3, 2, 1, padding=0, bias=False)
		self.Incption1_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
		self.Incption1_bn = nn.BatchNorm2d(11)


		#InceptionLayer2
		self.Incption2_conv1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
		self.Incption2_conv2_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
		self.Incption2_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
		self.Incption2_conv3_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
		self.Incption2_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
		self.Incption2_conv4_1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
		self.Incption2_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
		self.Incption2_bn = nn.BatchNorm2d(12)

		#Normal Layer
		self.conv1 = nn.Conv2d(12, 16, 5, padding=2, bias=False)
		self.relu = nn.ReLU(inplace=True)
		self.leakyrelu = nn.LeakyReLU(0.1)
		self.bn1 = nn.BatchNorm2d(16)
		self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2))

		self.conv2 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
		self.maxpooling2 = nn.MaxPool2d(kernel_size=(4, 4))

		self.dropout = nn.Dropout2d(0.5)
		self.fc1 = nn.Linear(16*8*8, 16)
		self.fc2 = nn.Linear(16, num_classes)


	#InceptionLayer
	def InceptionLayer1(self, input):
		x1 = self.Incption1_conv1(input)
		x2 = self.Incption1_conv2_1(input)
		x2 = self.Incption1_conv2_2(x2)
		x3 = self.Incption1_conv3_1(input)
		x3 = self.Incption1_conv3_2(x3)
		x4 = self.Incption1_conv4_1(input)
		x4 = self.Incption1_conv4_2(x4)
		y = torch.cat((x1, x2, x3, x4), 1)
		y = self.Incption1_bn(y)
		y = self.maxpooling1(y)

		return y

	def InceptionLayer2(self, input):
		x1 = self.Incption2_conv1(input)
		x2 = self.Incption2_conv2_1(input)
		x2 = self.Incption2_conv2_2(x2)
		x3 = self.Incption2_conv3_1(input)
		x3 = self.Incption2_conv3_2(x3)
		x4 = self.Incption2_conv4_1(input)
		x4 = self.Incption2_conv4_2(x4)
		y = torch.cat((x1, x2, x3, x4), 1)
		y = self.Incption2_bn(y)
		y = self.maxpooling1(y)

		return y

	def forward(self, input):
		x = self.InceptionLayer1(input) #(Batch, 11, 128, 128)
		x = self.InceptionLayer2(x) #(Batch, 12, 64, 64)

		x = self.conv1(x) #(Batch, 16, 64 ,64)
		x = self.relu(x)
		x = self.bn1(x)
		x = self.maxpooling1(x) #(Batch, 16, 32, 32)

		x = self.conv2(x) #(Batch, 16, 32, 32)
		x = self.relu(x)
		x = self.bn1(x)
		x = self.maxpooling2(x) #(Batch, 16, 8, 8)

		x = x.view(x.size(0), -1) #(Batch, 16*8*8)
		x = self.dropout(x)
		x = self.fc1(x) #(Batch, 16)
		x = self.leakyrelu(x)
		x = self.dropout(x)
		x = self.fc2(x)

		return x
	def features(self,input):
		feature_outputs = {}
		x = self.InceptionLayer1(input)  # (Batch, 11, 128, 128)
		x = self.InceptionLayer2(x)  # (Batch, 12, 64, 64)

		x = self.conv1(x)  # (Batch, 16, 64 ,64)
		x = self.relu(x)
		x = self.bn1(x)
		x = self.maxpooling1(x)  # (Batch, 16, 32, 32)

		x = self.conv2(x)  # (Batch, 16, 32, 32)
		feature_outputs["conv1"] = x
		x = self.relu(x)
		x = self.bn1(x)
		x = self.maxpooling2(x)  # (Batch, 16, 8, 8)
		#feature_outputs["conv"] = x

		x = x.view(x.size(0), -1)  # (Batch, 16*8*8)
		x = self.dropout(x)
		x = self.fc1(x)  # (Batch, 16)
		x = self.leakyrelu(x)
		x = self.dropout(x)
		x = self.fc2(x)

		return feature_outputs
