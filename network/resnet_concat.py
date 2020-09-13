import torch.nn as nn
import torch
import math

def model1(input1):
    model = resnet50(num_classes=2)
    model = model.cuda()
    out,layer1 = model(input1)
    #layer1 = res.stack1
    return layer1
def model_concat(input1,input2):
    layer_1 = model1(input1)
    layer_2 = model1(input2)
    layer_con = torch.cat([layer_1,layer_2],dim = 1)
    model = resnet50_1(num_classes=2)
    model = model.cuda()
    out,_ = model(layer_con)
    return out
def resnet50_1(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet([3, 4, 6, 3,3,4,6,3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(model.modelPath))
    return model

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet([3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(model.modelPath))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet([3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(model.modelPath))
    return model


class ResNet(nn.Module):
    """
    block: A sub module
    """

    def __init__(self, layers, num_classes=1000, model_path="/home/liangyf/env/py3_mesonet/ws/dct_pytorch/network/resnet50-19c8e357.pth"):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.modelPath = model_path
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if len(layers)==4:
            #print ('yy')
            self.stack1 = self.make_stack(64, layers[0])
            self.stack2 = self.make_stack(128, layers[1], stride=2)
            self.stack3 = self.make_stack(256, layers[2], stride=2)
            self.stack4 = self.make_stack(512, layers[3], stride=2)
            self.avgpool = nn.AvgPool2d(7, stride=1)
            self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
            self.init_param()
        else:
            #print ('nn')
            self.stack1 = self.make_stack(64, layers[0],expansion=8)
            self.stack2 = self.make_stack(128, layers[1], stride=2)
            self.stack3 = self.make_stack(256, layers[2], stride=2)
            self.stack4 = self.make_stack(512, layers[3], stride=2)
            self.avgpool = nn.AvgPool2d(7, stride=1)
            self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
            self.init_param()


    def init_param(self):
        # The following is initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.shape[0] * m.weight.shape[1]
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def make_stack(self, planes, blocks, stride=1, expansion=4):
        downsample = None
        layers = []
        #print ('exp')
        #print (expansion)

        if stride != 1 or self.inplanes != planes * expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * expansion),
            )

        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if (x.shape==torch.Size([32, 3, 224, 224])):
            #print ('yes')
            #print(x.shape)
            x = self.conv1(x)
            #print (x.shape)
            x = self.bn1(x)
            #print(x.shape)
            x = self.relu(x)
            #print(x.shape)
            x = self.maxpool(x)
            #print(x.shape)

            x1 = self.stack1(x)
            #print(x1.shape)
            x = self.stack2(x1)
            #print(x.shape)
            x = self.stack3(x)
            x = self.stack4(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        else:
            x1 = x
            #print('no')
            #print (x1.shape)
            x = self.stack2(x1)
            x = self.stack3(x)
            x = self.stack4(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x,x1


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        #print ('inplanes')
        #print (inplanes)
        #print (planes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        #print ('x:')
        #print (x.shape)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
