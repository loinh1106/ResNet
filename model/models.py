import torch
import torch.nn as nn
import numpy as np

def conv3x3(in_planes, out_planes, stride =1):
  return nn.Conv2D(in_planes, out_planes, stride =stride, kernel_size= 3 ,padding =1)
  
class BasicBlock(nn.Module):
  expansion =1
  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.conv3x3(inplanes, planes, stride)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride
  def forward(self, x):
   residual = x

   out = self.conv1(x)
   out = self.bn1(out)
   out = self.relu(out)

   out = self.conv2(out)
   out = self.bn2(out)
   if self.downsample != None:
     residual = self.downsample(x)
   out += residual
   out = self.relu(out)
   return out


class Bottleneck(nn.Module):
  expansion = 4
  def __init__(self,inplanes, planes, stride =1, downsample =None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size =1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.conv3x3(inplanes, planes, stride)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(inplanes, planes*4, kernel_size =1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes*4)
    self.downsample = downsample
    self.stride = stride
  def forward(self, x):
    residual = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample != None: 
        residual = self.downsample(x)

    out += residual
    out = self.relu(out)
    return out




class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 6):
        super(ResNet,self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2,padding=3, bias =False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers1 = self._make_layers(block, 64, layers[0])
        self.layers2 = self._make_layers(block,128, layers[1])
        self.layers3 = self._make_layers(block, 256, layers[2])
        self.layers4 = self._make_layers(block, 512, layers[3])
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(512* block.expansion, num_classes)

    def _make_layers(self, block, planes, blocks, stride=1):
        downsample = None
        if self.stride !=1 or self.inplanes != planes* block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x= self.relu(x)
        x= self.maxpool(x)

        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        x = self.layers4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0),-1)
        return x

  


  
  

  
