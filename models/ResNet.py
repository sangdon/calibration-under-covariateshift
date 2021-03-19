import os, sys
import numpy as np

import torch as tc
import torch.tensor as T
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from .BaseForecasters import *

        
class ResNet50_Office31(Forecaster):
    def __init__(self, pretrained=True, n_labels = 31):
        super().__init__()
        model = torchvision.models.resnet50(pretrained=pretrained)
        n_features = model.fc.in_features
        fc = tc.nn.Linear(n_features, n_labels)
        model.fc = fc
        self.pred = model
    
    def custom_parameters(self, lr):
        param_group = [] 
        for n, p in self.named_parameters():            
            if 'fc' in n:
                param_group += [{'params': p, 'lr': lr}]
            else:
                param_group += [{'params': p, 'lr': lr * 0.1}]
        return param_group
        
    def forward(self, x):
        return self.pred(x)
    
    def feature(self, x):
        x = self.pred.conv1(x)
        x = self.pred.bn1(x)
        x = self.pred.relu(x)
        x = self.pred.maxpool(x)

        x = self.pred.layer1(x)
        x = self.pred.layer2(x)
        x = self.pred.layer3(x)
        x = self.pred.layer4(x)

        x = self.pred.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    

class ResNet152(Forecaster):
    def __init__(self, pretrained=True, n_labels = 1000, load_type='none'):
        super().__init__()
        self.load_type = load_type
        model = torchvision.models.resnet152(pretrained=pretrained)

        if self.load_type == 'none':
            self.pred = model
        elif 'feature' in self.load_type:
            self.pred = model.fc
        elif 'logit' in self.load_type:
            self.pred = lambda xs: xs
        else:
            raise NotImplementedError
    
    def forward(self, xs):
        return self.pred(xs)
    
    def feature(self, x):
        if self.load_type == 'none':
            x = self.pred.conv1(x)
            x = self.pred.bn1(x)
            x = self.pred.relu(x)
            x = self.pred.maxpool(x)

            x = self.pred.layer1(x)
            x = self.pred.layer2(x)
            x = self.pred.layer3(x)
            x = self.pred.layer4(x)

            x = self.pred.avgpool(x)
            x = x.view(x.size(0), -1)
            return x
        elif 'feature' in self.load_type:
            return x
        else:
            raise NotImplementedError
            

