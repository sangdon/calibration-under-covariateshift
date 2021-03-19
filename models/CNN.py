import os, sys
import numpy as np

import torch as tc
import torch.tensor as T
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from .BaseForecasters import *

class LeNet5(Forecaster):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.lin1 = nn.Linear(256, 120)
        self.lin2 = nn.Linear(120, 84)
        self.lin3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.feature(x)
        x = self.lin3(x)
        return x
    
    def feature(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2), 2)
        x = x.view(-1, np.prod(x.shape[1:]))
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return x

# For SVHN dataset
class DTN(Forecaster):
    def __init__(self, dropout=True):
        super(DTN, self).__init__()
        self.conv_params = nn.Sequential (
                nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.Dropout2d(0.1 if dropout else 0.0),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(128),
                nn.Dropout2d(0.3 if dropout else 0.0),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(256),
                nn.Dropout2d(0.5 if dropout else 0.0),
                nn.ReLU()
                )

        self.fc_params = nn.Sequential (
                nn.Linear(256*4*4, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5 if dropout else 0.0)
                )

        self.classifier = nn.Linear(512, 10)

    def feature(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        return x
    
    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        return x



