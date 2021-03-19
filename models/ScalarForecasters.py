import os, sys
import numpy as np
import math

import torch as tc
import torch.tensor as T
import torch.nn as nn

from .BaseForecasters import Forecaster

class TempForecaster(Forecaster):
    def __init__(self, baseF):
        super().__init__()
        self.baseF = baseF
        self.T = nn.Parameter(T(1.0))
    
    def forward(self, xs):
        return self.baseF(xs) / self.T
    
    def train(self, train_flag=True):
        self.training = True
        self.baseF.eval()
        return self

    def eval(self):
        self.training = False
        self.baseF.eval()
        return self
    
    def train_parameters(self):
        return self.baseF.parameters()
    
    def cal_parameters(self):
        return [self.T]

    
