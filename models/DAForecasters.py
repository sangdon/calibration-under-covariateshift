import os, sys
import numpy as np
import itertools

import torch as tc
import torch.tensor as T
import torch.nn as nn
import torch.nn.functional as F

from .BaseForecasters import *

##
## discriminator, code from 
##
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    
def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

class SimpleDiscriminatorNet(Forecaster):
    def __init__(self, in_feature, hidden_size, dropout=0.5):
        super(SimpleDiscriminatorNet, self).__init__()
        self.no_hidden = hidden_size == 0

        if self.no_hidden:
            self.ad_layer = nn.Linear(in_feature, 1)
        else:
            self.ad_layer1 = nn.Linear(in_feature, hidden_size)
            self.ad_layer2 = nn.Linear(hidden_size, 1)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout)
            
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

        self.beta = nn.Parameter(T(1.0))

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        if x.requires_grad:
            x.register_hook(grl_hook(coeff))

        if self.no_hidden:
            x = self.ad_layer(x)
        else:
            x = self.ad_layer1(x)
            x = self.relu1(x)
            x = self.dropout1(x)
            x = self.ad_layer2(x)
        x = x / self.beta
        return x
    
    def train_parameters(self):
        if self.no_hidden:
            return self.ad_layer.parameters()
        else:
            return itertools.chain(
                self.ad_layer1.parameters(),
                self.ad_layer2.parameters())
    
    def cal_parameters(self):
        return [self.beta]

##
## forecaster
##
class NaiveForecaster(Forecaster):
    def __init__(self):
        super().__init__()
        
    def feature(self, xs):
        return xs
        
    def forward(self, xs):
        xs = self.feature(xs)
        return xs

    def train_parameters(self):
        return []
    
    def cal_parameters(self):
        return []
    

class ScalarForecaster(Forecaster):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(T(1.0))
        
    def feature(self, xs):
        return xs
        
    def forward(self, xs):
        xs = self.feature(xs)
        xs = xs / self.beta
        return xs

    def train_parameters(self):
        return []
    
    def cal_parameters(self):
        return [self.beta]
    
    
class SimpleFNNForecaster(Forecaster):
    def __init__(self, in_feature, hidden_size, n_labels, dropout=0.5):
        super().__init__()
        self.layer1 = nn.Linear(in_feature, hidden_size)
        self.layer2 = nn.Linear(hidden_size, n_labels)
        self.relu1 = nn.ReLU()
        self.dropout = dropout
        self.dropout1 = nn.Dropout(dropout)
        self.beta = nn.Parameter(T(1.0))

    def feature(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        return x

    def forward(self, x):
        x = self.feature(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = x / self.beta
        return x

    def train_parameters(self):
        return itertools.chain(self.layer1.parameters(), self.layer2.parameters())
    
    def cal_parameters(self):
        return [self.beta]

##
## forecaster for DA
##
class DAForecaster(Forecaster):
    def __init__(self, model_S, model_D, model_F):
        super().__init__()
        self.model_S = model_S
        self.model_D = model_D
        self.model_F = model_F
        
    def feature(self, xs):
        with tc.no_grad():
            xs = self.model_S.feature(xs)
        xs = self.model_F.feature(xs)
        return xs
    
    def feature_S(self, xs):
        with tc.no_grad():
            xs = self.model_S.feature(xs)
        return xs
    
    def forward(self, xs):
        return self.forward_F(xs)
    
    def forward_D(self, xs):
        xs = self.feature(xs)
        xs = self.model_D(xs)
        return xs
    
    def prob_D(self, xs):
        xs = self.feature(xs)
        xs = self.model_D.prob_pred(xs)
        return xs
        
    def forward_F(self, xs):
        with tc.no_grad():
            xs = self.model_S.feature(xs)
        xs = self.model_F(xs)
        return xs
    
    def prob_F(self, xs):
        with tc.no_grad():
            xs = self.model_S.feature(xs)
        xs = self.model_F.prob_pred(xs)
        return xs
        
    def importance_weight(self, xs, U_importance, return_ghs=False):
        ghs = self.ghs(xs, U_importance)
        ghs_inv = 1/ghs
        iws = (ghs_inv - 0.5).squeeze()
        if return_ghs:
            return iws, ghs
        else:
            return iws
    
    def ghs(self, xs, U_importance=np.inf):
        ghs = self.prob_D(xs).detach()
        ghs = tc.max(ghs, T(1/(1+U_importance), device=ghs.device))
        return ghs
        
    def train(self):
        pass
    
    def eval(self):
        self.model_S.eval()
        self.model_F.eval()
        if self.model_D is not None:
            self.model_D.eval()
    
    def train_D(self):
        self.model_S.eval()
        self.model_D.train()
        
    def train_F(self):
        self.model_S.eval()
        self.model_F.train()
        
    def cal_D(self):
        self.model_S.eval()
        self.model_D.cal()
        
    def cal_F(self):
        self.model_S.eval()
        self.model_F.cal()
        
    def eval_D(self):
        self.model_S.eval()
        self.model_D.eval()
        
    def eval_F(self):
        self.model_S.eval()
        self.model_F.eval()
        
    
    def train_parameters_D(self):
        return self.model_D.train_parameters()
    
    def cal_parameters_D(self):
        return self.model_D.cal_parameters()
    
    def train_parameters_F(self):
        return self.model_F.train_parameters()
    
    def cal_parameters_F(self):
        return self.model_F.cal_parameters()
            
    
class DAForecaster_Temp_FL_IW(DAForecaster):
    def __init__(self, model_S, model_D, model_F):
        super().__init__(model_S, model_D, model_F)
        
class DAForecaster_Temp_FL(DAForecaster):
    def __init__(self, model_S, model_D, model_F):
        super().__init__(model_S, model_D, model_F)
        
    def ghs(self, xs, U_importance=np.inf):
        ghs = tc.ones(xs.size(0), 1, device=xs.device) * 0.5
        return ghs

class DAForecaster_FL(DAForecaster):
    def __init__(self, model_S, model_D, model_F):
        super().__init__(model_S, model_D, model_F)
        
    def ghs(self, xs, U_importance=np.inf):
        ghs = tc.ones(xs.size(0), 1, device=xs.device) * 0.5
        return ghs

class DAForecaster_Temp_IW(DAForecaster):
    def __init__(self, model_S, model_D, model_F):
        super().__init__(model_S, model_D, model_F)
        
    def feature(self, xs):
        xs = self.model_S.feature(xs)
        return xs
        
    def forward_F(self, xs):
        xs = self.model_S(xs)
        xs = self.model_F(xs)
        return xs
    
    def prob_F(self, xs):
        xs = self.model_S(xs)
        xs = self.model_F.prob_pred(xs)
        return xs
    
class DAForecaster_Temp(DAForecaster):
    def __init__(self, model_S, model_F):
        super().__init__(model_S, None, model_F)
        
    def forward_F(self, xs):
        xs = self.model_S(xs)
        xs = self.model_F(xs)
        return xs
    
    def prob_F(self, xs):
        xs = self.model_S(xs)
        xs = self.model_F.prob_pred(xs)
        return xs
    
    def ghs(self, xs, U_importance=np.inf):
        ghs = tc.ones(xs.size(0), 1, device=xs.device) * 0.5
        return ghs

class DAForecaster_Naive(DAForecaster):
    def __init__(self, model_S, model_F):
        super().__init__(model_S, None, model_F)
        
    def forward_F(self, xs):
        xs = self.model_S(xs)
        xs = self.model_F(xs)
        return xs
    
    def prob_F(self, xs):
        xs = self.model_S(xs)
        xs = self.model_F.prob_pred(xs)
        return xs
    
    def ghs(self, xs, U_importance=np.inf):
        ghs = tc.ones(xs.size(0), 1, device=xs.device) * 0.5
        return ghs
