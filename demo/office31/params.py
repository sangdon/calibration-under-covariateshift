import os, sys
import types
import numpy as np

class BaseParamParser:
    def __init__(self):
        pass

    def read_params_static(self, exp_name):
        params = types.SimpleNamespace()
        
        # alg options
        params.U_importance = 1e3 # choose a large value where GD DOES converge
        params.lambda_D = 1.0 # 1.0 for two steps, no need to tune
        
        # loss    
        params.loss_D_type = "L2" # "L2", "BCE"
        params.loss_F_type = "L2" # "L2", "CE"
        
        # train optimizer options
        params.optimizer_train = "SGD"
        params.n_epochs_train = 500
        params.lr_D_train = 0.001
        params.lr_F_train = 0.001
        params.lr_decay_epoch_train = 100
        params.lr_decay_rate_train = 0.5
        params.weight_decay_train_DF = 0.0005
        params.weight_decay_train_D = 0.0005
        params.momentum_train = 0.9 #SGD
        
        params.early_stop_cri_train = 0.2
        params.train_DF_cross_val = False
        
        params.train_DF_loss_D_min = 0.2
        #params.train_DF_loss_F_max = 0.1 # this should be changed depending the initial loss_F
        
        # calibration optimizer options
        params.optimizer_cal = "SGD"
        params.n_epochs_cal = 4000 ## 1000?
        params.lr_D_cal = 0.01
        params.lr_F_cal = 0.1
        params.lr_decay_epoch_cal = 800
        params.lr_decay_rate_cal = 0.5
        params.weight_decay_cal = 0.0
        params.momentum_cal = 0.9 #SGD
        params.early_stop_cri_cal = 0.2
        
        # others
        params.exp_name = exp_name
        params.batch_size = 32
        params.use_gpu = True
        params.n_labels = 31
        params.snapshot_root = "snapshots"
        params.load_model = True
        params.n_epochs_eval = 1
        ##---------------------------- PARAMETERS
        return params

class AmazonParamParser(BaseParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)

        params.n_features = 2048
        params.D_n_hiddens = 1000
        params.F_n_hiddens = 1000

        return params

class WebcamParamParser(BaseParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)

        params.n_features = 2048
        params.D_n_hiddens = 1000
        params.F_n_hiddens = 1000

        return params

class DSLRParamParser(BaseParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)

        params.n_features = 2048
        params.D_n_hiddens = 1000
        params.F_n_hiddens = 1000

        return params



##
## Temp
class Temp_AmazonParamParser(AmazonParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)
        
        return params

class Temp_WebcamParamParser(WebcamParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)

        return params
    
class Temp_DSLRParamParser(DSLRParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)

        return params

##
## Temp_IW
class Temp_IW_AmazonParamParser(AmazonParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)

        return params

class Temp_IW_WebcamParamParser(WebcamParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)
        
        return params

class Temp_IW_DSLRParamParser(DSLRParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)
        
        return params

##
## Temp_FL
class Temp_FL_AmazonParamParser(AmazonParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)
    
        return params

class Temp_FL_WebcamParamParser(WebcamParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)

        return params

class Temp_FL_DSLRParamParser(DSLRParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)

        return params


##
## FL
class FL_AmazonParamParser(AmazonParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)
    
        return params

class FL_WebcamParamParser(WebcamParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)

        return params

class FL_DSLRParamParser(DSLRParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)

        return params


##
## Temp_FL_IW
class Temp_FL_IW_AmazonParamParser(AmazonParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)
        
        params.lr_F_cal = 0.01 # smaller learning rate to avoid divergence due to IW

        return params
    
class Temp_FL_IW_WebcamParamParser(WebcamParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)

        params.lr_F_cal = 0.01 # smaller learning rate to avoid divergence due to IW

        return params

class Temp_FL_IW_DSLRParamParser(DSLRParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)

        params.lr_F_cal = 0.01 # smaller learning rate to avoid divergence due to IW

        return params
    
