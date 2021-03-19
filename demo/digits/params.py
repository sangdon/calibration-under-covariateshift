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
        params.n_epochs_cal = 1000        
        params.lr_D_cal = 0.01
        params.lr_F_cal = 0.1
        params.lr_decay_epoch_cal = 200
        params.lr_decay_rate_cal = 0.5
        params.weight_decay_cal = 0.0
        params.momentum_cal = 0.9 #SGD
        params.early_stop_cri_cal = 0.2
        
        # others
        params.exp_name = exp_name
        params.batch_size = 200
        params.use_gpu = True
        params.n_labels = 10
        params.snapshot_root = "snapshots"
        params.load_model = True
        params.n_epochs_eval = 1
        ##---------------------------- PARAMETERS
        return params

    
class MNISTParamParser(BaseParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)

        params.n_features = 84
        params.D_n_hiddens = 84
        params.F_n_hiddens = 84

        params.ratio_n_tr_FL_cross_val = 1.0
                
        return params

class USPSParamParser(BaseParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)

        params.n_features = 84
        params.D_n_hiddens = 84
        params.F_n_hiddens = 84

        params.ratio_n_tr_FL_cross_val = 1.0

        return params

            
class SVHNParamParser(BaseParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)

        params.n_features = 512
        params.D_n_hiddens = 100
        params.F_n_hiddens = 256

        params.ratio_n_tr_FL_cross_val = 0.2
        
        return params


##
## Temp
class Temp_MNISTParamParser(MNISTParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)
        
        params.lr_F_cal = 1.0 # larger lr for better results
        
        return params

class Temp_USPSParamParser(USPSParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)

        params.lr_F_cal = 1.0 # larger lr for better results

        return params
    
class Temp_SVHNParamParser(SVHNParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)

        params.lr_F_cal = 1.0 # larger lr for better results

        return params

##
## Temp_IW
class Temp_IW_MNISTParamParser(MNISTParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)
        
        return params

class Temp_IW_USPSParamParser(USPSParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)
        
        return params

class Temp_IW_SVHNParamParser(SVHNParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)
        
        return params

##
## Temp_FL
class Temp_FL_MNISTParamParser(MNISTParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)
    
        return params

class Temp_FL_USPSParamParser(USPSParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)

        return params

class Temp_FL_SVHNParamParser(SVHNParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)

        return params


##
## FL
class FL_MNISTParamParser(MNISTParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)
    
        return params

class FL_USPSParamParser(USPSParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)

        return params

class FL_SVHNParamParser(SVHNParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)

        return params



##
## Temp_FL_IW
class Temp_FL_IW_MNISTParamParser(MNISTParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)
        
        params.lr_F_cal = 0.01 # smaller learning rate to avoid divergence due to IW

        return params
    
class Temp_FL_IW_USPSParamParser(USPSParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)

        params.lr_F_cal = 0.01 # smaller learning rate to avoid divergence due to IW

        return params

class Temp_FL_IW_SVHNParamParser(SVHNParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)

        params.lr_F_cal = 0.01 # smaller learning rate to avoid divergence due to IW

        return params
    
class Temp_FL_IW_reg_SVHNParamParser(SVHNParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)

        params.lr_F_cal = 0.01 # smaller learning rate to avoid divergence due to IW
        params.weight_decay_train_D = 0.01

        return params



##
## USPS analysis
##
class Temp_FL_IW_USPSParamParser_Analysis(USPSParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)

        params.lr_F_cal = 0.01 # smaller learning rate to avoid divergence due to IW

        ## analysis
        params.weight_decay_train_DF = 0.001 # 0.0005

        return params

class Temp_FL_IW_USPSParamParser_Analysis_small_decay(USPSParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)

        params.lr_F_cal = 0.01 # smaller learning rate to avoid divergence due to IW

        ## analysis
        params.weight_decay_train_DF = 0.0001 # 0.0005

        return params

class Temp_FL_IW_USPSParamParser_Analysis_larger_decay(USPSParamParser):
    def __init__(self):
        super().__init__()

    def read_params_static(self, exp_name):
        params = super().read_params_static(exp_name)

        params.lr_F_cal = 0.01 # smaller learning rate to avoid divergence due to IW

        ## analysis
        params.weight_decay_train_DF = 0.01 # 0.0005

        return params
