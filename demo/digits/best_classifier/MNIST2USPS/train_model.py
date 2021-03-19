import os, sys
import types

import torch as tc

sys.path.append("../../../../../")
from classification import learner
from calibration.utils import *

def train_model(run_eval=True):
    ## parameters
    
    #---------- setup dependent parameters
    exp_name = "MNIST_model"
    from models.CNN import LeNet5 as Model
    from data.digits import loadUSPS, loadMNIST
    loadSrc = lambda: loadMNIST('datasets/MNIST', batch_size=100)
    loadTar = lambda: loadUSPS('datasets/USPS', batch_size=100, image_size=28)

    #---------- 
    
    # meta parameters
    params = types.SimpleNamespace()
    params.exp_name = exp_name
    params.snapshot_root = "snapshots"
    params.load_model = True
    params.save_model = True
    params.use_gpu = True
    # learning params
    params.optimizer = "Adam"
    params.n_epochs = 500
    params.lr = 0.01
    params.lr_decay_epoch = 100
    params.lr_decay_rate = 0.5
    params.keep_best = True
    params.n_epoch_eval = 10
    ## load a network
    F = Model()
    
    ## loader
    ld_src = loadSrc()
    ld_tar = loadTar()
    
    ## train
    sgd = learner.SGD(params, F)
    sgd.set_opt_params(F.parameters())
    sgd.train(ld_src.train, ld_src.val)
    
    ## eval
    if run_eval:
        sgd.test([ld_src.val, ld_src.test, ld_tar.test], ["val_src", "test_src", "test_tar"])
        
        ## confidence prediction error
        for ld_name, ld in zip(["val_src", "test_src", "test_tar"], [ld_src.val, ld_src.test, ld_tar.test]):
            ECE_L1_aware = CalibrationError_L1()(
                F.label_pred, F.tar_prob_pred, [ld])
            ECE_L1_aware_over = CalibrationError_L1()(
                F.label_pred, F.tar_prob_pred, [ld], 
                measure_overconfidence=True)
                
            print("# [%s] ECE_aware: %.2f%%, ECE_aware_over: %.2f%%"%(
                ld_name, 
                ECE_L1_aware*100.0, ECE_L1_aware_over*100.0))
        print()
    
    return F
    
if __name__ == "__main__":
    model = train_model()
