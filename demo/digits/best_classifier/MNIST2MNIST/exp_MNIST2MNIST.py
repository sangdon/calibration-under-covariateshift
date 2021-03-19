import os, sys
# import numpy as np
import time
# import pickle
# import types

# # vis libs
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

import torch as tc
import torch.tensor as T
# import torch.nn.functional as F
# from torch import nn, optim
# from torch.utils.data import DataLoader, TensorDataset

##--------------------------------------------------
sys.path.append("../../../../../")
from calibration.utils import *
from data.digits import loadMNIST
loadSrc = lambda batch_size: loadMNIST('datasets/MNIST', batch_size)
loadTar = lambda batch_size: loadMNIST('datasets/MNIST', batch_size)

##--------------------------------------------------

def exp(exp_name, ParamParser, model_init_fn, model_S_fn, DACalibrator, n_exps=10):
    
    ## run multiple-experiments
    ECEs_aware = []
    ECEs_aware_over = []
    best_time_train_DF = []
    end_time_train_DF = []
    best_time_train_D = []
    end_time_train_D = []
    best_time_cal_D = []
    end_time_cal_D = []
    best_time_cal_F = []
    end_time_cal_F = []

    for i_exp in range(n_exps):
        t_cur = time.time()
        exp_name_i = "%s_%d"%(exp_name, i_exp)
        
        ## parameters
        params = ParamParser().read_params_static(exp_name_i)
        
        ## datasets
        ld_src = loadSrc(params.batch_size)
        ld_tar = loadTar(params.batch_size)

        ## model
        model_S = model_S_fn()        
        model = model_init_fn(params)
        
        ## calibration
        calibrator = DACalibrator(params, model)
        calibrator.train(ld_src.train, ld_src.val, ld_tar.train, ld_tar.val)
        calibrator.test([ld_src.test, ld_tar.test], ["src_test", "tar_test"])

        ## measure ECE
        with tc.no_grad():
            ECE_aware = CalibrationError_L1()(
                model_S.label_pred, model.tar_prob_pred, [ld_tar.test])
            ECE_aware_over = CalibrationError_L1()(
                model_S.label_pred, model.tar_prob_pred, [ld_tar.test], True)
            print("[%6s] ECE_aware = %4.2f%%, ECE_aware_over = %4.2f%%"%(
                exp_name_i, ECE_aware*100.0, ECE_aware_over*100.0))
            print()

            # plot the reliability diagram
            fig_fn = os.path.join(params.snapshot_root, "plot", exp_name_i, "rel_diag.png")
            os.makedirs(os.path.dirname(fig_fn), exist_ok=True)
            CalibrationError_L1().plot_reliablity_diagram(fig_fn, model_S.label_pred, model.tar_prob_pred, [ld_tar.test])
    
        ## accumulate results
        ECEs_aware.append(ECE_aware.cpu().view(1))
        ECEs_aware_over.append(ECE_aware_over.cpu().view(1))
        print("[exp: %d/%d, %s, %f sec.] "
              "mean(ECEs_aware) = %4.2f%%, std(ECEs_aware) = %4.2f%%, "
              "mean(ECEs_aware_over) = %4.2f%%, std(ECEs_aware_over) = %4.2f%%"%(
                  i_exp+1, n_exps, exp_name, time.time() - t_cur,
                  tc.cat(ECEs_aware).mean()*100.0, tc.cat(ECEs_aware).std()*100.0,
                  tc.cat(ECEs_aware_over).mean()*100.0, tc.cat(ECEs_aware_over).std()*100.0))

        ## overead analysis
        print("[learning overhead analysis]")
        if hasattr(calibrator, "train_DF_time"):
            best_time = calibrator.train_DF_time['best_time'] - calibrator.train_DF_time['start_time']
            end_time = calibrator.train_DF_time['end_time'] - calibrator.train_DF_time['start_time']
            best_time_train_DF.append(best_time)
            end_time_train_DF.append(end_time)
            print("train_DF time (best/end): %f +- %f sec./ %f +- %f sec."%(
                T(best_time_train_DF).mean(), T(best_time_train_DF).std(),
                T(end_time_train_DF).mean(), T(end_time_train_DF).std()))

        if hasattr(calibrator, "train_D_time"):
            best_time = calibrator.train_D_time['best_time'] - calibrator.train_D_time['start_time']
            end_time = calibrator.train_D_time['end_time'] - calibrator.train_D_time['start_time']
            best_time_train_D.append(best_time)
            end_time_train_D.append(end_time)
            print("train_D time (best/end): %f +- %f sec./ %f +- %f sec."%(
                T(best_time_train_D).mean(), T(best_time_train_D).std(),
                T(end_time_train_D).mean(), T(end_time_train_D).std()))

        if hasattr(calibrator, "cal_D_time"):
            best_time = calibrator.cal_D_time['best_time'] - calibrator.cal_D_time['start_time']
            end_time = calibrator.cal_D_time['end_time'] - calibrator.cal_D_time['start_time']
            best_time_cal_D.append(best_time)
            end_time_cal_D.append(end_time)
            print("cal_D time (best/end): %f +- %f sec./ %f +- %f sec."%(
                T(best_time_cal_D).mean(), T(best_time_cal_D).std(),
                T(end_time_cal_D).mean(), T(end_time_cal_D).std()))

        if hasattr(calibrator, "cal_F_time"):
            best_time = calibrator.cal_F_time['best_time'] - calibrator.cal_F_time['start_time']
            end_time = calibrator.cal_F_time['end_time'] - calibrator.cal_F_time['start_time']
            best_time_cal_F.append(best_time)
            end_time_cal_F.append(end_time)
            print("cal_F time (best/end): %f +- %f sec./ %f +- %f sec."%(
                T(best_time_cal_F).mean(), T(best_time_cal_F).std(),
                T(end_time_cal_F).mean(), T(end_time_cal_F).std()))

        print()

    
    
    
    
