import os, sys
import torch as tc
import torch.tensor as T
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


##--------------------------------------------------
from exp_MNIST2USPS import exp
sys.path.append("../../")
sys.path.append("../../../../../")
## param
from params import Temp_FL_IW_MNISTParamParser as ParamParser
## model
from train_model import train_model as netS
from models.DAForecasters import SimpleDiscriminatorNet as netD
from models.DAForecasters import SimpleFNNForecaster as netF
from models.DAForecasters import DAForecaster_Temp_FL_IW as DAF
model_init_fn = lambda params: DAF(netS(False), 
                                   netD(params.F_n_hiddens, params.D_n_hiddens), 
                                   netF(params.n_features, params.F_n_hiddens, params.n_labels))
## algorithm
from calibration.DA import Temp_FL_IW as DACalibrator
##--------------------------------------------------

##--------------------------------------------------
sys.path.append("../../../../../")
from models.DAForecasters import DAForecaster as DAF
from calibration.utils import *
from data.digits import loadUSPS
from data.digits import loadMNIST
loadSrc = lambda batch_size: loadMNIST('datasets/MNIST', batch_size, train_shuffle=False, val_shuffle=False)
loadTar = lambda batch_size: loadUSPS('datasets/USPS', batch_size, 32, gray=False, train_shuffle=False, val_shuffle=False)
##--------------------------------------------------


def plot_box(iws, save_root, font_size, n_vis=15):

    ## plot data
    iws = iws.detach().cpu()
    plot_fn = os.path.join(save_root, "plot_iws_box.png")

    ## chose top n_vis
    print("mean(IW):", iws.mean())
    ms = iws.mean(0)
    ms_sort, idx_sort = ms.sort(descending=True)
    iws = iws[:, idx_sort[:n_vis]]
    print(iws)
    print(iws.min())
    
    ## box plot
    #plt.rc("text", usetex=True)
    with PdfPages(plot_fn + '.pdf') as pdf: 
        plt.figure(1)
        plt.clf()
        plt.boxplot(iws.numpy(), whis=np.inf,
                    boxprops=dict(linewidth=3), medianprops=dict(linewidth=3.0))
        plt.xticks(fontsize=int(font_size*0.8))
        plt.yticks(fontsize=int(font_size*0.8))
        plt.grid(True)
        plt.xlabel("example index", fontsize=font_size)
        plt.ylabel("importance weight", fontsize=font_size)
        plt.savefig(plot_fn, bbox_inches='tight')
        pdf.savefig(bbox_inches='tight')

def plot_std(iws_tr, iws_val, save_root, font_size, N=1000):

    ## plot data
    plot_fn = os.path.join(save_root, "plot_iws_std.png")
    
    iws_tr = iws_tr.detach().cpu()
    iws_tr_std = iws_tr.std(0)
    iws_tr_std = iws_tr_std.sort(descending=True)[0]
    iws_tr_std = iws_tr_std[0:N]

    iws_val = iws_val.detach().cpu()
    iws_val_std = iws_val.std(0)
    iws_val_std = iws_val_std.sort(descending=True)[0]
    iws_val_std = iws_val_std[0:N]

    ## box plot
    #plt.rc("text", usetex=True)
    with PdfPages(plot_fn + '.pdf') as pdf: 
        plt.figure(1)
        plt.clf()
        h1 = plt.plot(iws_tr_std.numpy(), linewidth=3.0, label="train")
        h2 = plt.plot(iws_val_std.numpy(), linewidth=3.0, label="val")
        plt.xticks(fontsize=int(font_size*0.8))
        plt.yticks(fontsize=int(font_size*0.8))
        plt.grid(True)
        plt.xlabel("example index", fontsize=font_size)
        plt.ylabel("importance weight std.", fontsize=font_size)
        plt.legend(handles=[h1[0], h2[0]])
        plt.savefig(plot_fn, bbox_inches='tight')
        pdf.savefig(bbox_inches='tight')


if __name__ == "__main__":
    ## meta
    save_root = os.path.join("snapshots", os.path.splitext(os.path.basename(__file__))[0])
    exp_name = "exp_Temp_FL_IW"
    n_exps = 10
    model_S_fn = netS
    iws_fn = os.path.join(save_root, "iws.pk")
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    
    ## load models and compute iws
    if not os.path.exists(iws_fn):
        iws_exps_tr = []
        iws_exps_val = []
        iws_exps_te = []
        for i_exp in range(n_exps):
            t_cur = time.time()
            exp_name_i = "%s_%d"%(exp_name, i_exp)
        
            ## parameters
            params = ParamParser().read_params_static(exp_name_i)

            ## datasets
            ld_src = loadSrc(params.batch_size)
            ld_tar = loadTar(params.batch_size)
        
            ## model
            model_S = model_S_fn(False)
            model = model_init_fn(params)
        
            ## calibration
            calibrator = DACalibrator(params, model)
            calibrator.train(ld_src.train, ld_src.val, ld_tar.train, ld_tar.val)

            ## extract importance weights on train
            model.eval()
            iws_i = []
            for xs, ys in ld_src.train:
                xs = xs.to(calibrator.device)
                iws = model.importance_weight(xs, U_importance=params.U_importance)
                iws_i.append(iws.unsqueeze(1))
            iws_i = tc.cat(iws_i, 0)
            iws_exps_tr.append(iws_i)

            ## extract importance weights on val
            model.eval()
            iws_i = []
            for xs, ys in ld_src.val:
                xs = xs.to(calibrator.device)
                iws = model.importance_weight(xs, U_importance=params.U_importance)
                iws_i.append(iws.unsqueeze(1))
            iws_i = tc.cat(iws_i, 0)
            iws_exps_val.append(iws_i)

            ## extract importance weights on test
            model.eval()
            iws_i = []
            for xs, ys in ld_src.test:
                xs = xs.to(calibrator.device)
                iws = model.importance_weight(xs, U_importance=params.U_importance)
                iws_i.append(iws.unsqueeze(1))
            iws_i = tc.cat(iws_i, 0)
            iws_exps_te.append(iws_i)

        iws_exps_tr = tc.cat(iws_exps_tr, 1)
        iws_exps_val = tc.cat(iws_exps_val, 1)
        iws_exps_te = tc.cat(iws_exps_te, 1)
        
        # save
        pickle.dump((iws_exps_tr, iws_exps_val, iws_exps_te), open(iws_fn, "wb"))
    else:
        iws_exps_tr, iws_exps_val, iws_exps_te = pickle.load(open(iws_fn, "rb"))
    
    ## print and plot
    iws_exps_tr = iws_exps_tr.t() + 0.5 - 1.0 # get w(x)
    iws_exps_val = iws_exps_val.t() + 0.5 - 1.0 # get w(x)
    iws_exps_te = iws_exps_te.t() + 0.5 - 1.0 # get w(x)

    print(iws_exps_tr.size())

    ## sumary
    iws_exps_tr = iws_exps_tr.cpu().detach()
    iws_exps_val = iws_exps_val.cpu().detach()
    iws_exps_te = iws_exps_te.cpu().detach()

    #for iws in iws_exps.t():
    #    Q1 = iws.kthvalue(int(np.round(iws.size(0)*0.25)))[0]
    #    Q2 = iws.median()
    #    Q3 = iws.kthvalue(int(np.round(iws.size(0)*0.75)))[0]
    #    print("[summary] min = %f, 1st-Q = %f, median = %f, 3rd-Q = %f, max = %f, mean = %f "%(
    #        iws.min(), Q1, Q2, Q3, iws.max(), iws.mean()))
    
    ## plot
    #plot_std(iws_exps_tr, iws_exps_val, save_root, 20)
    plot_box(iws_exps_te, save_root, font_size=20)

    
        
