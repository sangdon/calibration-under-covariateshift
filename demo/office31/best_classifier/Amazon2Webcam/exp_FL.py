import os, sys
##--------------------------------------------------
from exp_Amazon2Webcam import exp
sys.path.append("../../")
sys.path.append("../../../../../")
## param
from params import FL_AmazonParamParser as ParamParser
## model
from train_model import train_model as netS
from models.DAForecasters import SimpleDiscriminatorNet as netD
from models.DAForecasters import SimpleFNNForecaster as netF
from models.DAForecasters import DAForecaster_FL as DAF
model_init_fn = lambda params: DAF(netS(False), 
                                   netD(params.F_n_hiddens, params.D_n_hiddens), 
                                   netF(params.n_features, params.F_n_hiddens, params.n_labels))
## algorithm
from calibration.DA import FL as DACalibrator
##--------------------------------------------------

if __name__ == "__main__":
    ## meta
    exp_name = os.path.splitext(os.path.basename(__file__))[0]
    ## run exps
    exp(exp_name, ParamParser, model_init_fn, netS, DACalibrator)
    
