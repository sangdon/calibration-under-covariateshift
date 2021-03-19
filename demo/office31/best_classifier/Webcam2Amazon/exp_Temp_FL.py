import os, sys
##--------------------------------------------------
from exp_Webcam2Amazon import exp
sys.path.append("../../")
sys.path.append("../../../../../")
## param
from params import Temp_FL_WebcamParamParser as ParamParser
## model
from train_model import train_model as netS
from models.DAForecasters import SimpleDiscriminatorNet as netD
from models.DAForecasters import SimpleFNNForecaster as netF
from models.DAForecasters import DAForecaster_Temp_FL as DAF
model_init_fn = lambda params: DAF(netS(False), 
                                   netD(params.F_n_hiddens, params.D_n_hiddens), 
                                   netF(params.n_features, params.F_n_hiddens, params.n_labels))
## algorithm
from calibration.DA import Temp_FL as DACalibrator
##--------------------------------------------------

if __name__ == "__main__":
    ## meta
    exp_name = os.path.splitext(os.path.basename(__file__))[0]
    ## run exps
    exp(exp_name, ParamParser, model_init_fn, netS, DACalibrator)
    
