import os, sys
##--------------------------------------------------
from exp_USPS2MNIST import exp
sys.path.append("../../")
sys.path.append("../../../../../")
## param
from params import Temp_IW_USPSParamParser as ParamParser
## model
from train_model import train_model as netS
from models.DAForecasters import SimpleDiscriminatorNet as netD
from models.DAForecasters import ScalarForecaster as netF
from models.DAForecasters import DAForecaster_Temp_IW as DAF
model_init_fn = lambda params: DAF(netS(False), 
                                   netD(params.n_features, params.D_n_hiddens), 
                                   netF())
## algorithm
from calibration.DA import Temp_IW as DACalibrator
##--------------------------------------------------

if __name__ == "__main__":
    ## meta
    exp_name = os.path.splitext(os.path.basename(__file__))[0]
    ## run exps
    exp(exp_name, ParamParser, model_init_fn, netS, DACalibrator)
    
