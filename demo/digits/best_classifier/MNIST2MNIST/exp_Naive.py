import os, sys
##--------------------------------------------------
from exp_MNIST2MNIST import exp
sys.path.append("../../")
sys.path.append("../../../../../")
## param
from params import Temp_MNISTParamParser as ParamParser
## model
from train_model import train_model as netS
from models.DAForecasters import NaiveForecaster as netF
from models.DAForecasters import DAForecaster_Naive as DAF
model_init_fn = lambda params: DAF(netS(False), netF())
## algorithm
from calibration.DA import Naive as DACalibrator
##--------------------------------------------------

if __name__ == "__main__":
    ## meta
    exp_name = os.path.splitext(os.path.basename(__file__))[0]
    ## run exps
    exp(exp_name, ParamParser, model_init_fn, netS, DACalibrator)
    
