import os, sys
import numpy as np
import time 

import torch as tc
import torch.tensor as T
from torch import nn, optim

sys.path.append("../")
from .calibrator import BaseCalibrator
from .utils import *
from classification.utils import *
    
class DACalibator(BaseCalibrator):
    def __init__(self, params, model):
        super().__init__(params, model)
        
        # init model_save_root                        
        self.params.model_save_root = os.path.join(
            self.params.snapshot_root, self.params.exp_name)
        if not os.path.exists(self.params.model_save_root):
            os.makedirs(self.params.model_save_root)
        
        if params.use_gpu:
            self.device = tc.device("cuda:0")
        else:
            self.device = tc.device("cpu")

        ## init fns
        self.model_train_DF_final_fn = os.path.join(
            self.params.model_save_root, "model_train_DF_final.pk")
        self.model_train_DF_best_fn = os.path.join(
            self.params.model_save_root, "model_train_DF_best.pk")
        
        self.model_train_D_final_fn = os.path.join(
            self.params.model_save_root, "model_train_D_final.pk")
        self.model_train_D_best_fn = os.path.join(
            self.params.model_save_root, "model_train_D_best.pk")
        
        self.model_cal_D_final_fn = os.path.join(
            self.params.model_save_root, "model_cal_D_final.pk")
        self.model_cal_D_best_fn = os.path.join(
            self.params.model_save_root, "model_cal_D_best.pk")
        
        self.model_cal_F_final_fn = os.path.join(
            self.params.model_save_root, "model_cal_F_final.pk")
        self.model_cal_F_best_fn = os.path.join(
            self.params.model_save_root, "model_cal_F_best.pk")
        
    def loss_D(self, xs_src, xs_tar, reduction='mean'):
        if self.params.loss_D_type == "L2":
            loss_D_fn = nn.MSELoss(reduction=reduction)
        elif self.params.loss_D_type == "BCE":
            loss_D_fn = nn.BCELoss(reduction=reduction)
        else:
            raise NotImplementedError
            
        ys_src = tc.ones(xs_src.size(0), device=self.device)
        ys_tar = tc.zeros(xs_tar.size(0), device=self.device)
        
        ## compute loss
        xs = tc.cat((xs_src, xs_tar), 0)
        ys = tc.cat((ys_src, ys_tar)).unsqueeze(1)
        fhs = self.model.prob_D(xs)
        loss = loss_D_fn(fhs, ys)
        return loss
    
    def loss_F(self, xs_src, ys_src, ws=T(1.5), reduction='mean'):
        if self.params.loss_F_type == "L2":
            if reduction=='mean':
                loss_F_fn = lambda fhs, ys, ws: (fhs - ys.float()).pow(2).sum(1).mul(ws).mean()
            elif reduction=='sum':
                loss_F_fn = lambda fhs, ys, ws: (fhs - ys.float()).pow(2).sum(1).mul(ws).sum()
            elif reduction=='none':
                loss_F_fn = lambda fhs, ys, ws: (fhs - ys.float()).pow(2).sum(1).mul(ws)
            else:
                raise NotImplementedError
            ys_src = one_hot(ys_src, self.params.n_labels).to(self.device)
        elif self.params.loss_F_type == "CE":
            loss_F_fn = nn.CrossEntropyLoss(reduction=reduction)
        else:
            raise NotImplemenetedError
            
        ## compute loss
        if self.params.loss_F_type == "L2":
            fhs = self.model.prob_F(xs_src)
        elif self.params.loss_F_type == "CE":
            fhs = self.model.forward_F(xs_src)
        else:
            raise NotImplemenetedError
        loss = loss_F_fn(fhs, ys_src, ws)
        return loss
    
    def train_DF(self, ld_src_tr, ld_src_val, ld_tar_tr, ld_tar_val, cross_validate_fn=None):
        ## load 
        if os.path.exists(self.model_train_DF_final_fn):
            self.model.load(self.model_train_DF_best_fn)
            self.model.eval()
            return
        
        ## init optimizers
        if self.params.optimizer_train == "Adam":
            optimD = optim.Adam(self.model.train_parameters_D(), lr=self.params.lr_D_train, 
                                weight_decay=self.params.weight_decay_train_DF)
            optimF = optim.Adam(self.model.train_parameters_F(), lr=self.params.lr_F_train, 
                                weight_decay=self.params.weight_decay_train_DF)
        elif self.params.optimizer_train == "SGD":
            optimD = optim.SGD(self.model.train_parameters_D(), lr=self.params.lr_D_train, 
                               weight_decay=self.params.weight_decay_train_DF, 
                               momentum=self.params.momentum_train)
            optimF = optim.SGD(self.model.train_parameters_F(), lr=self.params.lr_F_train, 
                               weight_decay=self.params.weight_decay_train_DF, 
                               momentum=self.params.momentum_train)
        else:
            raise NotImplementedError
            
        ## scheduler
        schD = optim.lr_scheduler.StepLR(
            optimD, self.params.lr_decay_epoch_train, self.params.lr_decay_rate_train)    
        schF = optim.lr_scheduler.StepLR(
            optimF, self.params.lr_decay_epoch_train, self.params.lr_decay_rate_train)    

        ## train
        self.init_cross_validation()
        self.train_DF_time = {"start_time": time.time(), "best_time": None, "end_time": None}
        for i_epoch in range(self.params.n_epochs_train):
            ## init
            t_start = time.time()
            self.model.train_D()
            self.model.train_F()
            jld = JointLoader(ld_src_tr, ld_tar_tr)
            ## train for one epoch
            for (xs_src, ys_src, xs_tar, _) in jld:   
                ## training    
                xs_src = xs_src.to(self.device)
                ys_src = ys_src.to(self.device)
                xs_tar = xs_tar.to(self.device)
                
                ## init grads
                optimD.zero_grad()
                optimF.zero_grad()

                ## compute loss
                loss_D = self.loss_D(xs_src, xs_tar)
                loss_F = self.loss_F(xs_src, ys_src)
                loss = loss_D*self.params.lambda_D + loss_F
                                                                
                ## update a model
                loss.backward()
                optimD.step()
                optimF.step()

            ## update schedulers
            schD.step()
            schF.step()
                
            ## output status
            print("[%s, %4d/%4d, lr_D = %f, ld_F = %f, %f sec.] loss_D = %f, loss_F = %f"%(
                      "train_DF", i_epoch+1, self.params.n_epochs_train, 
                      optimD.param_groups[0]['lr'], optimF.param_groups[0]['lr'],
                      time.time() - t_start, loss_D, loss_F))
            
            ## eval
            if (i_epoch+1)%self.params.n_epochs_eval == 0:
                ## compute test errors, just for reference, not for model selection
                self.test([ld_src_val, ld_tar_val], ["src_val", "tar_val"])
                
            ## cross_validate and save the best model
            if self.params.train_DF_cross_val and cross_validate_fn is not None and (i_epoch+1)%self.params.n_epochs_eval == 0:
                
                if self.cross_validate(i_epoch, cross_validate_fn, self.model_train_DF_best_fn):
                    self.train_DF_time['best_time'] = time.time()
                    
                if self.early_stop(i_epoch, self.params.early_stop_cri_train):
                    break
                
            ## loss-based early stop
            if self.early_stop_train_DF_loss(i_epoch, ld_src_tr, ld_tar_tr):
                break
            
        ## save the end time
        self.train_DF_time['end_time'] = time.time()
        ## save the final model
        self.model.eval()
        self.model.save(self.model_train_DF_final_fn)
        ## load the best model
        self.model.load(self.model_train_DF_best_fn)
        ## evaluate just for reference
        self.test([ld_src_val, ld_tar_val], ["src_val", "tar_val"])
        print()
    
    def train_D(self, ld_src_tr, ld_src_val, ld_tar_tr, ld_tar_val, cross_validate_fn=None):
        ## load 
        if os.path.exists(self.model_train_D_final_fn):
            self.model.load(self.model_train_D_best_fn)
            self.model.eval()
            return
        
        ## init optimizers
        if self.params.optimizer_train == "Adam":
            optimD = optim.Adam(self.model.train_parameters_D(), lr=self.params.lr_D_train, 
                                weight_decay=self.params.weight_decay_train_D)
        if self.params.optimizer_train == "SGD":
            optimD = optim.SGD(self.model.train_parameters_D(), lr=self.params.lr_D_train, 
                               weight_decay=self.params.weight_decay_train_D, 
                               momentum=self.params.momentum_train)
        else:
            raise NotImplementedError
            
        ## scheduler
        schD = optim.lr_scheduler.StepLR(
            optimD, self.params.lr_decay_epoch_train, self.params.lr_decay_rate_train)    
        
        ## train
        self.init_cross_validation()
        self.train_D_time = {"start_time": time.time(), "best_time": None, "end_time": None}
        for i_epoch in range(self.params.n_epochs_train):
            ## init
            t_start = time.time()
            self.model.train_D()
            jld = JointLoader(ld_src_tr, ld_tar_tr)
            ## train for one epoch
            for (xs_src, _, xs_tar, _) in jld:
                ## training    
                xs_src = xs_src.to(self.device)
                xs_tar = xs_tar.to(self.device)
                
                ## init grads
                optimD.zero_grad()
                
                ## compute loss
                loss_D = self.loss_D(xs_src, xs_tar)
                                                                
                ## update a model
                loss_D.backward()
                optimD.step()
                
            ## update schedulers
            schD.step()
                
            ## output status
            print("[%s, %4d/%4d, lr_D = %f, %f sec.] loss_D = %f"%(
                "train_D", i_epoch+1, self.params.n_epochs_train, 
                optimD.param_groups[0]['lr'], time.time() - t_start, loss_D))
            
            ## cross_validate and save the best model
            if cross_validate_fn is not None and (i_epoch+1)%self.params.n_epochs_eval == 0: 
                if self.cross_validate(i_epoch, cross_validate_fn, self.model_train_D_best_fn):
                    self.train_D_time['best_time'] = time.time()
                    
                if self.early_stop(i_epoch, self.params.early_stop_cri_train):
                    break

        ## save the end time
        self.train_D_time['end_time'] = time.time()
        ## save the final model
        self.model.eval()
        self.model.save(self.model_train_D_final_fn)
        ## load the best model
        self.model.load(self.model_train_D_best_fn)
        ## evaluate just for reference
        self.test([ld_src_val, ld_tar_val], ["src_val", "tar_val"])
        print()
    
    def cal_D(self, ld_src_val, ld_tar_val, cross_validate_fn=None):
        ## load 
        if os.path.exists(self.model_cal_D_final_fn):
            self.model.load(self.model_cal_D_best_fn)
            self.model.eval()
            return
        
        ## init optimizers
        if self.params.optimizer_cal == "Adam":
            optimD = optim.Adam(self.model.cal_parameters_D(), lr=self.params.lr_D_cal, 
                                weight_decay=self.params.weight_decay_cal)
        if self.params.optimizer_cal == "SGD":
            optimD = optim.SGD(self.model.cal_parameters_D(), lr=self.params.lr_D_cal, 
                               weight_decay=self.params.weight_decay_cal, 
                               momentum=self.params.momentum_cal)
        else:
            raise NotImplementedError
            
        ## scheduler
        schD = optim.lr_scheduler.StepLR(
            optimD, self.params.lr_decay_epoch_cal, self.params.lr_decay_rate_cal)
        
        ## train
        self.init_cross_validation()
        self.cal_D_time = {"start_time": time.time(), "best_time": None, "end_time": None}
        for i_epoch in range(self.params.n_epochs_cal):
            ## init
            t_start = time.time()
            self.model.cal_D()
            jld = JointLoader(ld_src_val, ld_tar_val)
            ## train for one epoch
            for (xs_src, _, xs_tar, _) in jld:   
                ## training    
                xs_src = xs_src.to(self.device)
                xs_tar = xs_tar.to(self.device)
                
                ## init grads
                optimD.zero_grad()
                
                ## compute loss
                loss_D = self.loss_D(xs_src, xs_tar)
                                                                
                ## update a model
                loss_D.backward()
                optimD.step()
                
            ## update schedulers
            schD.step()
            
            ## output status
            print("[%s, %4d/%4d, lr_D = %f, %f sec.] T = %f, loss_D = %f"%(
                "cal_D", i_epoch+1, self.params.n_epochs_cal, 
                optimD.param_groups[0]['lr'], time.time() - t_start, 
                self.model.cal_parameters_D()[0].item(), loss_D))
            
            ## cross_validate and save the best model
            if cross_validate_fn is not None and (i_epoch+1)%self.params.n_epochs_eval == 0: 
                if self.cross_validate(i_epoch, cross_validate_fn, self.model_cal_D_best_fn):
                    self.cal_D_time['best_time'] = time.time()
                    
                if self.early_stop(i_epoch, self.params.early_stop_cri_cal):
                    break

        ## save the end time
        self.cal_D_time['end_time'] = time.time()
        ## save the final model
        self.model.eval()
        self.model.save(self.model_cal_D_final_fn)
        ## load the best model
        self.model.load(self.model_cal_D_best_fn)
        ## evaluate just for reference
        self.test([ld_src_val, ld_tar_val], ["src_val", "tar_val"])
        print()
        
    def cal_F(self, ld_src_val, ld_tar_val, cross_validate_fn=None):
        ## load 
        if os.path.exists(self.model_cal_F_final_fn):
            self.model.load(self.model_cal_F_best_fn)
            self.model.eval()
            return
        
        ## init optimizers
        if self.params.optimizer_cal == "Adam":
            optimF = optim.Adam(self.model.cal_parameters_F(), lr=self.params.lr_F_cal, 
                                weight_decay=self.params.weight_decay_cal)
        if self.params.optimizer_cal == "SGD":
            optimF = optim.SGD(self.model.cal_parameters_F(), lr=self.params.lr_F_cal, 
                               weight_decay=self.params.weight_decay_cal, 
                               momentum=self.params.momentum_cal)
        else:
            raise NotImplementedError
            
        ## scheduler
        schF = optim.lr_scheduler.StepLR(
            optimF, self.params.lr_decay_epoch_cal, self.params.lr_decay_rate_cal)
        
        ## train
        self.init_cross_validation()
        self.cal_F_time = {"start_time": time.time(), "best_time": None, "end_time": None}
        for i_epoch in range(self.params.n_epochs_cal):
            ## init
            t_start = time.time()
            self.model.cal_F()
            ## train for one epoch
            for xs_src, ys_src in ld_src_val:   
                ## training    
                xs_src = xs_src.to(self.device)
                ys_src = ys_src.to(self.device)
                
                ## init grads
                optimF.zero_grad()
                
                ## compute loss
                iws, ghs = self.model.importance_weight(xs_src, self.params.U_importance, True)
                loss_F = self.loss_F(xs_src, ys_src, iws)
                                                                
                ## update a model
                loss_F.backward()
                optimF.step()
                
            ## update schedulers
            schF.step()
            
            ## output status
            print("[%s, %4d/%4d, ld_F = %f, %f sec.] "
                  "gh_min = %f, gh_max = %f, gh_mean = %f, T = %f, loss_F = %f"%(
                      "cal_F", i_epoch+1, self.params.n_epochs_cal, 
                      optimF.param_groups[0]['lr'], time.time() - t_start, 
                      ghs.min(), ghs.max(), ghs.mean(), 
                      self.model.cal_parameters_F()[0].item(), loss_F))
                        
            ## cross_validate and save the best model
            if cross_validate_fn is not None and (i_epoch+1)%self.params.n_epochs_eval == 0: 
                if self.cross_validate(i_epoch, cross_validate_fn, self.model_cal_F_best_fn):
                    self.cal_F_time['best_time'] = time.time()
                    
                if self.early_stop(i_epoch, self.params.early_stop_cri_cal):
                    break

        ## save the end time
        self.cal_F_time['end_time'] = time.time()
        ## save the final model
        self.model.eval()
        self.model.save(self.model_cal_F_final_fn)
        ## load the best model
        self.model.load(self.model_cal_F_best_fn)
        ## evaluate just for reference
        self.test([ld_src_val, ld_tar_val], ["src_val", "tar_val"])
        print()
    
    def train(self, ld_src_tr, ld_src_val, ld_tar_tr, ld_tar_val):
        ## 1. train a source-discriminator and forecaster joinly to learn a indistingushible feature
        cv_fn_train_DF = lambda i_epoch: self.cross_validate_fn_train_DF(ld_src_tr, ld_src_val, ld_tar_tr, ld_tar_val, i_epoch)
        self.train_DF(ld_src_tr, ld_src_val, ld_tar_tr, ld_tar_val, cv_fn_train_DF)
        
        ## 2. train a source-discriminator for IW
        cv_fn_train_D = lambda i_epoch: self.cross_validate_fn_train_D(ld_src_val, ld_tar_val, i_epoch)
        self.train_D(ld_src_tr, ld_src_val, ld_tar_tr, ld_tar_val, cv_fn_train_D)
        
        ## 3. calibrate the source-discriminator
        cv_fn_cal_D = lambda i_epoch: self.cross_validate_fn_cal_D(ld_src_val, ld_tar_val, i_epoch)
        self.cal_D(ld_src_val, ld_tar_val, cv_fn_cal_D)
        
        ## 4. calibrate the forecaster with IW
        cv_fn_cal_F = lambda i_epoch: self.cross_validate_fn_cal_F(ld_src_val, i_epoch)
        self.cal_F(ld_src_val, ld_tar_val, cv_fn_cal_F)
        
    def cross_validate_fn_train_DF(self, ld_src_tr, ld_src_val, ld_tar_tr, ld_tar_val, i_epoch=None):
        self.model.eval()
        loss_sum = 0.0
        n_total = 0.0
        with tc.no_grad():
            # maintain the same seed such that a ramdom loader produce the same order
            # estimate importance weight                                                
            tc.manual_seed(0)
            p_lsexp = self.kde_gaussian(ld_src_tr, ld_src_val, n_tr_samples_ratio=self.params.ratio_n_tr_FL_cross_val) # compute over src_val
            tc.manual_seed(0)
            q_lsexp = self.kde_gaussian(ld_tar_tr, ld_src_val, n_tr_samples_ratio=self.params.ratio_n_tr_FL_cross_val) # compute over src_val again
            ws = (q_lsexp - p_lsexp).clamp(T(0.0), T(self.params.U_importance).log()).exp()
            tc.manual_seed(0)
            i_ws = 0
            for xs_src, ys_src in ld_src_val:
                ws_i = ws[i_ws:i_ws+xs_src.size(0)]
                xs_src = xs_src.to(self.device)
                ys_src = ys_src.to(self.device)
                
                # compute cross-validation loss
                loss = self.loss_F(xs_src, ys_src, ws_i, reduction='none')
                # accumulate results          
                loss_sum += loss.sum()
                n_total += float(xs_src.size(0))
                i_ws += xs_src.size(0)

            ## set a random seed
            tc.manual_seed(time.time())
        return loss_sum/n_total

    def cross_validate_fn_train_D(self, ld_src_val, ld_tar_val, i_epoch=None):
        self.model.eval()
        return self.loss_D_all(ld_src_val, ld_tar_val)
    
    def cross_validate_fn_cal_D(self, ld_src_val, ld_tar_val, i_epoch=None):
        self.model.eval()
        return self.loss_D_all(ld_src_val, ld_tar_val)
    
    def cross_validate_fn_cal_F(self, ld_src_val, i_epoch=None):
        self.model.eval()
        return self.loss_F_all(ld_src_val)
    
    def kde_gaussian(self, ld_train, ld_val, n_tr_samples_ratio=0.2, n_val_samples_ratio=1.0, h=1e1):
        # assume constants are canceled out later and use the same h for kde for p and q
        # compute: s(x, x') = -||x - x'||^2 / 2h^2 for x' in train and x in val 
        n_tr_samples = len(ld_train)*ld_train.batch_size * n_tr_samples_ratio
        n_val_samples = len(ld_val)*ld_val.batch_size * n_val_samples_ratio

        lsexps_val = []
        n_val = 0
        #for xs_val, _ in ld_val:
        for xs_val, ys_val in ld_val:
            xs_val = xs_val.to(self.device)
            zs_val = self.model.feature_S(xs_val)
            sims_tr = []
            n_tr = 0
            for xs_train, _ in ld_train:
                xs_train = xs_train.to(self.device)
                zs_train = self.model.feature_S(xs_train)

                diff = zs_train.unsqueeze(1) - zs_val.unsqueeze(0)
                dist = diff.pow(2.0).sum(-1)
                sim = -0.5 * dist / h**2
                sims_tr.append(sim)
                n_tr += zs_train.size(0)
                if n_tr >= n_tr_samples: # the sampled training set may differ for each xs_val
                    break
            sims_tr = tc.cat(sims_tr, 0)
            sims_tr -= T(n_tr).float().log()
            lsexp = sims_tr.logsumexp(0)
            lsexps_val.append(lsexp)
            n_val += xs_val.size(0)
            if n_val >= n_val_samples:
                break
        lsexps_val = tc.cat(lsexps_val, 0)
        return lsexps_val

    
    def loss_D_all(self, ld_src_val, ld_tar_val):
        loss_sum = 0.0
        n_total = 0.0
        jld = JointLoader(ld_src_val, ld_tar_val)
        for xs_src, _, xs_tar, _ in jld:
            xs_src = xs_src.to(self.device)
            xs_tar = xs_tar.to(self.device)
            with tc.no_grad():
                loss = self.loss_D(xs_src, xs_tar, reduction='none')
                loss_sum += loss.sum()
                n_total += float(xs_src.size(0))
        return loss_sum/n_total
    
    def loss_F_all(self, ld_src_val):
        loss_sum = 0.0
        n_total = 0.0
        for xs_src, ys_src in ld_src_val:
            xs_src = xs_src.to(self.device)
            ys_src = ys_src.to(self.device)
            with tc.no_grad():
                iws = self.model.importance_weight(xs_src, self.params.U_importance)
                loss = self.loss_F(xs_src, ys_src, iws, reduction='sum')
                loss_sum += loss
                n_total += float(xs_src.size(0))
        return loss_sum/n_total
        
    def init_cross_validation(self):
        self.cv_loss_best = np.inf
        self.cv_loss_best_epoch = -np.inf
            
    def cross_validate(self, i_epoch, cross_validate_fn, model_best_fn):
        t_cur = time.time()
        ## cross-valiate
        self.cv_loss_current = cross_validate_fn(i_epoch)
        print("[%f sec.] cv_loss_current = %f, cv_loss_best = %f"%(
            time.time()-t_cur, self.cv_loss_current, self.cv_loss_best))
        ## save the best model
        if self.cv_loss_current < self.cv_loss_best:
            self.cv_loss_best = self.cv_loss_current
            self.cv_loss_best_epoch = i_epoch
            print("[save] the best cv_loss = %f"%(self.cv_loss_best))
            self.model.save(model_best_fn)
            return True
        else:
            return False
        
                
    def early_stop(self, i_epoch, early_stop_cri):
        if i_epoch - self.cv_loss_best_epoch > self.params.n_epochs_train*early_stop_cri:
            print("[early stop] (current epoch) - (best epoch) >> "
                  "%f * (total epoch)"%(early_stop_cri))
            return True
        else:
            return False

    def early_stop_train_DF_loss(self, i_epoch, ld_src_val, ld_tar_val):
        self.model.eval()
        loss_D = self.loss_D_all(ld_src_val, ld_tar_val)
        if loss_D < self.params.train_DF_loss_D_min:
            print("[early stop] loss_D = %f < train_DF_loss_D_min = %f"%(loss_D, self.params.train_DF_loss_D_min))
            return True
        else:
            # save the current model as the best
            print("[save] the current model as the best (loss_D = %f)"%(loss_D))
            self.model.save(self.model_train_DF_best_fn)
            
            self.train_DF_time['best_time'] = time.time()
            return False

        
    def test(self, lds, ld_names=None, compute_loss=False):
        self.model.eval()
        if ld_names is not None:
            assert(len(lds) == len(ld_names))
                
        ## classification error
        with tc.no_grad():    
            for i, ld in enumerate(lds):
                ## label prediction error
                loss_fn = None
                if compute_loss:
                    loss_fn = lambda xs, ys: self.loss_F(
                        xs, ys, 
                        self.model.importance_weight(xs_src, self.params.U_importance), 
                        reduction='none')
                cls_error, n_error, n_total = compute_cls_error(
                    [ld], self.model, self.device, loss_fn=loss_fn)
                ## confidence prediction error
                ECE_L1_aware = CalibrationError_L1()(
                    self.model.model_S.label_pred, self.model.tar_prob_pred, [ld])
                ECE_L1_aware_over = CalibrationError_L1()(
                    self.model.model_S.label_pred, self.model.tar_prob_pred, [ld], 
                    measure_overconfidence=True)
                
                if ld_names is not None:
                    if compute_loss:
                        print("# %s: loss: %.6f, ECE_aware: %.2f%%, ECE_aware_over: %.2f%%"%(
                            ld_names[i], cls_error, ECE_L1_aware*100.0, ECE_L1_aware_over*100.0))
                    else:            
                        print("# %s: cls error: %d / %d = %.2f%%, "
                              "ECE_aware: %.2f%%, ECE_aware_over: %.2f%%"%(
                                  ld_names[i], n_error, n_total, cls_error*100.0, 
                                  ECE_L1_aware*100.0, ECE_L1_aware_over*100.0))

class Temp_FL_IW(DACalibator):
    def __init__(self, params, model):
        super().__init__(params, model)
        
class Temp_FL(DACalibator):
    def __init__(self, params, model):
        super().__init__(params, model)

    def train(self, ld_src_tr, ld_src_val, ld_tar_tr, ld_tar_val):
        ## 1. train a source-discriminator and forecaster joinly to learn a indistinguishible feature
        cv_fn_train_DF = lambda i_epoch: self.cross_validate_fn_train_DF(
            ld_src_tr, ld_src_val, ld_tar_tr, ld_tar_val, i_epoch)
        self.train_DF(ld_src_tr, ld_src_val, ld_tar_tr, ld_tar_val, cv_fn_train_DF)
        
        ## 2. calibrate the forecaster on top of the indistinguishible feature
        cv_fn_cal_F = lambda i_epoch: self.cross_validate_fn_cal_F(ld_src_val, i_epoch)
        self.cal_F(ld_src_val, ld_tar_val, cv_fn_cal_F)    

class FL(DACalibator):
    def __init__(self, params, model):
        super().__init__(params, model)

    def train(self, ld_src_tr, ld_src_val, ld_tar_tr, ld_tar_val):
        ## 1. train a source-discriminator and forecaster joinly to learn a indistinguishible feature
        cv_fn_train_DF = lambda i_epoch: self.cross_validate_fn_train_DF(
            ld_src_tr, ld_src_val, ld_tar_tr, ld_tar_val, i_epoch)
        self.train_DF(ld_src_tr, ld_src_val, ld_tar_tr, ld_tar_val, cv_fn_train_DF)
        
class Temp_IW(DACalibator):
    def __init__(self, params, model):
        super().__init__(params, model)

    def train(self, ld_src_tr, ld_src_val, ld_tar_tr, ld_tar_val):
        ## 1. train a source-discriminator for IW
        cv_fn_train_D = lambda i_epoch: self.cross_validate_fn_train_D(
            ld_src_val, ld_tar_val, i_epoch)
        self.train_D(ld_src_tr, ld_src_val, ld_tar_tr, ld_tar_val, cv_fn_train_D)
        
        ## 2. calibrate the source-discriminator
        cv_fn_cal_D = lambda i_epoch: self.cross_validate_fn_cal_D(ld_src_val, ld_tar_val, i_epoch)
        self.cal_D(ld_src_val, ld_tar_val, cv_fn_cal_D)
        
        ## 3. calibrate the forecaster with IW
        cv_fn_cal_F = lambda i_epoch: self.cross_validate_fn_cal_F(ld_src_val, i_epoch)
        self.cal_F(ld_src_val, ld_tar_val, cv_fn_cal_F)

class Temp(DACalibator):
    def __init__(self, params, model):
        super().__init__(params, model)

    def train(self, ld_src_tr, ld_src_val, ld_tar_tr, ld_tar_val):
        ## 1. calibrate the forecaster on top of the indistinguishible feature
        cv_fn_cal_F = lambda i_epoch: self.cross_validate_fn_cal_F(ld_src_val, i_epoch)
        self.cal_F(ld_src_val, ld_tar_val, cv_fn_cal_F)    

class Naive(DACalibator):
    def __init__(self, params, model):
        super().__init__(params, model)

    def train(self, ld_src_tr, ld_src_val, ld_tar_tr, ld_tar_val):
        pass
