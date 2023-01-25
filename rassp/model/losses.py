
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import scipy.stats
import time


class WeightedLoss(nn.Module):
    """
    Different weightings
    """
    def __init__(self, func='l2',
                 config = {},
                 log_true = False,
                 loss_pow = 1,
                 extra_loss_args= {}, 
                 **kwargs):
        super(WeightedLoss, self).__init__()

        self.swap_arg= False

        if func == 'kl':
            self.loss = nn.KLDivLoss(log_target=True, reduction='none')
        elif func == 'l2':
            self.loss = nn.MSELoss(reduction='none')
        elif func == 'l1':
            self.loss = nn.L1Loss(reduction='none')
        elif func == 'l1smooth':
            self.loss = nn.SmoothL1Loss(reduction='none', **extra_loss_args)
        elif func == 'subtract':
            self.loss = lambda x, y : y -x 

        self.config = config

        self.log_true = log_true

        self.pos = 0
        self.loss_pow  = loss_pow
        
    def __call__(self, res, spect, input_mask_t, **kwargs): 

        wc = self.config
        pred_spect = res['spect']
        SPECT_N = spect.shape[1]

        true_spect_np = spect.to('cpu').detach().numpy()
        if self.log_true:
            true_spect_np = np.exp(true_spect_np)

        if wc['kind'] == 'beta':
            
            w = np.clip(scipy.stats.beta.pdf(true_spect_np, wc['alpha'], wc['beta']) + wc.get('offset', 0.0), 
                        a_max=4, a_min=0)
        elif wc['kind'] == 'mass':
            w = np.linspace(0, 1, SPECT_N) ** wc.get('power', 1.0)
            w = w + wc.get('offset', 1.0)
            w = w / np.max(w)
        elif wc['kind'] == 'prob':
            w = true_spect_np ** wc.get('power', 1.0)
            w = w + wc.get('offset', 1.0)
        elif wc['kind'] == 'inv-prob':
            w = 1 - true_spect_np ** wc.get('power', 1.0)
            w = w + wc.get('offset', 1.0)
        elif wc['kind'] == 'prob_mass':
            w0 = true_spect_np ** wc.get('prob_power', 1.0)
            w0 = w0 + wc.get('prob_offset', 1.0)
            w1 = np.linspace(0, 1, SPECT_N) ** wc.get('mass_power', 1.0)
            w1 = w1 + wc.get('mass_offset', 1.0)

            w = w0 * w1 
            w = w + wc.get('offset', 0.0)
            w = w / np.max(w)
        
        elif wc['kind'] == 'sparse':
            
            is_zero = (true_spect_np < wc.get('zero_threshold', 1e-3)).astype(np.float32)
            w = is_zero 
            #w = np.linspace(0, 1, SPECT_N) ** wc.get('power', 1.0)
            w = w + wc.get('offset', 1.0)

        elif wc['kind'] == 'noop':
            w = np.ones_like(true_spect_np)

        w_t = torch.tensor(w.astype(np.float32)).to(spect.device)
        l = self.loss(pred_spect, spect)
        
        if self.loss_pow != 1:
            l = torch.pow(l, self.loss_pow)

        l_batch = torch.sum(w_t * l, dim=1)

        out = res.copy()
        out['true_spect'] = spect
        out['w_t'] = w_t
        out['l'] = l
        out['l_batch'] = l_batch


        self.pos += 1
        return l_batch.mean()


class WeightedMSELoss(nn.Module):
    def __init__(self, mass_scale=1.0,
                 intensity_pow=0.5, pred_eps = 1e-9,
                 **kwargs):
        super(WeightedMSELoss, self).__init__()

        self.mass_scale = mass_scale
        self.intensity_pow = intensity_pow
        self.loss = nn.MSELoss()
        self.pred_eps = pred_eps
        print("pred_eps=", pred_eps)

    def __call__(self, res, true_spect, input_mask_t, **kwargs): 
        SPECT_N = true_spect.shape[1]
        pred_spect = res['spect']


        w = torch.arange(SPECT_N).to(true_spect.device) * self.mass_scale

        eps = self.pred_eps
        
        pred_weighted = (pred_spect+eps)**self.intensity_pow
        pred_weighted_norm = torch.sqrt((pred_weighted**2).sum(dim=1)).unsqueeze(-1)
        
        true_weighted = (true_spect+eps)**self.intensity_pow
        true_weighted_norm = torch.sqrt((true_weighted**2).sum(dim=1)).unsqueeze(-1)

        pred_weighted_normed = pred_weighted / pred_weighted_norm
        true_weighted_normed = true_weighted / true_weighted_norm

        return self.loss(pred_weighted_normed, true_weighted_normed)
    
    
class CustomL1Loss(nn.Module):
    ## eqn from Adams paper
    def __init__(self, foo=None, **kwargs):
        super(CustomL1Loss, self).__init__()

        self.loss = nn.L1Loss()

    def __call__(self, res, true_spect, input_mask_t, **kwargs): 
        SPECT_N = true_spect.shape[1]
        pred_spect = res['spect']


        w = torch.arange(SPECT_N).to(true_spect.device)

        pred_weighted = (pred_spect+1e-9).sqrt() #* w.reshape(1, -1)
        pred_weighted_norm = torch.sqrt((pred_weighted**2).sum(dim=1)).unsqueeze(-1)
        
        true_weighted = (true_spect+1e-9).sqrt() #* w.reshape(1, -1)
        true_weighted_norm = torch.sqrt((true_weighted**2).sum(dim=1)).unsqueeze(-1)

        pred_weighted_normed = pred_weighted / pred_weighted_norm
        true_weighted_normed = true_weighted / true_weighted_norm

        return self.loss(pred_weighted_normed, true_weighted_normed)


class CustomWeightedMSELoss(nn.Module):
    def __init__(self, mass_scale=0.0,
                 intensity_pow=0.5,
                 pred_eps = 1e-9,
                 weight_scheme = None,
                 func = 'l2', 
                 loss_pow = 1, 
                 invalid_mass_weight = 0.0, 
                 **kwargs):
        super(CustomWeightedMSELoss, self).__init__()

        self.mass_scale = mass_scale
        self.intensity_pow = intensity_pow
        self.func = func
        if self.func == 'l1':
            self.loss = nn.L1Loss(reduce='none')
        elif self.func == 'smoothl1':
            self.loss = nn.SmoothL1Loss(reduce='none')
        elif self.func == 'bce':
            self.loss = nn.BCELoss(reduce='none')
        elif self.func == 'l2':
            self.loss = nn.MSELoss(reduce='none')
        else:
            raise ValueError(f"Unknown loss func {func}")
        self.pred_eps = pred_eps
        self.weight_scheme = weight_scheme
        self.invalid_mass_weight = invalid_mass_weight
        self.loss_pow = loss_pow

    def __call__(self, res, true_spect, input_mask_t, **kwargs): 
        SPECT_N = true_spect.shape[1]
        pred_spect = res['spect']

        mw = 1 + torch.arange(SPECT_N).to(true_spect.device) * self.mass_scale
        mw = mw.unsqueeze(0)

        w = 1

        
        eps = self.pred_eps
        
        pred_weighted = (pred_spect+eps)**self.intensity_pow * mw * w
        pred_weighted_norm = torch.sqrt((pred_weighted**2).sum(dim=1)).unsqueeze(-1)
        
        true_weighted = (true_spect+eps)**self.intensity_pow * mw * w
        
        true_weighted_norm = torch.sqrt((true_weighted**2).sum(dim=1)).unsqueeze(-1)

        pred_weighted_normed = pred_weighted / pred_weighted_norm
        true_weighted_normed = true_weighted / true_weighted_norm
        

        l = self.loss(pred_weighted_normed, true_weighted_normed)

        lw = 1+ (true_spect < 0.01).float() * self.invalid_mass_weight
        return ((l*lw)**self.loss_pow).mean()
