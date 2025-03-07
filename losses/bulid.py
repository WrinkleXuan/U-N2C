import torch
import torch.nn as nn
from .losses import NuclearLossFunc,AdversarialLoss,CovarianceLossFunc,EntropyLossFunc,KLLossFunc,NegativeLikelihooldLossFunc,PerceptualLossFunc,InfoNCE


from timm.utils import AverageMeter 
def build_criterion(config):
    criterion={}
    for k,v in  config.TRAIN.LOSS.items():
        if type(v) is list:
            if v[0]=='l1' and v[1]!=0:
                criterion[k]=nn.L1Loss()
            elif v[0]=='l2' and v[1]!=0:
                criterion[k]=nn.MSELoss()
            elif v[0]=='kl' and v[1]!=0:
                #criterion[k]=nn.KLDivLoss()
                criterion[k]=KLLossFunc()
                
            elif v[0]=='nlf' and v[1]!=0:
                
                criterion[k]=NuclearLossFunc()
            elif v[0]=='cov' and v[1]!=0:
                criterion[k]=CovarianceLossFunc()
            elif v[0]=='entropy' and v[1]!=0:
                #criterion[k]=nn.CrossEntropyLoss()
                criterion[k]=EntropyLossFunc()
            elif v[0]=='nll' and v[1]!=0:
                
                criterion[k]=NegativeLikelihooldLossFunc()
            elif v[0]=='perceptual' and v[1]!=0:
                criterion[k]=PerceptualLossFunc()
            elif v[0]=='InfoNce' and v[1]!=0:
                criterion[k]=InfoNCE()
            elif v[0]=='contrastive' and v[1]!=0:
                 criterion[k]=None
            elif v[0]=='dl' and v[1]!=0:
                criterion[k]=AdversarialLoss(config.TRAIN.LOSS.GAN_TYPE).to(config.LOCAL_RANK)
            elif v[0]=='dh' and v[1]!=0:
                criterion[k]=AdversarialLoss(config.TRAIN.LOSS.GAN_TYPE).to(config.LOCAL_RANK)
            
    return criterion       


def build_loss_meter(criterion):
    loss_meter={'total':AverageMeter()}
    loss={'total':None}
    for k in criterion.keys():
        if k != 'DH' and k!= 'DL':
            loss_meter[k]= AverageMeter()
            loss[k]=None
        else :
            loss_meter[k+'_G']=AverageMeter()
            loss_meter[k+'_D']=AverageMeter()

            loss[k+'_G']=None
            loss[k+'_D']=None
    return loss_meter,loss

def build_metric(config):
    metrics={}
    for k in config.METRICS:
        if not config.DATA.NOISE.ADD_NOISE and (k=='psnr' or k=='ssim'): continue

        metrics[k]=AverageMeter()
    return metrics