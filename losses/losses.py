import sys
sys.path.append('losses/')
import types
import torch
import torch.nn as nn

import torch.autograd as autograd
import numpy as np
from torchvision import models
from torch.nn import functional as F
from einops import rearrange

from info_nce import info_nce
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.

    This class is adopted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class AdversarialLoss(nn.Module):
    def __init__(self, gan_mode='lsgan'):
        super().__init__()
        self.gan=GANLoss(gan_mode)

    def get_d_loss(self, pred_fake, pred_real):
        
        """ Get the loss that updates the discriminator
        """

        #device = get_device(self)
        #gan_loss = gan_loss.to(device)

        pred_fake = pred_fake.detach()

        loss_fake = self.gan(pred_fake, False)
        loss_real = self.gan(pred_real, True)

        loss = (loss_real + loss_fake) * 0.5
        return loss

    def get_g_loss(self, pred_fake,pred_real):
        """ Get the loss that updates the generator
        """

        

        #device = get_device(self)
        #gan_loss = gan_loss.to(device)

        loss = self.gan(pred_fake, True)

        return loss 

    def forward(self,pred_fake,pred_real=None):
        if pred_real is None:
            return self.get_g_loss(pred_fake,pred_real)
        else:
            return self.get_d_loss(pred_fake,pred_real)

    
class CriterionLowRank(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,inputs):
        list_svd,_ = torch.sort(torch.sqrt(torch.sum(torch.pow(inputs,2),dim=0)), descending=True)
        nums = min(inputs.shape[0],inputs.shape[1])
        loss = torch.sum(list_svd[:nums])
        return loss
    
class NuclearLossFunc(nn.Module):
    def __init__(self):
        super(NuclearLossFunc, self).__init__() 
        
    def forward(self,inputs):
        loss = torch.zeros([1])
        total_batch, total_channel, Sx, Sy = inputs.size()
        inputs = inputs.contiguous().view(-1,Sx,Sy) 
        inputs_trans = torch.transpose(inputs,1,2)
        m_total = torch.bmm(inputs_trans,inputs)
        loss = m_total.sum(0).trace()
        loss /= (total_batch*total_channel)
        return loss

class EntropyLossFunc(nn.Module):
    def __init__(self):
        super(EntropyLossFunc, self).__init__() 
        
    def forward(self,inputs):
        B,C,H,W=inputs.size()
        inputs=inputs.reshape(-1,C)
        mean=torch.mean(inputs,dim=1,keepdim=True)
        std=torch.std(inputs,dim=1,keepdim=True)
        inputs=(inputs-mean)/(std)
        inputs=torch.softmax(inputs,dim=1)
        max=torch.tensor(C,device=inputs.device)
        loss=torch.log(max)-(-torch.sum(inputs * torch.log(inputs),dim=1))
        #loss /= (B*H*W)
        #print(loss.size())
        loss=torch.mean(loss)
        print(loss)
        return loss

class KLLossFunc(nn.Module):
    def __init__(self):
        super(KLLossFunc, self).__init__() 
        self.loss_func=nn.KLDivLoss(reduce='mean')
    def forward(self,target,inputs):
        assert target.size()==inputs.size()
        B,C,H,W=inputs.size()
        target=target.reshape(-1,C)
        mean=torch.mean(target,dim=1,keepdim=True)
        std=torch.std(target,dim=1,keepdim=True)
        target=(target-mean)/(std)
        
        inputs=inputs.reshape(-1,C)
        mean=torch.mean(inputs,dim=1,keepdim=True)
        std=torch.std(inputs,dim=1,keepdim=True)
        inputs=(inputs-mean)/(std)
        
        inputs=torch.softmax(inputs,dim=1)
        target=torch.softmax(target,dim=1)
        loss=self.loss_func(inputs.log(),target)
        return loss
class CovarianceLossFunc(nn.Module):
    def __init__(self):
        super(CovarianceLossFunc, self).__init__() 
    
    def batch_cov(self,points):
        B, N, D = points.size()
        mean = points.mean(dim=1).unsqueeze(1)
        diffs = (points - mean).reshape(B * N, D)
        prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
        bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
        return bcov  # (B, D, D)
    
    def forward(self,inputs,delta=0.01):
        B,C,H,W=inputs.size()
        inputs=inputs.reshape(B,C,-1)
        mean=torch.mean(inputs,dim=1,keepdim=True)
        std=torch.std(inputs,dim=1,keepdim=True)
        inputs=(inputs-mean)/(std+1e-30)
        inputs=torch.softmax(inputs,dim=1)
        
        covariance_matrix=self.batch_cov(inputs)
        eye_matrix=torch.eye(H*W).unsqueeze(0).expand(B,-1,-1).to(covariance_matrix.device)

        covariance_matrix=covariance_matrix*(1-eye_matrix)
        #print(covariance_matrix[10])
        #covariance_matrix[torch.abs(covariance_matrix)>delta]=2*delta*torch.abs(covariance_matrix[torch.abs(covariance_matrix)>delta])-delta**2
        #covariance_matrix[torch.abs(covariance_matrix)<delta]**=2
        
        loss=(covariance_matrix**2).sum()
        #loss=(covariance_matrix.sum()-torch.diagonal(covariance_matrix,dim1=1,dim2=2).sum())**2
        loss/=B*C*(H*W)*((H*W)-1)
        return loss
    
class NegativeLikelihooldLossFunc(nn.Module):
    def __init__(self):
        super(NegativeLikelihooldLossFunc, self).__init__() 
        
    def forward(self,img,mean,var):
        #assert img.size()==std.size()
        assert img.size()==mean.size()
        #_,_,P1,P2=inputs.size()
        #_,C,H,W=img.size()
        #img=rearrange('b c (p1 n1) (p2 n2) -> b (c n1 n2) p1 p2',p1=H,p2=W)
        #img=nn.Unfold((H//P1,W//P2),stride=H//P1)(img)
        #print(img.size())
        #img=rearrange(img,'b (c p1 p2) l-> b l c p1 p2',c=C,p1=H//P1,p2=W//P2)
        #print(img.size())
        
        #img=torch.var(img,dim=(-2,-1))
        #img=torch.mean(img,dim=-1,keepdim=True)
        
        #print(img.size())
        
        #print(img.size())
        
        #img=rearrange(img,'b (l1 l2) c -> b c l1 l2',l1=P1,l2=P2)
        #print(img.size())
        
        #loss=((std**2).log())+(img-mean)**2/std**2-0.1*torch.mean(std,dim=1,keepdim=True)
        #img=img.detach()
        img=img.detach()
        mean=mean.detach()
        
        T=img-mean
        T=T.permute(0,2,3,1).unsqueeze(-1)
        #print(std[30])
        t1=0.5*(T.transpose(3,4)@torch.linalg.inv(var)@T).squeeze(-1).squeeze(-1)
        dets=torch.det(var)
        #dets=0.5*torch.log(dets.clamp(1e-3))
        dets=0.5*torch.log(dets.clamp(1e-6))
        
        #t1=((img-mean)**2)/(std**2)
        #t2=(std**2).log()
        #t2=nn.ReLU()(t2)
        loss=t1+dets
        #loss[t1>1e+8]=0.
        #loss=((std**2).log())+(img-mean)**2/(std**2)
        #loss=torch.sum(loss,dim=(-2,-1))
        loss=torch.mean(loss)
        #if t1.max()>1e+7:
        #    loss.data.zero_()
        #print(loss)
        return loss
    


    
class PerceptualLossFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.contentFunc=self.contentfunc()
        
    def contentfunc(self):
        conv_3_3_layer=14     
        cnn=models.vgg19().features
        cnn[0]=nn.Conv2d(2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i,layer in enumerate(list(cnn)):
            model.add_module(str(i),layer)
            if i == conv_3_3_layer:
                break
        return model
	 
    def forward(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss
    

class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)



def sum_losses(loss:dict):
    sums=0.
    for k,v in loss.items():
        if k!='DH_D' and k!= 'DL_D' and k!='CONTRASTIVE' and k!='IDENTITY' and k!='total':
            #print(k,v)
            sums+=v
    return sums

def updata_losses_meter(meter,results,batch):
    for k,v in meter.items():
        if results[k] is not None:
            v.update(results[k].item(),batch)

    return meter

def reset_losses_meter(meter):
    for v in meter.values():
        v.reset()
    return meter



if __name__=="__main__":
    x=torch.randn(128,64)
    x_1=torch.randn(128,64)
    img=torch.randn(256,64)
    #optimizer=torch.optim.SGD([x],lr=0.01)
    #criterion=NegativeLikelihooldLossFunc()
    criterion=InfoNCE()
    criterion(x,x_1,img)
    #loss=criterion(img,x)
    #print(loss)
    """
    for i in range(100):
        loss=criterion(x)
        if i % 100:
            print(f"i == {i}: Nuclear Norm Loss == {loss}")
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
    """