import os.path as path
import os
import torch
import torch.distributed as dist
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from timm.utils import  AverageMeter
from tqdm import tqdm
from utils.metrics import calculate_nrmse_psnr,snr_change
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nrmse

from networks import build_model

from utils.lr_scheduler import build_scheduler
from utils.optimizer import build_optimizer
from utils.utils import load_checkpoint, save_checkpoint, NativeScalerWithGradNormCount, \
    reduce_tensor


class BaseModel():
    def __init__(self,logger,config,data_loader):
        self.config=config
        self.logger=logger
        self.data_loader=data_loader    
        self.frmat_float=lambda x: np.format_float_scientific(x, exp_digits=1, precision=2)     
        self.model={}
        self.model_cmnet={}
        
        self.model_dl=None

        self.model_dh=None
        self.model_without_ddp=None
        
    
    def _tqdm(self,iterable,epoch):
        if dist.get_rank()==0:
            progress = tqdm(iterable, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
        else:
            progress=iterable
    
        for obj in progress:
            yield obj
        """
            if dist.get_rank()!=0: continue
            desc = '[epoch: {}]'.format(epoch)
            loss_str= " ".join(["{} {:.2e} ({:.2e})".format(k,v.val,v.avg)for k, v in self.loss_meter.items()])
            
            desc+=loss_str
            progress.set_description(desc=desc)
        """
        
    
    def _make_visuals(self, lookup, n, func=None,):
        if func is None: func = lambda x: x
        pairs = [(t, func(getattr(self, k)[:n])) for t, k in lookup if hasattr(self, k)]
        tags, images = zip(*pairs)
        tags, images = "_".join(tags), torch.cat(images)
        #turn to complex
        images=images[:,0].unsqueeze(1)
        #min,_=images.min(dim=0)
        #max,_=images.max(dim=0)
        #images=(images-min)/(max-min+1e-20)
        images*=0.5+0.5
        visuals = make_grid(images, nrow=images.shape[0] // len(pairs),padding=1, normalize=True,scale_each=True)
        visuals=visuals.squeeze().detach().cpu().numpy()
        
        return {tags: visuals}
    
    def _visuals(self,lookup,epoch,idx, n,prefix='train'):
        #lookup = [
        #    ("l", "noisy"),
        #    ("h", "clean"),]
    
        for k, v in self._make_visuals(lookup,n).items():
            visual_dir = path.join(self.config.OUTPUT,prefix+'_'+self.config.VISUAL.NAME)
            if not path.isdir(visual_dir): os.makedirs(visual_dir)
            visual_file = path.join(visual_dir,
                "epoch{}_iter{}_{}.png".format(epoch, idx, k))
            plt.imsave(visual_file,v,cmap='gray')
    
   
    def _load_checkpoint(self,logger,model,model_cmnet,model_dl=None,model_dh=None):
        load_checkpoint(self.config,logger,model=model,model_cmnet=model_cmnet,model_dl=model_dl,model_dh=model_dh)

    def _save_checkpoint(self,epoch,logger,model,model_dl,model_dh,model_cmnet,is_best=False):
        
        save_checkpoint(self.config,epoch,logger,model=model,model_dl=model_dl,model_dh=model_dh,model_cmnet=model_cmnet,is_best=is_best)
        
    def _get_model(self,model_type):
        
        self.logger.info(f"Creating model:{model_type}/{self.config.MODEL.NAME}")
        
        model = build_model(model_type,self.config)
        
        self.logger.info(str(model))
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"number of params: {n_parameters}")
        if hasattr(model, 'flops'):
            flops = model.flops()
            self.logger.info(f"number of GFLOPs: {flops / 1e9}")
        return model
    
    def _get_optimier(self,model):
        return build_optimizer(self.config,model)
    
    def _get_scheduler(self,optimizer,len_data_loader_train):
        if self.config.TRAIN.ACCUMULATION_STEPS > 1: 
            return build_scheduler(self.config, optimizer, len_data_loader_train // self.config.TRAIN.ACCUMULATION_STEPS)
        else:
            return build_scheduler(self.config, optimizer, len_data_loader_train)
    

    def _build_model(self,model_type,len_data_loader_train):

        model_dict={}
        model=self._get_model(model_type)
        
        model.cuda()
        model_without_ddp = model
        model_dict['model'] = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.config.LOCAL_RANK], broadcast_buffers=False)
        model_dict['model_without_ddp'] = model_without_ddp
        model_dict['optimizer']=self._get_optimier(model_dict['model'])
        model_dict['lr_scheduler'] = self._get_scheduler(model_dict['optimizer'],len_data_loader_train)
        model_dict['loss_scaler'] = NativeScalerWithGradNormCount()
        model_dict['norm'] = AverageMeter()
        model_dict['loss_scale_value'] = AverageMeter()
        return model_dict
    

    def calculate_nrmse_psnr(self,ground_truth,denoised_images):
        nrmse,psnr=calculate_nrmse_psnr(ground_truth,denoised_images)
        #return reduce_tensor(nrmse(ground_truth,denoised_images,normalization='min-max')),reduce_tensor(psnr(ground_truth,denoised_images),data_range=2.)
        return reduce_tensor(nrmse),reduce_tensor(psnr)

    def calculate_sbrc(self,noisy_imgs,denoised_images):
        
        return reduce_tensor(snr_change(noisy_imgs=noisy_imgs,denoised_images=denoised_images))
        

    

    def evaluate(self,batch_size,noisy,outputs,target=None):
        #batch_size=1
        if self.metric_meter.get('sbrc',None) is not None:
            
            sbrc=self.calculate_sbrc(noisy*0.5+0.5,outputs*0.5+0.5)
            self.metric_meter['sbrc'].update(sbrc,1)
            
        if target is not None and self.metric_meter.get('nrmse',None) and self.metric_meter.get('psnr',None) is not None and self.config.DATA.NOISE.ADD_NOISE :
                nrmse_batch,psnr_batch=self.calculate_nrmse_psnr(target,outputs)
                #nrmse_batch=nrmse(target.cpu().numpy(),outputs.cpu().numpy(),normalization='min-max')
                #psnr_batch=psnr(target.cpu().numpy(),outputs.cpu().numpy(),data_range=2.)
                self.metric_meter['nrmse'].update(nrmse_batch,1) 
                self.metric_meter['psnr'].update(psnr_batch,1) 
                
    def _bulid(self,len_data_loader_train): raise NotImplementedError

    def visuals(self,epoch,idx,n=8):  raise NotImplementedError

    def forward(self,inputs): raise NotImplementedError

    def forward_dh(self,fake,real):  raise NotImplementedError

    def forward_dl(self,fake,real):  raise NotImplementedError

    def forward_one_epoch(self,epoch): raise NotImplementedError
   