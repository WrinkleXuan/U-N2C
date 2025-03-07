import torch
import torch.distributed as dist
import numpy as np
from timm.utils import  AverageMeter
import matplotlib.pyplot as plt
from  .Base import BaseModel
from losses import build_metric

from  MPI import MPI
from utils.metrics import calculate_psnr,calculate_nrmse_reco,calculate_ssim

import scipy.io as sio
import os.path as path
import os
import time

class Tester(BaseModel):
    def __init__(self,logger,config,data_loader):
        super(Tester,self).__init__(logger,config,data_loader)
        self._bulid(len(self.data_loader))

    def _visuals(self,lookup,freq,slices,channel, n,prefix='train'):
        #lookup = [
        #    ("l", "noisy"),
        #    ("h", "clean"),]
    
        for k, v in self._make_visuals(lookup,n).items():
            visual_dir = path.join(self.config.OUTPUT,prefix+'_'+self.config.VISUAL.NAME)
            if not path.isdir(visual_dir): os.makedirs(visual_dir)
            visual_file = path.join(visual_dir,
                "fre{}_channel{}_slice{}_{}.png".format(freq,channel,slices, k))
            plt.imsave(visual_file,v,cmap='gray')
    
    def visuals(self,freq,slices,channel,n=8,prefix='test'):
        lookup = [
            #('G','target'),("l", "noisy"),("n", "noisy_part"),("lh", "outputs")]
            ('G','target'),("l", "noisy"),("lh", "outputs"),('bm3d','outputs_bm3d'),('lhl','noisy_reco'),
            ('h','clean'), ('hl','generated_noisy')]
        return self._visuals(lookup,freq,slices,channel,n,prefix=prefix)
    
    def _bulid(self,len_data_loader_train):


        self.model=self._build_model(self.config.MODEL.TYPE,len_data_loader_train)
        self.model_cmnet=self._build_model(self.config.MODEL.MTYPE,len_data_loader_train)
        
        
        self.metric_meter=build_metric(self.config)
        
        self._load_checkpoint(self.logger,model=self.model,model_cmnet=self.model_cmnet)
    
    def evaluate_image_reco(self,denoised_sm,
                            noised_sm=None,
                            system_matrix_name='6.mdf',
                            phantom_dir='/data/zwx/data/OpenMPIData/measurements',
                            reco_mode='3',
                            phantom_name:list | str=['resolutionPhantom','concentrationPhantom','shapePhantom']):
        if isinstance(phantom_name,str):
            phantom_name=[phantom_name]
        for phantom in phantom_name:
            
            data=MPI(measurements=phantom_dir+'/'+phantom+'/'+reco_mode+'.mdf',system_matrix="/data/zwx/data/OpenMPIData/calibrations/"+system_matrix_name+'.mdf')
            snr_threshold=(5,1e+20) if self.config.DATA.NOISE.ADD_NOISE else (2,5)
            img_denoised=data.reco_3d(denoised_sm,snr_threshold=snr_threshold,reco_name='denoised',output_dir=self.config.OUTPUT) 
            
            gt=data.reco_3d(snr_threshold=5,output_dir=self.config.OUTPUT)
            if self.config.DATA.NOISE.ADD_NOISE:
                noised=data.reco_3d(noised_sm,snr_threshold=snr_threshold,reco_name='noised',add_noise=True,snr_value=self.config.DATA.NOISE.SNR,noise_type=self.config.DATA.NOISE.NOISE_TYPE,output_dir=self.config.OUTPUT)
            else:
                noised=data.reco_3d(snr_threshold=snr_threshold[0],reco_name='noised',add_noise=True,snr_value=self.config.DATA.NOISE.SNR,noise_type=self.config.DATA.NOISE.NOISE_TYPE,output_dir=self.config.OUTPUT)
            
            if self.config.DATA.NOISE.ADD_NOISE:
                psnr=calculate_psnr(gt,img_denoised)

                self.metric_meter['psnr_reco'].update(psnr,1) 
                nrmse=calculate_nrmse_reco(gt,img_denoised)
                self.metric_meter['nrmse_reco'].update(nrmse,1)

                ssim=calculate_ssim(gt,img_denoised)
                self.metric_meter['ssim'].update(ssim,1)
            sio.savemat(self.config.OUTPUT+'/'+phantom+'.mat',{'gt':gt,'denoised':img_denoised,'noised':noised})    

    def forward(self,inputs):
        self.model['model'].eval()
        self.model_cmnet['model'].eval()
        
        self.index=inputs['name']
        self.noisy = inputs['noisy'].cuda(non_blocking=True)
        
        self.noise_freq=inputs['freq_noise'].cuda(non_blocking=True)
        self.noise_channel=inputs['channel_noise'].cuda(non_blocking=True)
        
        self.noise_position=inputs['position_noise'].cuda(non_blocking=True)
        
        if self.config.DATA.NOISE.ADD_NOISE:
            self.target = inputs['target'].cuda(non_blocking=True)
        #self.outputs,_,self.noisy_reco,self.clean_reco,self.generated_noisy = self.model['model'](self.noisy,self.clean)
        #self.outputs,self.noisy_reco,self.generated_noisy,self.generated_noisy_denoised,self.noise,self.noise_y,self.content_y,self.generated_content_y = self.model['model'](self.noisy,self.clean)
        self.outputs,_,self.noise_map,self.noise_std=self.model['model'].module.denoised_forward(self.noisy,
                                                                                                self.noise_freq,
                                                                                                self.noise_position,
                                                                                                self.noise_channel,
                                                                                                self.model_cmnet['model'],
                                                                                                update_flag=False)
        
        self.noisy_reco=self.model['model'].module.clean_forward(self.outputs)[0].detach()+self.noise_map
        
        #self.outputs_bm3d=torch.tensor([torch.from_numpy(bm3d.bm3d(img.permute(1,2,0).cpu().numpy(),estimate_sigma(img.permute(1,2,0).cpu().numpy(),channel_axis=-1))).permute(2,0,1) for img in self.outputs])
    
    
    @torch.no_grad()        
    def forward_one_epoch(self,epoch):
        output=[]
        noisy=[]
        slices=0    
        for idx, data in enumerate(self._tqdm(self.data_loader,epoch)):
            self.forward(data)
            torch.cuda.synchronize()
            #torch.tensor([self.data_loader.dataset.to_numpy(out,self.index[i].data) for i,out in enumerate(self.outputs)]).cuda()
            self.evaluate(self.noisy.size(0),
                          self.noisy,
                          self.outputs,
                          self.target if hasattr(self,'target') else None)
            slices=0 if idx%37==0 else slices+1 
            #self.evaluate(self.noisy.size(0),
            #              self.data_loader.dataset.denormalize_per_batch(self.index,self.noisy),
            #              self.data_loader.dataset.denormalize_per_batch(self.index,self.outputs),
            #              self.data_loader.dataset.denormalize_per_batch(self.index,self.target) if hasattr(self,'target') else None)
            if dist.get_rank() == 0 and slices==19:
                self.visuals(freq=self.noise_freq[0][0]*10e6,slices=slices,channel=self.noise_channel[0][0],n=self.noisy.size(0))
            noisy.extend(
                self.data_loader.dataset.to_numpy(img,self.index[i].data) for i,img in enumerate(self.noisy)
            )
            output.extend(
               self.data_loader.dataset.to_numpy(out,self.index[i].data) for i,out in enumerate(self.outputs)
            )
        output=np.array(output)
        noisy=np.array(noisy)
        if self.config.DATA.DATA_NAME_TEST=='calibrations_6.npz':
            self.evaluate_image_reco(output,noisy,system_matrix_name=self.config.DATA.DATA_NAME_TEST.split('.')[0].split('_')[-1],phantom_name='shapePhantom')
        np.save(self.config.OUTPUT+'/sm_denoised.npy',output)
        np.save(self.config.OUTPUT+'/sm_noised.npy',noisy)

        self.logger.info(" ".join(["Test"] +["{} {:.2e}".format(k,v.avg) for k, v in self.metric_meter.items() if v.count!=0]))
    
    @torch.no_grad()
    def forward_throughput(self):
        for idx ,data in enumerate(self.data_loader):
        
            #images = images.cuda(non_blocking=True)
            batch_size = data['noisy'].shape[0]
            for i in range(100):
                self.forward(data)
            torch.cuda.synchronize()
            self.logger.info(f"throughput averaged with 100 times")
            tic1 = time.time()
            for i in range(100):
                self.forward(data)
            torch.cuda.synchronize()
            tic2 = time.time()
            self.logger.info(f"total_time: {tic2-tic1} batch_size {batch_size} throughput {100 * batch_size / (tic2 - tic1)}")
            return