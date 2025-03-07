import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from timm.utils import  AverageMeter
import matplotlib.pyplot as plt
from  .Base import BaseModel
from losses import build_metric

from  MPI import MPI
from MPI.kaczmarzReg import kaczmarzReg
from MPI.PP import rec
from utils.metrics import calculate_psnr,calculate_nrmse_reco,calculate_ssim

import scipy.io as sio
import mat73
import os.path as path
import os
from einops import repeat,rearrange
class Tester_FFL(BaseModel):
    def __init__(self,logger,config,data_loader):
        super(Tester_FFL,self).__init__(logger,config,data_loader)
        self._bulid(len(self.data_loader))

    def _visuals(self,lookup,freq,slices,channel, n,prefix='train'):
        #lookup = [
        #    ("l", "noisy"),
        #    ("h", "clean"),]
    
        for k, v in self._make_visuals(lookup,n).items():
            visual_dir = path.join(self.config.OUTPUT,prefix+'_'+self.config.VISUAL.NAME)
            if not path.isdir(visual_dir): os.makedirs(visual_dir)
            visual_file = path.join(visual_dir,
                "fre{}_channel{}_angle{}_{}.png".format(freq,channel,slices, k))
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
    
    
    def evaluate_image_reco(self,denoised_sm=None,
                            system_matrix_name='Sz_reco',
                            phantom_dir='/data/zwx/data/OS_FFL_MPIData/data_OS-FFL-MPI',
                            phantom_name:list | str=['vessel','2mm','1mm']):
        if isinstance(denoised_sm,str):
            denoised_sm=np.load(denoised_sm)
        if isinstance(phantom_name,str):
            phantom_name=[phantom_name]
        #harmonic=np.arange(1,8+1,step=1)
        #harmonic=repeat(harmonic.reshape(-1,1),'n l->(n c) l',c=25) #25 is the number of sidebands
        #harmonic=repeat(harmonic,'n l ->(c n) l',c=30) # 30 is the number of angles
        #mask=np.logical_and(harmonic>=2,harmonic<=7)

        sidebands=np.arange(-12,12+1,step=1)*120.
        harmonic=np.arange(1,8+1,step=1)*3000.
        harmonic=repeat(harmonic.reshape(-1,1),'n l->(n c) l',c=len(sidebands))
        sidebands=repeat(sidebands.reshape(-1,1),'n l->(c n) l',c=8)
        freq=harmonic+sidebands
        freq=freq[freq!=3000.0].reshape(-1,1)
        #print(freq)
        freq=repeat(freq,'n l ->(c n) l',c=30)
        mask=np.logical_and(freq>=4560,freq<=22440).reshape(-1,)

        for p_name in phantom_name:
            output_dir=os.path.join(self.config.OUTPUT,'results/OS_FFL_MPIData/{}'.format(p_name))
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            phantom=np.expand_dims(mat73.loadmat(os.path.join(phantom_dir,p_name+'_Uz_reco.mat'))['Uz_reco'],1)
            
            phantom=np.concatenate((np.expand_dims(phantom[:phantom.shape[0]//2],1),np.expand_dims(phantom[phantom.shape[0]//2:,],1)),axis=1)
            phantom = phantom[:,0,:].squeeze()+1j*phantom[:,1,:].squeeze()
            
            bg=np.expand_dims(mat73.loadmat(os.path.join(phantom_dir,p_name+'_dz_reco.mat'))['dz_reco'],1)
            bg=np.concatenate((np.expand_dims(bg[:bg.shape[0]//2],1),np.expand_dims(bg[bg.shape[0]//2:,],1)),axis=1)
            bg = bg[:,0,:].squeeze()+1j*bg[:,1,:].squeeze()
            phantom=phantom[mask]
            
            bg=bg[mask]
            
            snr=np.abs(phantom)//np.abs(bg)
            #print(snr)
            phantom=phantom-bg
            
            snr=snr.reshape(-1,)
            sm=mat73.loadmat(os.path.join(phantom_dir,system_matrix_name+'.mat'))['Sz_reco']
            sm=np.concatenate((np.expand_dims(sm[:sm.shape[0]//2],1),np.expand_dims(sm[sm.shape[0]//2:,],1)),axis=1)
            
            sm = sm[:,0,:].squeeze()+1j*sm[:,1,:].squeeze()
            sm=sm[mask]
            #S=np.concatenate((np.real(sm[snr>3]),np.imag(sm[snr>3])),0)
            #c=np.concatenate((np.real(phantom[snr>3]),np.imag(phantom[snr>3])),0)
            gt=kaczmarzReg(sm[snr>3],phantom[snr>3],1,np.linalg.norm(sm[snr>3],ord='fro')*1e-1,False,True,True)
            #gt=rec(A=torch.tensor(sm[snr>3],dtype=torch.complex64).cuda(),u=torch.tensor(phantom[snr>3],dtype=torch.complex64).cuda(),img_size=(self.config.DATA.IMG_SIZE,self.config.DATA.IMG_SIZE),device=dist.get_rank())
            gt=gt.reshape(self.config.DATA.IMG_SIZE,self.config.DATA.IMG_SIZE)
            
            gt=np.real(gt)
            plt.imsave(output_dir+'/gt.png',gt,cmap='gray')

            if denoised_sm is not None:
                #phantom=np.expand_dims(mat73.loadmat(os.path.join(phantom_dir,p_name+'_Uz_reco.mat'))['Uz_reco'],1)
            
                #phantom=np.concatenate((np.expand_dims(phantom[:phantom.shape[0]//2],1),np.expand_dims(phantom[phantom.shape[0]//2:,],1)),axis=1)
                #phantom = phantom[:,0,:].squeeze()+1j*phantom[:,1,:].squeeze()
            
                #noised_sm=mat73.loadmat(os.path.join(phantom_dir,'Sz_reco.mat'))['Sz_reco']
            
                #noised_sm=np.concatenate((np.expand_dims(noised_sm[:noised_sm.shape[0]//2],1),np.expand_dims(noised_sm[noised_sm.shape[0]//2:,],1)),axis=1)
                #noised_sm = noised_sm[:,0,:].squeeze()+1j*noised_sm[:,1,:].squeeze()
                
                noised=kaczmarzReg(sm,phantom,1,np.linalg.norm(sm,ord='fro')*1e-1,False,True,True)
                noised=noised.reshape(self.config.DATA.IMG_SIZE,self.config.DATA.IMG_SIZE)
                noised=np.real(noised)
                plt.imsave(output_dir+'/noised.png',noised,cmap='gray')
            
            
                
                img_denoised=kaczmarzReg(denoised_sm,phantom,3,np.linalg.norm(denoised_sm,ord='fro')*1e-5,False,True,True)
                #img_denoised=rec(A=torch.tensor(denoised_sm,dtype=torch.complex64).cuda(),u=torch.tensor(phantom,dtype=torch.complex64).cuda(),img_size=(self.config.DATA.IMG_SIZE,self.config.DATA.IMG_SIZE),device=dist.get_rank())
            
                img_denoised=img_denoised.reshape(self.config.DATA.IMG_SIZE,self.config.DATA.IMG_SIZE)
                img_denoised=np.real(img_denoised)
                plt.imsave(output_dir+'/img_denoised.png',img_denoised,cmap='gray')

            

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
        #self.outputs_bm3d=torch.tensor([torch.from_numpy(bm3d.bm3d(img.permute(1,2,0).cpu().numpy(),estimate_sigma(img.permute(1,2,0).cpu().numpy(),channel_axis=-1))).permute(2,0,1) for img in self.outputs])
    
    
    @torch.no_grad()        
    def forward_one_epoch(self,epoch):
        output=[]
        noisy=[]
        for idx, data in enumerate(self._tqdm(self.data_loader,epoch)):
            self.forward(data)
            torch.cuda.synchronize()
            #torch.tensor([self.data_loader.dataset.to_numpy(out,self.index[i].data) for i,out in enumerate(self.outputs)]).cuda()
            
            self.evaluate(self.noisy.size(0),
                          self.noisy,
                          self.outputs,
                          self.target if hasattr(self,'target') else None)
            
            #self.evaluate(self.noisy.size(0),
            #              self.data_loader.dataset.denormalize_per_batch(self.index,self.noisy),
            #              self.data_loader.dataset.denormalize_per_batch(self.index,self.outputs),
            #              self.data_loader.dataset.denormalize_per_batch(self.index,self.target) if hasattr(self,'target') else None)
            if dist.get_rank() == 0:
                self.visuals(freq=self.noise_freq[0][0],slices=idx//199,channel=self.noise_channel[0][0],n=self.noisy.size(0),prefix='test_ffl')
            output.extend(
               self.data_loader.dataset.to_numpy(out,self.index[i].data) for i,out in enumerate(self.outputs)
            )
        output=np.array(output)
        output=rearrange(output,'f h w->f (h w)',h=self.config.DATA.IMG_SIZE,w=self.config.DATA.IMG_SIZE)
        np.save(self.config.OUTPUT+'/ffl_sm_denoised.npy',output)
        self.evaluate_image_reco(denoised_sm=output,phantom_name='1mm')
        #self.evaluate_image_reco()
        
        self.logger.info(" ".join(["Test"] +["{} {:.2e}".format(k,v.avg) for k, v in self.metric_meter.items() if v.count!=0]))


if __name__=='__main__':
    Tester_FFL.evaluate_image_reco(denoised_sm='ADNet_gaussion_noise_2.0_refine/ffl_sm_denoised.npy')
