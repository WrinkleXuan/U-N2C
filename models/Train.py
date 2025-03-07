import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from timm.utils import  AverageMeter
from tqdm import tqdm
from einops import rearrange


from losses import build_criterion,build_loss_meter,updata_losses_meter,reset_losses_meter,sum_losses,build_metric

from .Base import BaseModel


class Trainer(BaseModel):
    def __init__(self,logger,config,data_loader_train,data_loader_val):
        super(Trainer,self).__init__(logger,config,data_loader_train)
        self.config=config
        self.logger=logger
        if dist.get_rank()==0:
            self.writer=SummaryWriter(self.config.OUTPUT_TRAIN_LOG)
            self.writer_val=SummaryWriter(self.config.OUTPUT_VAL_LOG)
        self.data_loader_val=data_loader_val
        self.max_snr_change=1e-30
        
        self.max_psnr=1e-30
        self.min_nrmse=1e+30
        self._bulid(len(self.data_loader))
        
    def _reset(self):
        #self.batch_time.reset()
        for m in self.metric_meter.values():
            m.reset()
        reset_losses_meter(self.loss_meter)

        self.model['norm'].reset()
        self.model['loss_scale_value'].reset()
        if not self.config.FINETUNE:
        
            self.model_cmnet['norm'].reset()
            self.model_cmnet['loss_scale_value'].reset()
            self.model_dh['norm'].reset()
            self.model_dh['loss_scale_value'].reset()
            
            self.model_dl['norm'].reset()
            self.model_dl['loss_scale_value'].reset()

    def _write_loss(self,epoch,writer):
        for k,v in self.loss_meter.items():
            writer.add_scalar(k,v.avg,epoch) 

    def _write_metric(self,epoch,writer):
        for k,v in self.metric_meter.items():
            writer.add_scalar(k,v.avg,epoch) 
    def visuals(self,epoch,idx,n=8,prefix='train'):
        lookup = [
            ('G','target'),("l", "noisy"),("lh", "outputs"),('ll','autoencoder_reco'),('lhl','noisy_reco'),
            ('h','clean'), ('hl','generated_noisy'),('hlh','generated_noisy_denoised'),('hh','clean_reco'),('mh','memory_clean_reco')]
        #lookup = [
        #    ('G','target'),("l", "noisy"),("lh", "outputs"),('lhl','noisy_reco'),
        #    ('h','clean'), ('hl','generated_noisy'),('hlh','generated_noisy_denoised')]
        

        return self._visuals(lookup,epoch,idx,n,prefix=prefix)
    


    def _bulid(self,len_data_loader_train):

        self.model=self._build_model(self.config.MODEL.TYPE,len_data_loader_train)
        self.model_cmnet=self._build_model(self.config.MODEL.MTYPE,len_data_loader_train)
        if not self.config.FINETUNE:
            self.model_dl=self._build_model(self.config.MODEL.DTYPE,len_data_loader_train)
            self.model_dh=self._build_model(self.config.MODEL.DTYPE,len_data_loader_train)
        
        self.criterion=build_criterion(self.config)
        self.loss_meter,self.loss=build_loss_meter(self.criterion)
        #self.batch_time=AverageMeter()
        self.metric_meter=build_metric(self.config)

        if self.config.MODEL.RESUME:
            self._load_checkpoint(self.logger,model=self.model,model_cmnet=self.model_cmnet,model_dl=self.model_dl,model_dh=self.model_dh)
        
    
    def _update_parameters(self,model,idx,epoch,num_steps,acumulation_steps,loss,clip_grad,update_grad):
        network=model['model']
        optimizer=model['optimizer']
        lr_scheduler=model['lr_scheduler']
        loss_scaler=model['loss_scaler']
        norm_meter=model['norm']
        scaler_meter=model['norm']
        
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=clip_grad,
                                parameters=network.parameters(), create_graph=is_second_order,
                                update_grad=update_grad)
        #if dist.get_rank() == 0:
            #for name, param in network.named_parameters():
            #    if param.requires_grad:
            #        print(name)
            #if hasattr(network.module,'encoder_clean'):
            #    print(network.module.encoder_clean.requires_grad)
            
        if update_grad:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // acumulation_steps)
            norm_meter.update(grad_norm)
            scaler_meter.update(loss_scaler.state_dict()["scale"])
    
    
    def forward(self,inputs,is_train=True):
        if is_train:
            self.model['model'].train()
        else:
            self.model['model'].eval()
        self.model_cmnet['model'].eval()
        
        self.clean = inputs['clean'].cuda(non_blocking=True)
        self.clean_freq=inputs['freq_clean'].cuda(non_blocking=True)
        self.clean_channel=inputs['channel_clean'].cuda(non_blocking=True)
        self.clean_position=inputs['position_clean'].cuda(non_blocking=True)
        
        self.noisy = inputs['noisy'].cuda(non_blocking=True)
        
        self.noise_freq=inputs['freq_noise'].cuda(non_blocking=True)
        self.noise_channel=inputs['channel_noise'].cuda(non_blocking=True)
        
        self.noise_position=inputs['position_noise'].cuda(non_blocking=True)
        
        if self.config.DATA.NOISE.ADD_NOISE:
            self.target = inputs['target'].cuda(non_blocking=True)
        
        self.clean_reco,self.content_y=self.model['model'].module.clean_forward(self.clean)
        self.outputs,_,self.noise_map,self.noise_std=self.model['model'].module.denoised_forward(self.noisy,
                                                                                                self.noise_freq,
                                                                                                self.noise_position,
                                                                                                self.noise_channel,
                                                                                                self.model_cmnet['model'],
                                                                                                update_flag=True)
        
                                                                                
        self.generated_noisy=self.clean_reco.detach()+self.noise_map
        
        self.generated_noisy_denoised,self.generated_content_y,self.generated_noise_map,self.noise_std_y=self.model['model'].module.denoised_forward(self.generated_noisy,
                                                                                                                                                    self.clean_freq,
                                                                                                                                                    self.clean_position,
                                                                                                                                                    self.clean_channel,
                                                                                                                                                    self.model_cmnet['model'],
                                                                                                                                                    update_flag=False
                                                                                                                                                    )
        self.noisy_reco=self.model['model'].module.clean_forward(self.outputs)[0].detach()+self.generated_noise_map
        
        
                            
        
        #LL
        if self.config.TRAIN.LOSS.LL[1]!=0:
            self.loss['LL']=self.config.TRAIN.LOSS.LL[1]*self.criterion['LL'](self.outputs,self.noisy_reco.detach()) # autoencoder
            self.loss['LL']/=self.config.TRAIN.ACCUMULATION_STEPS
        
        #HH
        if self.config.TRAIN.LOSS.HH[1]!=0:
            self.loss['HH']=self.config.TRAIN.LOSS.HH[1]*self.criterion['HH'](self.clean_reco,self.clean) # autoencoder
            self.loss['HH']/=self.config.TRAIN.ACCUMULATION_STEPS
        #LH
        if self.config.TRAIN.LOSS.LH[1]!=0:
            self.loss['LH']=self.config.TRAIN.LOSS.LH[1]*self.criterion['LH'](self.outputs,self.noisy) # content consistensy constrain
            self.loss['LH']/=self.config.TRAIN.ACCUMULATION_STEPS
        #HL
        if self.config.TRAIN.LOSS.HL[1]!=0:
            self.loss['HL']=self.config.TRAIN.LOSS.HL[1]*self.criterion['HL'](self.generated_noisy,self.clean) # content consistensy constrain
            self.loss['HL']/=self.config.TRAIN.ACCUMULATION_STEPS
            

        #LHL
        if self.config.TRAIN.LOSS.LHL[1]!=0:
            self.loss['LHL']=self.config.TRAIN.LOSS.LHL[1]*self.criterion['LHL'](self.noisy_reco,self.noisy) # content consistensy constrain
            self.loss['LHL']/=self.config.TRAIN.ACCUMULATION_STEPS
        

        #HLH
        if self.config.TRAIN.LOSS.HLH[1]!=0:
            self.loss['HLH']=self.config.TRAIN.LOSS.HLH[1]*self.criterion['HLH'](self.generated_noisy_denoised,self.clean) # content consistensy constrain
            self.loss['HLH']/=self.config.TRAIN.ACCUMULATION_STEPS
        
        
        #CC
        if self.config.TRAIN.LOSS.CC[1]!=0:
            self.loss['CC']=self.config.TRAIN.LOSS.CC[1]*self.criterion['CC'](self.generated_content_y,self.content_y.detach()) # content consistensy constrain
            self.loss['CC']/=self.config.TRAIN.ACCUMULATION_STEPS
        #NE
        if self.config.TRAIN.LOSS.NE[1]!=0:
            self.loss['NE']=self.config.TRAIN.LOSS.NE[1]*self.criterion['NE'](self.noisy,self.outputs,self.noise_std) 
            self.loss['NE']/=self.config.TRAIN.ACCUMULATION_STEPS

        #DH
        if self.config.TRAIN.LOSS.DH[1]!=0:
            pred_fake=self.model_dh['model'](self.outputs)
            self.loss['DH_G']=self.config.TRAIN.LOSS.DH[1]*self.criterion['DH'](pred_fake=pred_fake) # generate clean sm
            self.loss['DH_G']/=self.config.TRAIN.ACCUMULATION_STEPS
        
        #DL
        if self.config.TRAIN.LOSS.DL[1]!=0:
            pred_fake=self.model_dl['model'](self.generated_noisy)    
            self.loss['DL_G']=self.config.TRAIN.LOSS.DL[1]*self.criterion['DL'](pred_fake=pred_fake) # cyclegan
            self.loss['DL_G']/=self.config.TRAIN.ACCUMULATION_STEPS
        
        self.loss['total']=sum_losses(self.loss)
        self.loss['total'] = self.loss['total'] / self.config.TRAIN.ACCUMULATION_STEPS
    
    def forward_finetune(self,inputs,is_train=True):
        if is_train:
            self.model['model'].train()
        else:
            self.model['model'].eval()
        self.model_cmnet['model'].eval()
        
        self.clean = inputs['clean'].cuda(non_blocking=True)
        self.clean_freq=inputs['freq_clean'].cuda(non_blocking=True)
        self.clean_channel=inputs['channel_clean'].cuda(non_blocking=True)
        self.clean_position=inputs['position_clean'].cuda(non_blocking=True)
        
        self.noisy = inputs['noisy'].cuda(non_blocking=True)
        
        self.noise_freq=inputs['freq_noise'].cuda(non_blocking=True)
        self.noise_channel=inputs['channel_noise'].cuda(non_blocking=True)
        
        self.noise_position=inputs['position_noise'].cuda(non_blocking=True)
        
        if self.config.DATA.NOISE.ADD_NOISE:
            self.target = inputs['target'].cuda(non_blocking=True)
        
        with torch.no_grad():
            #self.clean_reco,self.content_y=self.model['model'].module.clean_forward(self.clean)
        
            self.outputs,_,self.noise_map,self.noise_std=self.model['model'].module.denoised_forward(self.noisy,
                                                                                                    self.noise_freq,
                                                                                                    self.noise_position,
                                                                                                    self.noise_channel,
                                                                                                    self.model_cmnet['model'],
                                                                                                    update_flag=False)
        
                                                                                
            self.generated_noisy=self.clean.detach()+self.noise_map
        
        self.generated_noisy_denoised,_,_,_=self.model['model'].module.denoised_forward(self.generated_noisy,
                                                                                        self.clean_freq,
                                                                                        self.clean_position,
                                                                                        self.clean_channel,
                                                                                        self.model_cmnet['model'],
                                                                                        update_flag=False)
        #HLH
        if self.config.TRAIN.LOSS.HLH[1]!=0:
            self.loss['HLH']=self.config.TRAIN.LOSS.HLH[1]*self.criterion['HLH'](self.generated_noisy_denoised,self.clean) # content consistensy constrain
            self.loss['HLH']/=self.config.TRAIN.ACCUMULATION_STEPS
        self.loss['total']=sum_losses(self.loss)
        self.loss['total'] = self.loss['total'] / self.config.TRAIN.ACCUMULATION_STEPS
    
    
    def forward_dh(self,fake,real):
        self.model_dh['model'].train()
        pred_fake=self.model_dh['model'](fake)
        pred_real=self.model_dh['model'](real)
        self.loss['DH_D']=self.criterion['DH'](pred_fake=pred_fake,pred_real=pred_real)
        self.loss['DH_D']=self.loss['DH_D'] 

    def forward_dl(self,fake,real):
        self.model_dl['model'].train()
        pred_fake=self.model_dl['model'](fake)
        pred_real=self.model_dl['model'](real)
        self.loss['DL_D']=self.criterion['DL'](pred_fake=pred_fake,pred_real=pred_real)
        self.loss['DL_D']=self.loss['DL_D'] 

    def forward_cmnet(self,content,is_train=True):
        if is_train:
            self.model_cmnet['model'].train()
        else:
            self.model_cmnet['model'].eval()
        
        
        
        #contrastive
        self.loss['IDENTITY'],self.loss['CONTRASTIVE']=self.model_cmnet['model'].module.self_supervised_loss(content.detach(),
                                                                                                        self.clean_freq,
                                                                                                        self.clean_position)
        self.memory_content_y=self.model_cmnet['model'](content,self.clean_freq,self.clean_position,self.clean_channel)
    
        with torch.no_grad():
            #if self.config.MODEL.TYPE == 'TransNet':
                #self.memory_content_y=self.memory_content_y.flatten(2).transpose(1,2).contiguous()
            self.memory_clean_reco=self.model['model'].module.clean_decoder(self.memory_content_y)
    
    def _optimize_step(self,idx,epoch,num_steps,inputs):
        if self.config.FINETUNE:
            self.forward_finetune(inputs)
        else:    
            self.forward(inputs)
        self._update_parameters(model=self.model,
                               idx=idx,epoch=epoch,num_steps=num_steps,
                               acumulation_steps=self.config.TRAIN.ACCUMULATION_STEPS,
                               loss=self.loss['total'],
                               clip_grad=self.config.TRAIN.CLIP_GRAD,
                               update_grad=(idx + 1) % self.config.TRAIN.ACCUMULATION_STEPS == 0
                               )
        if not self.config.FINETUNE and self.config.TRAIN.LOSS.IDENTITY!=0 and self.config.TRAIN.LOSS.CONTRASTIVE!=0:
            self.forward_cmnet(self.content_y)
            self._update_parameters(model=self.model_cmnet,
                                idx=idx,epoch=epoch,num_steps=num_steps,
                                acumulation_steps=self.config.TRAIN.ACCUMULATION_STEPS,
                                loss=self.loss['CONTRASTIVE']+self.loss['IDENTITY'],
                                #loss=self.loss['CONTRASTIVE'],
                                clip_grad=self.config.TRAIN.CLIP_GRAD,
                                update_grad=(idx + 1) % self.config.TRAIN.ACCUMULATION_STEPS == 0
                                )
            
        if not self.config.FINETUNE and self.config.TRAIN.LOSS.DH[1]!=0: 
            self.forward_dh(self.outputs,self.clean)
            self._update_parameters(model=self.model_dh,
                                idx=idx,epoch=epoch,num_steps=num_steps,
                                acumulation_steps=self.config.TRAIN.ACCUMULATION_STEPS,
                                loss=self.loss['DH_D'],
                                #clip_grad=self.config.TRAIN.CLIP_GRAD,
                                clip_grad=5.0,
                                #clip_grad=False,
                                update_grad=True,
                                )
        if not self.config.FINETUNE and self.config.TRAIN.LOSS.DL[1]!=0:
            self.forward_dl(self.generated_noisy,self.noisy)
            self._update_parameters(model=self.model_dl,
                                idx=idx,epoch=epoch,num_steps=num_steps,
                                acumulation_steps=self.config.TRAIN.ACCUMULATION_STEPS,
                                loss=self.loss['DL_D'],
                                #clip_grad=self.config.TRAIN.CLIP_GRAD,
                                clip_grad=5.0,
                                #clip_grad=False,
                                update_grad=True,
                                )
            

        torch.cuda.synchronize()
        
    def forward_one_epoch(self,epoch):
        num_steps=len(self.data_loader)
        self._reset()
        
        self.data_loader.sampler.set_epoch(epoch)
        for idx, data in enumerate(self._tqdm(self.data_loader,epoch)):
            self._optimize_step(idx,epoch,num_steps,data)
            updata_losses_meter(self.loss_meter,self.loss,data['clean'].size(0))

            if dist.get_rank() == 0 and self.config.VISUAL.FREQ != 0 and idx % self.config.VISUAL.FREQ == 0:
                self.visuals(epoch=epoch,idx=idx)
        
            #self.logger.info(" ".join(['Train']+["Epoch"]+[str(epoch)]+["{} {:.2e}".format(k,v.avg)for k, v in self.loss_meter.items()]))
            
        if epoch % self.config.PRINT_FREQ==0:
            self.logger.info(" ".join(['Train']+["Epoch"]+[str(epoch)]+["{} {:.2e}".format(k,v.avg)for k, v in self.loss_meter.items()]))
                
        if epoch % self.config.VAL_FREQ==0:
            self.vaild(epoch)
            if self.config.DATA.NOISE.ADD_NOISE  and self.metric_meter.get('nrmse',None) is not None and self.min_nrmse>self.metric_meter['nrmse'].avg:
                self.min_nrmse=self.metric_meter['nrmse'].avg

            if self.metric_meter.get('sbrc',None) is not None and self.max_snr_change <self.metric_meter['sbrc'].avg:
                self.max_snr_change=self.metric_meter['sbrc'].avg
        
        if dist.get_rank() == 0 :
            self._write_loss(epoch,self.writer)
            if  (epoch % self.config.SAVE_FREQ == 0 or epoch == (self.config.TRAIN.EPOCHS - 1)):
                self._save_checkpoint(epoch,self.logger,self.model,model_dl=self.model_dl,model_dh=self.model_dh,model_cmnet=self.model_cmnet) 
            if self.config.DATA.NOISE.ADD_NOISE :
                #if self.max_psnr==self.metric_meter['psnr'].avg:
                if self.min_nrmse==self.metric_meter['nrmse'].avg:
                
                    self._save_checkpoint(epoch,self.logger,self.model,model_dl=self.model_dl,model_dh=self.model_dh,model_cmnet=self.model_cmnet,is_best=True)
            else:
                if self.max_snr_change==self.metric_meter['sbrc'].avg:
                    self._save_checkpoint(epoch,self.logger,self.model,model_dl=self.model_dl,model_dh=self.model_dh,model_cmnet=self.model_cmnet,is_best=True)
            
    @torch.no_grad()
    def vaild(self,epoch):
        self._reset()
        
        for idx,data in enumerate(self._tqdm(self.data_loader_val,epoch)):
            if not self.config.FINETUNE:
                self.forward(data,is_train=False)
                self.forward_cmnet(self.content_y,is_train=False)
            else:
                self.forward_finetune(data,is_train=False)
            
            updata_losses_meter(self.loss_meter,self.loss,data['clean'].size(0))
            
            self.evaluate(self.noisy.size(0),self.noisy,self.outputs,self.target if hasattr(self,'target') else None)
                
            if dist.get_rank() == 0:
                self.visuals(epoch=epoch,idx=idx,prefix='val')
        if dist.get_rank()==0:
            self._write_loss(epoch,self.writer_val)
            self._write_metric(epoch,self.writer_val)
            
        
        self.logger.info(" ".join(['Val']+["Epoch"]+[str(epoch)]+["{} {:.2e}".format(k,v.avg)for k, v in self.metric_meter.items() if v.count!=0.]+["{} {:.2e}".format(k,v.avg)for k, v in self.loss_meter.items()]))
        
    
    

        