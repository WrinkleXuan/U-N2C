import os
import os.path as path
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from einops import rearrange,repeat
import torch.nn.functional as F
import math
import random
from random  import randint,uniform
from scipy import ndimage
from scipy.linalg import orth



def resize_image_with_crop_or_pad(image, img_size=(2,32, 32), **kwargs):
    """Image resizing. Resizes image by cropping or padding dimension
     to fit specified size.
    Args:
        image (np.ndarray): image to be resized
        img_size (list or tuple): new image size
        kwargs (): additional arguments to be passed to np.pad
    Returns:
        np.ndarray: resized image
    """

    assert isinstance(image, (np.ndarray, np.generic))
    assert (image.ndim - 1 == len(img_size) or image.ndim == len(img_size)), \
        'Example size doesnt fit image size'

    # Get the image dimensionality
    rank = len(img_size)

    # Create placeholders for the new shape
    from_indices = [[0, image.shape[dim]] for dim in range(rank)]
    to_padding = [[0, 0] for dim in range(rank)]

    slicer = [slice(None)] * rank

    # For each dimensions find whether it is supposed to be cropped or padded
    for i in range(rank):
        if image.shape[i] < img_size[i]:
            to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
            to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            from_indices[i][0] = int(np.floor((image.shape[i] - img_size[i]) / 2.))
            from_indices[i][1] = from_indices[i][0] + img_size[i]

        # Create slicer object to crop or leave each dimension
        slicer[i] = slice(from_indices[i][0], from_indices[i][1])
    # Pad the cropped image to extend the missing dimension
    return np.pad(image[slicer[0],slicer[1],slicer[2]], to_padding, **kwargs)


def random_rot_flip(image):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k,axes=(1,2))
    axis = np.random.randint(1, 3)
    image = np.flip(image, axis=axis).copy()
    return image


def random_rotate(image):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False,axes=(1,2))
    return image


class OpenMPIData(torch.utils.data.Dataset):
    def __init__(self, dir_path="/data/zwx/data/OpenMPIData/SM_Denoising", data_name="calibrations_7.npz",aux_noisy_data_name=None,aux_clean_data_name=None,
        random_rot_flip=False,mode='train',output_size=(32,32),add_noise=False,snr=1.,alpha=1.,noise_type='gaussion_noise'):
        super(OpenMPIData, self).__init__()
        
        data=np.load(os.path.join(dir_path,data_name))
        self.random_rot_flip=random_rot_flip
        self.output_size=output_size
        self.mode=mode
        self.add_noise=add_noise
        self.snr=snr
        self.alpha=alpha
        self.noise_type=noise_type
        self.aux_sm_nosiy=None 
        self.aux_sm_clean=None 
        self.aux_parameters_nosiy=None 
        self.aux_parameters_clean=None 
       
            
        self.sm_size=data['size']
        if self.add_noise :
            self.sm_noisy ,self.parameters_noisy= self.read_data(data["sm_clean"].astype(np.float32),
                                                                 data['freq_clean'].astype(np.float32),
                                                                 data['channel_clean'].astype(np.float32),
                                                                 data['sequence'],
                                                                 self.sm_size)
            if aux_clean_data_name is not None:
            
                self.sm_clean ,self.parameters_clean= self.read_aux(dir_path,aux_clean_data_name)
            
            else:
                raise KeyError("The aux_clean_data_name must not be None.")

        else:
            self.sm_noisy,self.parameters_noisy = self.read_data(data["sm_noisy"].astype(np.float32),
                                                                 data['freq_noisy'].astype(np.float32),
                                                                 data['channel_noisy'].astype(np.float32),
                                                                 data['sequence'],
                                                                 self.sm_size)
            self.sm_clean,self.parameters_clean= self.read_data(data["sm_clean"].astype(np.float32),
                                                                data['freq_clean'].astype(np.float32),
                                                                data['channel_clean'].astype(np.float32),
                                                                data['sequence'],
                                                                self.sm_size) # F * C * N  N=z * y * x
            if aux_clean_data_name is not None: 
                self.aux_sm_clean,self.aux_parameters_clean=self.read_aux(dir_path,aux_clean_data_name)
            
        if self.mode=='train' and aux_noisy_data_name is not None:
            if self.add_noise:
                self.aux_sm_nosiy,self.aux_parameters_nosiy=self.read_aux(dir_path,aux_noisy_data_name,aux_type='clean')
                
            else:
                self.aux_sm_nosiy,self.aux_parameters_nosiy=self.read_aux(dir_path,aux_noisy_data_name,aux_type='noisy')
                
        assert self.sm_clean is not None
        assert self.sm_noisy is not None
        assert hasattr(self,'aux_sm_clean')
        assert hasattr(self,'aux_sm_nosiy')
        assert self.sm_noisy.shape[0]==self.parameters_noisy.shape[0]
        assert self.sm_clean.shape[0]==self.parameters_clean.shape[0]
        if self.aux_sm_nosiy is not None and self.aux_parameters_nosiy is not None:
            assert self.aux_sm_nosiy.shape[0]==self.aux_parameters_nosiy.shape[0]
        if self.aux_sm_clean is not None and self.aux_parameters_clean is not None:
            assert self.aux_sm_clean.shape[0]==self.aux_parameters_clean.shape[0]

        if self.mode=='test':
            self.sm_noisy_min=np.min(self.sm_noisy,axis=(1,2),keepdims=True)
            self.sm_noisy_max=np.max(self.sm_noisy,axis=(1,2),keepdims=True)
            
    def __len__(self):
       
       
        if self.aux_sm_nosiy is not None:
            
            return self.sm_noisy.shape[0]+self.aux_sm_nosiy.shape[0]
        else:
            return self.sm_noisy.shape[0]
            
    
    def read_data(self,sm,freq,channel,sequence,size):
        #energy=self.computeEnergy(sm)
        
        #sm=sm*self.energy
        if sequence=='3d':
            position=np.arange(0,size[2])
            position=repeat(position,'d-> (f d)',f=sm.shape[0]).reshape(-1,1)
        else:
            position=np.zeros((sm.shape[0]*size[2],1))
        channel=repeat(channel,'f->(f d)',d=size[2]).reshape(-1,1)
        freq/=10e6
        freq=repeat(freq,'f->(f d)',d=size[2]).reshape(-1,1)
        assert channel.shape==freq.shape and freq.shape==position.shape
        
        parameters=np.concatenate((freq,channel,position),axis=1)
                
        
        sm=rearrange(sm,'f c (d h w)->(f d) h w c',h=size[0],w=size[1])
        
        return sm,parameters

    def read_aux(self,dir_path,aux_data_name,aux_type='clean'):
        if isinstance(aux_data_name,(list,str)):
            if isinstance(aux_data_name,str):
                aux_data=np.load(os.path.join(dir_path,aux_data_name))
                size=aux_data['size']
                data=aux_data['sm_'+aux_type].astype(np.float32)
                freq=aux_data['freq_'+aux_type].astype(np.float32)
                channel=aux_data['channel_'+aux_type].astype(np.float32)
                sequence=data['sequence']
            else:
                data=[]
                freq=[]
                channel=[]
                for idx,s in enumerate(aux_data_name):
                    aux_data=np.load(os.path.join(dir_path,s))
                    if idx: 
                        assert np.array_equal(size,aux_data['size'])
                        assert sequence==aux_data['sequence']
                    size=aux_data['size']
                    sequence=aux_data['sequence']
                    data.append(aux_data['sm_'+aux_type].astype(np.float32))
                    freq.append(aux_data['freq_'+aux_type].astype(np.float32))
                    channel.append(aux_data['channel_'+aux_type].astype(np.float32))
                data=np.concatenate(data)
                freq=np.concatenate(freq)
                channel=np.concatenate(channel)
        else:
            raise ValueError("We only support List or str Now.")

        return self.read_data(data,freq,channel,sequence,size)
    
    
    def computeEnergy(self,data):
        """
        Calculate the energy of the system matrix
        """
        energy=1.0/np.linalg.norm(data,axis=(-2,-1),keepdims=True)
        return energy
    def normalize(self,x: torch.Tensor):
        data_max,_=torch.max(torch.max(x,dim=-1,keepdim=True)[0],dim=-2,keepdim=True)
        data_min,_=torch.min(torch.min(x,dim=-1,keepdim=True)[0],dim=-2,keepdim=True)
        #mean=torch.mean(x,dim=(-2,-1),keepdim=True)
        x=(x-data_min)/(data_max-data_min+1e-13)
        return x*2-1.0
    
    def denormalize(self,index,x:np.ndarray):
        """
        x: x must be in sm_noisy
        """
        x=x*0.5+0.5
        data_min=self.sm_noisy_min[index]
        data_max=self.sm_noisy_max[index]
        return x*(data_max-data_min+1e-13)+data_min
        
    def denormalize_per_batch(self,index,x:torch.tensor):
        """
        x: x must be in sm_noisy
        """
        x=x*0.5+0.5
        data_min=torch.tensor(self.sm_noisy_min[index]).cuda().transpose(1,3)
        
        data_max=torch.tensor(self.sm_noisy_max[index]).cuda().transpose(1,3)
        return x*(data_max-data_min+1e-13)+data_min
        
    def gaussion_noise(self,data,snr=1,alpha=1):
        if isinstance(snr,list):
            snr=uniform(snr[0],snr[1])
        power=torch.mean(data**2)
        noise_power=torch.sqrt(power)/snr
        
        noise = torch.randn_like(data)*noise_power
        #noise=np.clip(noise,0,1)
        gt=data.clone()
        data=data+noise
        
        return gt,data

    def poission_gaussion_noise(self,data,snr=1,alpha=1):
        if isinstance(snr,list):
            snr=uniform(snr[0],snr[1])
        
        power=torch.mean(data**2)
        noise_power=torch.sqrt(power)/snr
        #alpha_power=torch.sqrt(power)/alpha
        noise_signal_independent = torch.randn_like(data)*noise_power
        #noise=np.clip(noise,0,1) 
        alpha_power=torch.sqrt(power)/alpha
        noise_signal_dependent=torch.randn_like(data)*alpha_power
    
        gt=data.clone()
        data=data+noise_signal_independent+data*noise_signal_dependent
        
        return gt,data

    def multivariate_gaussion_noise(self,data,snr=1):
        N=data.size(0)
        power=torch.mean(data**2)
        noise_power=torch.sqrt(power)/snr
        
        U=orth(torch.rand(N,N))
        D=torch.diag(torch.rand(N))
        tmp=U.T@D@U
        tmp=(noise_power**2)*tmp
        noise_sigma=torch.abs(noise_sigma)
        sampler=torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(N),noise_sigma)
        noise=sampler.sample((data.size(1),data.size(2)))
        gt=data.clone()
        data=data+noise
        
        return gt,data



    def to_tensor(self, data,add_noise=False,snr=1.,alpha=1.):  
        if data.ndim == 2: data = data[np.newaxis, ...]
        elif data.ndim == 3: data = data.transpose(2, 0, 1) # c h w 
        if self.mode=='train' and self.random_rot_flip and random.random() > 0.5:
            data = random_rot_flip(data)
            data = random_rotate(data)    
        data=resize_image_with_crop_or_pad(data,(data.shape[0],self.output_size[0],self.output_size[1]),mode='edge')
        data = torch.FloatTensor(data)
        data=self.normalize(data)
        if add_noise:
            if hasattr(self,self.noise_type):
                noise_func=getattr(self,self.noise_type)
            else:
                raise NotImplementedError("This noise type is not  supported !!!")

            gt,data=noise_func(data,snr,alpha)
            return  gt,data
        else:
            return data
    
    def to_numpy(self, data,index):
        #data*=energy
        data = data.detach().cpu().numpy()
        data = data.squeeze()
        data=resize_image_with_crop_or_pad(data,(data.shape[0],self.sm_size[0],self.sm_size[1]),mode='edge')
        data=data.transpose(1,2,0)
        data=self.denormalize(index,data)
        if data.ndim == 3:   
            data = data[:,:,0].squeeze()+1j*data[:,:,1].squeeze()
        return data

    
    def __getitem__(self, index):
       
        sm_noisy = self.sm_noisy[index] if index<self.sm_noisy.shape[0] else self.aux_sm_nosiy[index-self.sm_noisy.shape[0]]
        parameters_noisy=self.parameters_noisy[index] if index<self.parameters_noisy.shape[0] else self.aux_parameters_nosiy[index-self.parameters_noisy.shape[0]]
        
        if self.add_noise:
            gt,sm_noisy=self.to_tensor(sm_noisy,add_noise=self.add_noise,snr=self.snr,alpha=self.alpha)
    
        else:
            sm_noisy=self.to_tensor(sm_noisy,add_noise=self.add_noise)
        len_aux_sm_clean=self.aux_sm_clean.shape[0] if self.aux_sm_clean is not None else 0      
        sm_clean_index = randint(0,self.sm_clean.shape[0]+len_aux_sm_clean-1)
        sm_clean=self.sm_clean[sm_clean_index] if sm_clean_index<self.sm_clean.shape[0] else self.aux_sm_clean[sm_clean_index-self.sm_clean.shape[0]]    
        parameters_clean=self.parameters_clean[sm_clean_index] if sm_clean_index<self.parameters_clean.shape[0] else self.aux_parameters_clean[sm_clean_index-self.parameters_clean.shape[0]]    
        
        sm_clean=self.to_tensor(sm_clean)

        data={"name": index, 
            'freq_noise':torch.tensor(parameters_noisy[0],dtype=sm_noisy.dtype).unsqueeze(-1),
            'freq_clean':torch.tensor(parameters_clean[0],dtype=sm_clean.dtype).unsqueeze(-1),
            'channel_noise':torch.tensor(parameters_noisy[1],dtype=torch.long).unsqueeze(-1),
            'channel_clean':torch.tensor(parameters_clean[1],dtype=torch.long).unsqueeze(-1),
            'position_noise':torch.tensor(parameters_noisy[2],dtype=torch.long).unsqueeze(-1),
            'position_clean':torch.tensor(parameters_clean[2],dtype=torch.long).unsqueeze(-1),
            "noisy": sm_noisy,
            "clean": sm_clean,
            'target':gt if self.add_noise else torch.zeros_like(sm_noisy)}
        
    
        return data
if __name__=="__main__":
    data=OpenMPIData(data_name='calibrations_7.npz',
                     random_rot_flip=True,mode='train',
                     #aux_noisy_data_name=['calibrations_8.npz','calibrations_9.npz'],
                     aux_clean_data_name=["calibrations_10.npz","calibrations_11.npz"],
                     add_noise=False,)
    print(len(data))

    #print(data.sm_noisy.shape)
    print(data.sm_clean.shape[0]+data.aux_sm_clean.shape[0])
    #print(data.aux_sm_nosiy.shape)
    #print(data.aux_sm_clean.shape)
    
    print(data[100]["noisy"].size())
    print(data[100]['channel_noise'])
    print(data[100]['position_noise'])
    print(data[100]['freq_noise'].size())
    
    #print(data[100]["clean"].size())
    #print(data[100]["target"].size())
    
    #print(data[355384]["noisy"].size())
    #print(data.to_numpy(data[100]['noisy']).shape)

