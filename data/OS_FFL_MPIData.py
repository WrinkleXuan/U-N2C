import os
from scipy import io
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange,repeat
import math
from random import randint
import mat73

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

class OSFFLMPIData(torch.utils.data.Dataset):
    def __init__(self,dir_path='/data/zwx/data/OS_FFL_MPIData/data_OS-FFL-MPI',
                 data_name='Sz_reco.mat',
                output_size=(32,32),
                base_freq=3000.,
                low_freq=120.,
                nums_bands=12,
                nums_harmonics=8,
                angles=30):
        sm=mat73.loadmat(os.path.join(dir_path,data_name))[data_name.split('.')[0]]
        self.base_freq=base_freq
        self.low_freq=low_freq
        self.nums_bands=nums_bands
        self.angles=angles
        self.nums_harmonics=nums_harmonics
        self.output_size=output_size
        
        self.sm,self.parameters=self.read_data(sm)
        assert self.sm.shape[0]==self.parameters.shape[0]

        self.sm_min=np.min(self.sm,axis=(2,3),keepdims=True)
        self.sm_max=np.max(self.sm,axis=(2,3),keepdims=True)
        
    def read_data(self,sm):
        
        h=int(math.sqrt(sm.shape[1]))
        w=int(math.sqrt(sm.shape[1]))
        #freq/=10e6
        #sm=np.concatenate((np.expand_dims(np.real(sm),1),np.expand_dims(np.imag(sm),1)),axis=1)
        sm=np.concatenate((np.expand_dims(sm[:sm.shape[0]//2],1),np.expand_dims(sm[sm.shape[0]//2:,],1)),axis=1)
        sm=rearrange(sm,'f c (h w)->f c h w',h=h,w=w)
        sidebands=np.arange(-self.nums_bands,self.nums_bands+1,step=1)*self.low_freq
        harmonic=np.arange(1,self.nums_harmonics+1,step=1)*self.base_freq
        harmonic=repeat(harmonic.reshape(-1,1),'n l->(n c) l',c=len(sidebands))
        sidebands=repeat(sidebands.reshape(-1,1),'n l->(c n) l',c=self.nums_harmonics)
        freq=harmonic+sidebands
        freq=freq[freq!=3000.0].reshape(-1,1)
        freq=repeat(freq,'n l ->(c n) l',c=self.angles)
        mask=np.logical_and(freq>=4560,freq<=22440).reshape(-1,)

        sm=sm[mask]
        freq=freq[mask]
        if (h,w)!=self.output_size:
            sm=F.interpolate(torch.tensor(sm),(self.output_size[0],self.output_size[1]),mode='bilinear')
            sm=sm.numpy()
        
        position=np.zeros((sm.shape[0],1))
        channel=np.zeros((sm.shape[0],1))
        
        parameters=np.concatenate((freq,channel,position),axis=1)
        return sm,parameters

    def normalize(self,x: torch.Tensor):
        data_max,_=torch.max(torch.max(x,dim=-1,keepdim=True)[0],dim=-2,keepdim=True)
        data_min,_=torch.min(torch.min(x,dim=-1,keepdim=True)[0],dim=-2,keepdim=True)
        #mean=torch.mean(x,dim=(-2,-1),keepdim=True)
        x=(x-data_min)/(data_max-data_min+1e-20)
        return x*2-1.0        

    def denormalize(self,index,x:np.ndarray):
        """
        x: x must be in sm_noisy
        """
        x=x*0.5+0.5
        data_min=self.sm_min[index]
        data_max=self.sm_max[index]
        return x*(data_max-data_min+1e-20)+data_min
    
    def __len__(self):
       
       
        return  self.sm.shape[0]
            
    def to_tensor(self, data):  
        if self.output_size[0]%4:
            data=resize_image_with_crop_or_pad(data,(data.shape[0],self.output_size[0]+4-self.output_size[0]%4,self.output_size[1]+4-self.output_size[1]%4),mode='edge')
        
        data = torch.FloatTensor(data)
        #data=F.interpolate(data,(self.output_size[0],self.output_size[1]),mode='bilinear')
        
        data=self.normalize(data)
        return data
    
    def to_numpy(self, data,index):
        #data*=energy
        data = data.detach().cpu().numpy()
        data = data.squeeze()
        if self.output_size[0]%4:
           
            data=resize_image_with_crop_or_pad(data,(data.shape[0],self.output_size[0],self.output_size[1]),mode='edge')
       
        data=self.denormalize(index,data)
        data=data.transpose(1,2,0)
        
        if data.ndim == 3:   
            data = data[:,:,0].squeeze()+1j*data[:,:,1].squeeze()
        return data
    
    def __getitem__(self,index):
        sm_noisy=self.to_tensor(self.sm[index])
    
        #clean_index=randint(0,self.sm_clean.shape[0]-1)
        #sm_clean=self.to_tensor(self.sm_clean[clean_index])
        data={'name':index,
              'noisy':sm_noisy,
              'freq_noise':torch.tensor(self.parameters[index][0],dtype=sm_noisy.dtype).unsqueeze(-1),
              'channel_noise':torch.tensor(self.parameters[index][1],dtype=torch.long).unsqueeze(-1),
              'position_noise':torch.tensor(self.parameters[index][2],dtype=torch.long).unsqueeze(-1),
              }
        return data
if __name__=='__main__':
    data=OSFFLMPIData(data_name='Sz_reco.mat')
    print(data.parameters[0][0])
    print(data.parameters[1][0])
    print(len(data))
    phatom=mat73.loadmat(os.path.join('/data/zwx/data/OS_FFL_MPIData/data_OS-FFL-MPI','2mm_dz_reco.mat'))['dz_reco']
    print(phatom.shape)
    
    #phatom=io.loadmat(os.path.join('/data/zwx/data/OS_FFL_MPIData/data_OS-FFL-MPI','b_S_m.mat'))['b_S_m']
    
    #print(phatom.shape)
    
    #print(len(data))
    #print(data.sm_clean.shape)
    #print(data[0]['noisy'].size())
    #print(data[23]['freq_noise'])
    import mat73
    sm=mat73.loadmat(os.path.join('/data/zwx/data/OS_FFL_MPIData/data_OS-FFL-MPI','Sz_reco.mat'))['Sz_reco']
    print(sm.shape)
    