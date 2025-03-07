import sys
sys.path.append('MPI/')

import torch
import h5py
import urllib
import requests
import re 
import os

import glob
import socket
import h5py
import matplotlib.pyplot as plt
import numpy as np
#np.set_printoptions(threshold=np.inf)

from kaczmarzReg import kaczmarzReg
#import kaczmarzReg
from einops import rearrange,repeat

import warnings

from glob import iglob
import  logging

formatter = logging.Formatter("%(asctime)s:%(levelname)s:[%(filename)s]:%(message)s")
logger=logging.getLogger('sm_denoising')
logger.setLevel(level = logging.INFO)
fh = logging.FileHandler("/data/zwx/data/OpenMPIData/sm_denoising.log")
sh =logging.StreamHandler()
sh.setLevel(logging.INFO)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
sh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(sh)
    

socket.setdefaulttimeout(10)
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
def isFile(url):
    """
    判断一个链接是否是文件

    """
    if url.endswith('/'):
        return False
    else:
        return True
def download(url,savedir):
    """"
    下载文件
    """
    full_name=url.split("//")[-1]
    
    filename=full_name.split('/')[-1]
    #dirname=savedir.join(full_name.split('/')[3:-1])
    dirname=os.path.join(savedir,'/'.join(full_name.split('/')[3:-1]))
    if os.path.exists(dirname):
        pass
    else:
        os.makedirs(dirname,exist_ok=True)
    try:
        urllib.request.urlretrieve(url,dirname+'/'+filename)
    except socket.timeout:
        count = 1
        while count <= 5:
            try:
                urllib.request.urlretrieve(url,dirname+'/'+filename)                                             
                break
            except socket.timeout:
                err_info = 'Reloading for %d time'%count if count == 1 else 'Reloading for %d times'%count
                print(err_info)
                count += 1
        if count > 5:
            print("downloading picture fialed!")

def get_url(base_url):
    '''
    :param base_url:给定一个网址
    :return: 获取给定网址中的所有链接
    '''
    text = ''
    try:
        text = requests.get(base_url).text
    except Exception as e:
        print("error - > ",base_url,e)
        pass
    #reg = '<a href="(.*)">.*</a>'
    reg = 'alt="\[(.*)\]"></td><td><a href="(.*)">.*</a>'
    urls = [base_url + url[1] for url in re.findall(reg, text) if url[0] != 'PARENTDIR']
    
    return urls

def get_file(url,savedir):
    """
    递归下载网站的文件
    """
    if isFile(url):
        print(url)
        download(url,savedir)
    #    try:
    #        download(url,savedir)
    #    except:
    #        print("Download Failed!")
    else:
        urls=get_url(url)
        for u in urls:
            get_file(u,savedir)

def Download_OpenMPIDataset_from_URL(url="https://media.tuhh.de/ibi/openMPIData/",save_dir="data/OpenMPIData/"):

    if os.path.exists(save_dir):
        if len(glob.glob(save_dir+"*"))==0:
            raise Exception("The dataset exists")
    else:
        #os.makedirs(save_dir)
        get_file(url,save_dir)


class MPI():
    def __init__(self,measurements=None,system_matrix=None):

        self.fmeas=h5py.File(measurements,'r') if measurements is not None else None
        
        self.phantom_name= measurements.split('/')[-2] if self.fmeas is not None else None
        self.sm=None
        self.fsm=h5py.File(system_matrix,'r') if system_matrix is not None else None
        self.u=None
        self.u_bg=None
    def _SM_preprocess(self):
        if self.fsm is None:
            raise Exception("The file of SystemMatrix is not existence")
        #print(self.fsm['/measurement/frequencySelection'].shape)
        if self.isFrequencySelection:
            warnings.warn("this sysyem matrix has been selected!!!!") #[()]

        self.sm=self.fsm['measurement/data']
        print("the spatial size of sm is {} order is {}".format(self.fsm['/calibration/size'][:],self.fsm['/calibration/order/'][()])) #37*37*37
        
        self.sm=self.sm[:,:,:,:].squeeze()
        print("the size of sm == {}".format(self.sm.shape))
        """
        if self.isFramePermutation:
            print("Frame rePermutation !!!")
            
            frame_order=self.framePermutation
            self.sm=self.sm[:,:,frame_order[frame_order<self.sm.shape[-1]]]
        
            print("the size of sm == {}".format(self.sm.shape))
          
        """

        if not self.fsm['/measurement/isFourierTransformed'][()]: 
            print("FourierTransformed!!!")
            self.sm = np.fft.rfft(self.sm)
        
        
        #if self.fsm['measurement/isBackgroundCorrected'][()]:
            
        isBG = self.fsm['/measurement/isBackgroundFrame'][:].view(bool)
        self.sm=self.sm[:,:,isBG == False]
        
        print("the size of sm == {}".format(self.sm.shape))
            
        #print("Before isFourierTransformed the size of sm == {}".format(self.sm.shape))
        #sm=sm+np.random.randn(sm.shape)*0.1
        #print( self.fsm['measurement/framePermutation'][:])
        #print(np.max(frame_order[frame_order<self.sm.shape[-1]]))
            
        
        print("the size of sm == {}".format(self.sm.shape))
        
    def _SM_revovery(self,sm_denoised,snr_threshold=None):
        if snr_threshold is None: return
        
        #print("the size of condition == {}".format(np.linalg.cond(self.sm)))
        
        sm_denoised=rearrange(sm_denoised,"(n d) h w->n (d h w)",d=self.fsm['/calibration/size'][0])
        #print("the size of sm_denoised == {}".format(sm_denoised.shape))
        self.sm=self.sm.reshape((self.sm.shape[0]*self.sm.shape[1],-1))
        
        print("the size of sm is {}".format(self.sm.shape))
        
        snr=self.fsm['/calibration/snr'][:].squeeze()
        snr=snr.reshape((snr.shape[0]*snr.shape[1],))
        print("the size of snr == {}".format(snr.shape)) 
        if not isinstance(snr_threshold,tuple):
            print("the size of snr>={} == {}".format(snr_threshold,snr[snr<snr_threshold].shape)) 
            self.sm[snr<snr_threshold,:]=sm_denoised

            assert snr[snr<snr_threshold].shape[0]==sm_denoised.shape[0]
        else:
            print("the size of {}<=snr<{} == {}".format(snr_threshold[0],snr_threshold[1],snr[np.logical_and(snr>=snr_threshold[0],snr<snr_threshold[1])].shape))     
            #assert np.all(self.sm[np.logical_and(snr>snr_threshold[0],snr<snr_threshold[1]),:]==sm_denoised)
            self.sm[np.logical_and(snr>=snr_threshold[0],snr<snr_threshold[1]),:]=sm_denoised
            assert snr[np.logical_and(snr>=snr_threshold[0],snr<snr_threshold[1])].shape[0]==sm_denoised.shape[0]
        
        
        print("the size of sm is {}".format(self.sm.shape))
        #print("the size of condition == {}".format(np.linalg.cond(self.sm)))

    def _Mea_preprocess(self):
        if self.fsm is None:
            raise Exception("The file of Measurement is not existence")
        self.u = self.fmeas['/measurement/data']
        print("the size of Measurement == {}".format(self.u.shape))
        if self.u.shape[0]==1:
            self.u = self.u[:,:,:,:].squeeze()
            isBG = self.fmeas['/measurement/isBackgroundFrame'][:].view(bool)

            self.u_bg=self.u[isBG,:,:]
            self.u=self.u[isBG == False,:,:]
            
        else:
            isBG = self.fmeas['/measurement/isBackgroundFrame'][:].view(bool)
            self.u_bg=self.u[isBG,:,:,:]
            
            self.u=self.u[isBG == False,:,:,:]
            
            self.u=self.u.squeeze()
            self.u_bg=self.u_bg.squeeze()
        
        
        print("the size of Measurement == {}".format(self.u.shape))
        print("the size of Measurement_BG == {}".format(self.u_bg.shape))
        
        if not self.fmeas['/measurement/isFourierTransformed'][()]: 
            self.u = np.fft.rfft(self.u,axis=-1) #傅立叶变换 时域--->频域
            self.u_bg = np.fft.rfft(self.u_bg,axis=-1) #傅立叶变换 时域--->频域
            print("FourierTransformed!!!")
        print("the size of Measurement == {}".format(self.u.shape))
        print("the size of Measurement_BG == {}".format(self.u_bg.shape))
         # isFrequencySelection 
        if self.fsm['/measurement/isFrequencySelection'][()]:
            
            self.u=self.u[:,:,self.fsm['/measurement/frequencySelection'][:].view(int)-1]
            self.u_bg=self.u_bg[:,:,self.fsm['/measurement/frequencySelection'][:].view(int)-1]
            print("FrequencySelection!!!")
        print("the size of Measurement == {}".format(self.u.shape))
        print("the size of Measurement_BG == {}".format(self.u_bg.shape))

    def _SNR_filter(self,snr_threshold=None):
        if snr_threshold is not None:
            
            snr=self.fsm['/calibration/snr'][:].squeeze()
            snr=snr.reshape((snr.shape[0]*snr.shape[1],))
            print("the size of snr == {}".format(snr.shape)) 
            print("the size of snr>={} == {}".format(snr_threshold,snr[snr>=snr_threshold].shape)) 
       
        
        # merge frequency and receive channel dimensions
        if self.sm.ndim==3: self.sm=self.sm.reshape((self.sm.shape[0]*self.sm.shape[1],-1))
        self.u=self.u.reshape((-1,self.u.shape[1]*self.u.shape[2]))
        assert self.u.shape[-1]==self.sm.shape[0]
        self.u_bg=self.u_bg.reshape((-1,self.u_bg.shape[1]*self.u_bg.shape[2]))
        print("the size of sm is {}".format(self.sm.shape))
        print("the size of measurement is {}".format(self.u.shape))
        print("the size of measurement_BG is {}".format(self.u_bg.shape))
        if snr_threshold is not None:
            
            self.sm=self.sm[snr>=snr_threshold,:]
        
            self.u=self.u[:,snr>=snr_threshold]
            self.u_bg=self.u_bg[:,snr>=snr_threshold]
        print("the size of measurement is {}".format(self.u.shape))
        print("the size of measurement_BG is {}".format(self.u_bg.shape))
        
            

        self.u=np.mean(self.u[:5,:],axis=0)
        self.u_bg=np.mean(self.u_bg[:5,:],axis=0)
        #self.u=np.mean(self.u[:,:],axis=0)
        #self.u_bg=np.mean(self.u_bg[:,:],axis=0)
        self.u=self.u-self.u_bg
        print(self.sm.shape)
        
        print(self.u.shape)
        assert self.sm.shape[0]==self.u.shape[0]
    
    @property
    def snr(self):
        snr=self.fsm['/calibration/snr'][:].squeeze()
        #snr=snr.reshape((snr.shape[0]*snr.shape[1],))
        return   snr

    @property
    def freq(self):
        # generate frequency vector
        numFreq = np.round(self.fsm['/acquisition/receiver/numSamplingPoints'][()]/2)+1
        rxBandwidth = self.fsm['/acquisition/receiver/bandwidth'][()]
        freq = np.arange(0,numFreq)/(numFreq-1)*rxBandwidth
        if self.fsm['/measurement/isFrequencySelection'][()]:
            freq=freq[self.fsm['/measurement/frequencySelection'][:].view(int)-1]
            print("FrequencySelection in freq!!!")
        
        return  freq
    @property
    def isFrequencySelection(self):
        return  self.fsm['/measurement/isFrequencySelection'][()]
    @property
    def isFramePermutation(self):
        return self.fsm['/measurement/isFramePermutation'][()]
    
    @property 
    def framePermutation(self):
        return self.fsm['/measurement/framePermutation'][:].view(int)-1
    
    @property
    def SM(self):
        self._SM_preprocess()
        return self.sm
        #return self.sm.reshape((self.sm.shape[0]*self.sm.shape[1],-1))
    
    @property
    def sm_size(self):
        return  self.fsm['/calibration/size'][:]
    
    @property
    def baseFreq(self):
        return self.fsm['/acquisition/drivefield/baseFrequency'][()]
    
    @property
    def divider(self):
        #Divider of the baseFrequency to determine the drive ﬁeld frequencies
        return self.fsm['/acquisition/drivefield/divider'][()][0]
    
    @property
    def order(self):
        flag=True
        if self.fsm['/calibration/order/'][()]!='xyz':
            flag=False
        return flag

    def row_energy(self,sm):
        """
        Calculate the energy of the system matrix
        SM: the size of f * n
        """
        energy=1.0/np.linalg.norm(sm,axis=-1,keepdims=True)
        return energy
    
    def gaussion_noise(self,data,noise_poser):
        
        
        noise = np.random.randn(*data.shape)*noise_poser
        #noise=np.clip(noise,0,1)
        data=data+noise
        
        return data

    def poission_gaussion_noise(self,data,noise_power,alpha=1.):
    
        noise_signal_independent = np.random.randn(*data.shape)*noise_power
        #noise=np.clip(noise,0,1)
        noise_signal_dependent=np.random.randn(*data.shape)*alpha
        data=data+noise_signal_independent+data*noise_signal_dependent
        
        return data
 
    def reco_2d(self,sm_denoised=None,snr_threshold=None,reco_name='traget'):
        self._SM_preprocess()
        self._Mea_preprocess()
        if sm_denoised is not None: self.SM_revovery(sm_denoised,snr_threshold)
        self._SNR_filter(snr_threshold=snr_threshold if not isinstance(snr_threshold,tuple) else snr_threshold[0])
        self.sm=rearrange(self.sm,"f (d h w)->d f (h w)",h=self.fsm['/calibration/size'][:][0],w=self.fsm['/calibration/size'][:][1])
        i=0
        img=[]
        #phantom_name=self.fmeas.split('/')[-1].split('.')[0]
        
        if not os.path.exists('../results/OpenMPI/2D/{}/reco_{}/z_direction'.format(self.phantom_name,reco_name)): os.makedirs('../results/OpenMPI/2D/{}/reco_{}/z_direction'.format(self.phantom_name,reco_name))
        for s in self.sm[:]:
            c = kaczmarzReg(s,self.u,3,np.linalg.norm(s,ord='fro')*1e-3,False,True,True)
            N=self.fsm['/calibration/size'][:]
            c=c.reshape(N[0],N[1])
            c_real=np.real(c)
            #print(c_real.shape)
            img.append(c_real)
            plt.imsave(os.path.join('../results/OpenMPI/2D/{}/reco_{}/z_direction'.format(self.phantom_name,reco_name),'{}_{}.png'.format(i,'z')),c_real,cmap='gray')
            i+=1
        print("The reconstruction is over!")
        return np.array(img)

    
    def reco_3d(self,sm_denoised=None,snr_threshold=None,reco_name='traget',add_noise=False,snr_value=1,noise_type='gaussion_noise',output_dir=''):
        self._SM_preprocess()
        self._Mea_preprocess()
        if sm_denoised is not None: 
            assert reco_name=='denoised' or 'noised'
            self._SM_revovery(sm_denoised,snr_threshold)
        self._SNR_filter(snr_threshold=snr_threshold if not isinstance(snr_threshold,tuple) else snr_threshold[0])
       
        c = kaczmarzReg(self.sm,self.u,3,np.linalg.norm(self.sm,ord='fro')*1e-3,False,True,True)
        print(c.shape)
        N=self.fsm['/calibration/size'][:]
        #print(N)
        c=c.reshape(-1,N[0],N[1]) #为复数
        #c=c.reshape(33,27,33) #为复数
            


        c_real=np.real(c)
        output_dir=os.path.join(output_dir,'results/OpenMPI/3D/{}/reco_{}'.format(self.phantom_name,reco_name))
        if not os.path.exists(output_dir+'/x_direction'): os.makedirs(output_dir+'/x_direction')
        if not os.path.exists(output_dir+'/y_direction'): os.makedirs(output_dir+'/y_direction')
        if not os.path.exists(output_dir+'/z_direction'): os.makedirs(output_dir+'/z_direction')
        
        for i in range(c_real.shape[2]):
            plt.imsave(os.path.join(output_dir+'/z_direction','{}_{}.png'.format(i,'z')),c_real[:,:,i],cmap='gray')
            
        for i in range(c_real.shape[1]):
            plt.imsave(os.path.join(output_dir+'/y_direction','{}_{}.png'.format(i,'y')),c_real[:,i,:],cmap='gray')
        for i in range(c_real.shape[0]):
            plt.imsave(os.path.join(output_dir+'/x_direction','{}_{}.png'.format(i,'x')),c_real[i,:,:],cmap='gray')
        z=np.max(c_real,axis=2)
        z=(z-np.min(z))/(np.max(z)-np.min(z))
        plt.imsave(os.path.join(output_dir+'/z_direction','{}.png'.format('z')),z,cmap='gray')
        
        y=np.max(c_real,axis=1)
        y=(y-np.min(y))/(np.max(y)-np.min(y))
        plt.imsave(os.path.join(output_dir+'/y_direction','{}.png'.format('y')),y,cmap='gray')
        
        x=np.max(c_real,axis=0)
        x=(x-np.min(x))/(np.max(x)-np.min(x))
        plt.imsave(os.path.join(output_dir+'/x_direction','{}.png'.format('x')),x,cmap='gray')
        
        print("The reconstruction is over!")
        return (c_real-np.min(c_real))/(np.max(c_real)-np.min(c_real))

def OpenMPI_SM_Denoising_Dataset(sm_dir=None,save_dir=None,snr_threshold=5,tracer_name=None,sequence=None):
    sm_remain=None
    freq_remain=None
    channel_remain=None
    # snr_threshold[0]<= <snr_threshold[1]
    if tracer_name is None: raise Exception("The name of tracer is not existence")
    if sequence is None: raise Exception("The name of sequence is not existence")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    data=MPI(system_matrix=sm_dir)
    sm=data.SM
    channel=sm.shape[0]
    freq=data.freq
    freq=repeat(freq,'n ->(c n)',c=sm.shape[0])
    
    channel=np.arange(0,sm.shape[0]).reshape(sm.shape[0],)
    channel=repeat(channel,'c-> (c n)',n=sm.shape[1])
        
    
    sm_size=data.fsm['/calibration/size'][:]
    snr=data.fsm['/calibration/snr'][:].squeeze()
    snr=snr.reshape((snr.shape[0]*snr.shape[1],))
    sm=sm.reshape((sm.shape[0]*sm.shape[1],-1))

    #row energy normalization
    #sm=sm*data.row_energy(sm)
    if not isinstance(snr_threshold,tuple):
        sm_noisy=sm[snr<snr_threshold,:]
        freq_noisy=freq[snr<snr_threshold]
        channel_noisy=channel[snr<snr_threshold]
        
        sm_clean=sm[snr>=snr_threshold,:]
        freq_clean=freq[snr>=snr_threshold]
        channel_clean=channel[snr>=snr_threshold]
        
    else:
        sm_remain=sm[snr<snr_threshold[0],:]
        freq_remain=freq[snr<snr_threshold[0]]
        channel_remain=channel[snr<snr_threshold[0]]
        
        sm_noisy=sm[np.logical_and(snr>=snr_threshold[0], snr<snr_threshold[1]),:]
        freq_noisy=freq[np.logical_and(snr>=snr_threshold[0], snr<snr_threshold[1])]
        channel_noisy=channel[np.logical_and(snr>=snr_threshold[0], snr<snr_threshold[1])]
        
        sm_clean=sm[snr>=snr_threshold[1],:]
        freq_clean=freq[snr>=snr_threshold[1]]
        channel_clean=channel[snr>=snr_threshold[1]]

    sm_noisy=np.concatenate((np.expand_dims(np.real(sm_noisy),1),np.expand_dims(np.imag(sm_noisy),1)),axis=1)
    sm_clean=np.concatenate((np.expand_dims(np.real(sm_clean),1),np.expand_dims(np.imag(sm_clean),1)),axis=1)
    if sm_remain is not None:
        sm_remain=np.concatenate((np.expand_dims(np.real(sm_remain),1),np.expand_dims(np.imag(sm_remain),1)),axis=1)
    
    logger.info("the size of SM == {}".format(sm.shape))
    logger.info("the size of SNR== {}".format(snr.shape))
    logger.info("the size of Freq== {}".format(freq.shape))
    logger.info("the size of Channel== {}".format(channel.shape))
    
    logger.info("the size of sm_remain=={}".format(sm_remain.shape if sm_remain is not None else 'NONE'))
    logger.info("the size of sm_noisy=={}".format(sm_noisy.shape))
    logger.info("the size of sm_clean=={}".format(sm_clean.shape))
    logger.info("the grid of sm_size=={}".format(sm_size))
    
    
    logger.info("the size of freq_remain=={}".format(freq_remain.shape if freq_remain is not None else 'NONE'))
    logger.info("the size of freq_noisy=={}".format(freq_noisy.shape))
    logger.info("the size of freq_clean=={}".format(freq_clean.shape))
    
    logger.info("the size of channel_remain=={}".format(channel_remain.shape if channel_remain is not None else 'NONE'))
    logger.info("the size of channel_noisy=={}".format(channel_noisy.shape))
    logger.info("the size of channel_clean=={}".format(channel_clean.shape))
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    name=sm_dir.split('/')[-2]+'_'+sm_dir.split('/')[-1].split('.')[0]
    np.savez(save_dir+'/'+name+'.npz',tracer=tracer_name,
             sequence=sequence,
             #sm_remain=sm_remain,
             sm_noisy=sm_noisy,
             sm_clean=sm_clean,
             freq_noisy=freq_noisy,
             freq_clean=freq_clean,
             #freq_remain=freq_remain,
             channel_noisy=channel_noisy,
             channel_clean=channel_clean,
             snr=snr,
             size=sm_size,
             channel=channel)
        
if __name__=="__main__":
    
    path=iglob('/data/zwx/data/OpenMPIData/calibrations/*.mdf')
    tracer_name='Perimag'
    sequence='3D'
    
    #for i_path in path:
    #    index=int(i_path.split('/')[-1].split('.')[0])
    #    if index== 1 or index==4 or index==17 or index== 16: continue

    #    tracer_name='Synomag-D' if index==7 else 'Perimag'
    #    sequence='2d' if index==(5 or 2) else '3d'
    #    logger.info("calibration== {} tracer =={} sequence =={}".format(index,tracer_name,sequence))
        
    #    OpenMPI_SM_Denoising_Dataset(i_path,"/data/zwx/data/OpenMPIData/SM_Denoising",(2,5) if index!=5 else (2,20),tracer_name=tracer_name,sequence=sequence)
    
    OpenMPI_SM_Denoising_Dataset('/data/zwx/data/OpenMPIData/calibrations/5.mdf',"/data/zwx/data/OpenMPIData/SM_Denoising",(2,100),tracer_name=tracer_name,sequence=sequence)
    
    """
    data=MPI(system_matrix='/data/zwx/data/OpenMPIData/calibrations/8.mdf')
    print(data.baseFreq/1e6)
    print(data.divider*data.baseFreq/1e6)
    print(data.divider)
    print(data.sm_size)
    print(data.fsm['/calibration/order/'][()])
    print(data.freq.shape)
    print(data.isFrequencySelection)
    print(data.isFramePermutation)

    print(data.framePermutation[:37])
    """