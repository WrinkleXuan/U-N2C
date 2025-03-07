import sys
sys.path.append('MPI/')
import os
from OpenMPIData import MPI
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import scipy.io as sio
import PIL.Image as Image
import glob
def reconstruction(sm:str | np.ndarray  = None,
                   system_matrix_name='6',
                   phantom_dir='/data/zwx/data/OpenMPIData/measurements',
                   reco_mode='3',
                   phantom_name: list | str =['resolutionPhantom','concentrationPhantom','shapePhantom'],
                   snr_threshold:tuple = (2,5),
                   reco_type='denoised',
                   add_noise=False,
                   output_dir='/data/zwx/SMNet',
                   draw_reolution_figure=False):
    if isinstance(sm,str):

        sm=np.load(sm)
    if isinstance(phantom_name,str):
            phantom_name=[phantom_name]
    for phantom in phantom_name:    
        data=MPI(measurements=phantom_dir+'/'+phantom+'/'+reco_mode+'.mdf',system_matrix="/data/zwx/data/OpenMPIData/calibrations/"+system_matrix_name+'.mdf')

        if reco_mode=='3':    
            if sm is not None:
                mpi_denoised=data.reco_3d(sm_denoised=sm,snr_threshold=snr_threshold,reco_name=reco_type,output_dir=output_dir)
            mpi_gt=data.reco_3d(snr_threshold=snr_threshold[1],add_noise=add_noise,output_dir=output_dir)
            mpi_noised=data.reco_3d(snr_threshold=snr_threshold[0],reco_name='noised',add_noise=add_noise,output_dir=output_dir)
        
        elif reco_mode=='2':
            if sm is not None:
                mpi_denoised=data.reco_2d(sm_denoised=sm,snr_threshold=snr_threshold,reco_name=reco_type,output_dir=output_dir)
            mpi_gt=data.reco_2d(snr_threshold=snr_threshold[1],add_noise=add_noise,output_dir=output_dir)
            mpi_noised=data.reco_2d(snr_threshold=snr_threshold[0],reco_name='noised',add_noise=add_noise,output_dir=output_dir)
        
        else:
            raise Exception("This reconstruction mode is not supported")
        if sm is not None:
            sio.savemat(output_dir+'/results/OpenMPI/3D/'+phantom+'.mat',{'gt':mpi_gt,'denoised':mpi_denoised,'noised':mpi_noised})        
        else:
            sio.savemat(output_dir+'/results/OpenMPI/3D/'+phantom+'.mat',{'gt':mpi_gt,'noised':mpi_noised})        
                

        if draw_reolution_figure:
            resolution_figure(mpi_noised,mpi_gt,mpi_denoised,output_dir=output_dir+'/results/OpenMPI/3D/{}'.format(phantom))
    
def resolution_figure(mpi_2,mpi_5,mpi_2_noise=None,output_dir='/data/zwx/SMNet'):

    #mpi_2_noise=reconstruction(sm='/data/zwx/SMNet/output/adnet_original_32/ADNet_real_denoise_refine/sm_denoised.npy',
    #                           snr_threshold=2,phantom_name='resolutionPhantom',reco_type='2_noise',add_noise=True)
    if mpi_2_noise is not None:
        mpi_2_noise=(mpi_2_noise-np.min(mpi_2_noise))/(np.max(mpi_2_noise)-np.min(mpi_2_noise))
        mip_mpi_2_noise=[np.max(mpi_2_noise,axis=i) for i in range(mpi_2_noise.ndim)]
        mip_mpi_2_noise=np.array(mip_mpi_2_noise)
        
    
    #mpi_2=reconstruction(sm=None,snr_threshold=2,phantom_name='resolutionPhantom',reco_type='2')
    mpi_2=(mpi_2-np.min(mpi_2))/(np.max(mpi_2)-np.min(mpi_2))
    
    mip_mpi_2=[np.max(mpi_2,axis=i) for i in range(mpi_2.ndim)]
    mip_mpi_2=np.array(mip_mpi_2)
    
    
    #mpi_5=reconstruction(sm=None,snr_threshold=5,phantom_name='resolutionPhantom',reco_type='5')
    mpi_5=(mpi_5-np.min(mpi_5))/(np.max(mpi_5)-np.min(mpi_5))
    
    mip_mpi_5=[np.max(mpi_5,axis=i) for i in range(mpi_5.ndim)]
    mip_mpi_5=np.array(mip_mpi_5)
    #color=['blue','red','orange']
    color=['orangered','lightsalmon','deepskyblue']

    x=np.array([i for i in range(37)])

    #for i in range(mpi_5.ndim):
    i=2
    font = {'family':'Times New Roman'  #'serif', 
        # ,'style':'italic'
        ,'weight':'bold'  # 'normal' 
         ,'size':25,
      }
   
    for j in range(mip_mpi_5[i].shape[1]):
        #mip_mpi_5=mip_mpi_5.transpose(0,2,1)
        #mip_mpi_2=mip_mpi_2.transpose(0,2,1)
        #mip_mpi_2_noise=mip_mpi_2_noise.transpose(0,2,1)
        
        mip_mpi_5[i][:,j]=(mip_mpi_5[i][:,j]-np.min(mip_mpi_5[i][:,j]))/(np.max(mip_mpi_5[i][:,j])-np.min(mip_mpi_5[i][:,j]))
        mip_mpi_2[i][:,j]=(mip_mpi_2[i][:,j]-np.min(mip_mpi_2[i][:,j]))/(np.max(mip_mpi_2[i][:,j])-np.min(mip_mpi_2[i][:,j]))
        mip_mpi_2_noise[i][:,j]=(mip_mpi_2_noise[i][:,j]-np.min(mip_mpi_2_noise[i][:,j]))/(np.max(mip_mpi_2_noise[i][:,j])-np.min(mip_mpi_2_noise[i][:,j]))
    #for image in zip(mip_mpi_2,mip_mpi_2_noise,mip_mpi_5):
        
        plt.figure(figsize=(12,12))

        plt.title('Resolution Phantom',fontsize=40,fontproperties=font)


        plt.ylabel("Normalized Intensity",fontsize=40,fontproperties=font)
        plt.xlabel("Distance[mm]",fontsize=40,fontproperties=font)
        plt.xticks(fontsize=30,fontproperties=font)
        plt.yticks(fontsize=30,fontproperties=font)
    
        m=make_interp_spline(x,mip_mpi_5[i][:,j])
        plt.plot(np.linspace(0,36,100),m(np.linspace(0,36,100)),color=color[0],linewidth=2,label='SNR >= 5')
        
        m=make_interp_spline(x,mip_mpi_2[i][:,j]) 
        plt.plot(np.linspace(0,36,100),m(np.linspace(0,36,100)),color=color[1],linewidth=2,label='SNR>=2')
        
        if mpi_2_noise is not None:
            m=make_interp_spline(x,mip_mpi_2_noise[i][:,j]) 
            plt.plot(np.linspace(0,36,100),m(np.linspace(0,36,100)),color=color[2],linewidth=2,label='Denoised\nSNR>=2')
        plt.legend(loc='best',prop=font)
        plt.savefig(output_dir+'/{}.png'.format(j),dpi=300,bbox_inches='tight', pad_inches=0.2)
        plt.close()

def resolution_Z(dir_path='/data/zwx/SMNet/results/reco_image',output_dir='/data/zwx/SMNet/results/'):
    image_list=sorted(glob.glob(os.path.join(dir_path,'*')))
    
    images=[]
    image_name=[]
    for img in image_list:
        #img=Image.open(img)
        image_name.append(img.split('/')[-1].split('.')[0])
        image=Image.open(img).convert('L')
        image=np.array(image)/255.0
        images.append(image)
    images=np.array(images)

    for j in range(images.shape[2]):
        color=['lightsalmon','orangered','yellowgreen','cyan','violet','chocolate','deepskyblue']
        x=np.array([i for i in range(images.shape[2])])
        font = {'family':'Times New Roman'  #'serif', 
        # ,'style':'italic'
        ,'weight':'bold'  # 'normal' 
        ,'size':25,
        }
     
        #images[:,:,j]=(images[:,:,j]-np.min(images[:,:,j]))/(np.max(images[:,:,j])-np.min(images[:,:,j]))

        for i in range(images.shape[0]):
            #print(i,j)
            images[i,:,j]=(images[i,:,j]-np.min(images[i,:,j]))/(np.max(images[i,:,j])-np.min(images[i,:,j]))

    
            m=make_interp_spline(x,images[i,:,j])
            plt.figure(figsize=(20,20))

            plt.title('Resolution Phantom',fontsize=40,fontproperties=font)


            plt.ylabel("Normalized Intensity",fontsize=40,fontproperties=font)
            plt.xlabel("Distance[mm]",fontsize=40,fontproperties=font)
            plt.xticks(fontsize=30,fontproperties=font)
            plt.yticks(fontsize=30,fontproperties=font)
            plt.plot(np.linspace(0,36,100),m(np.linspace(0,36,100)),color=color[i],linewidth=2,label=image_name[i])

            plt.legend(loc='best',prop=font)
            plt.savefig(output_dir+'/{}_{}.png'.format(image_name[i],j),dpi=300,bbox_inches='tight', pad_inches=0.2)
            plt.close()
def add_line(img_path='/data/zwx/SMNet/results/OpenMPI/3D/resolutionPhantom/reco_denoised/z_direction',imag_name='z.png',position=[16,22]):
    image=Image.open(img_path+'/'+imag_name)
    fix,ax=plt.subplots()
    ax.imshow(image)
    for p in position:
        ax.axvline(x=p,color='r',linestyle='--')
    ax.axis('off')
    plt.savefig(img_path+'/'+imag_name.split('.')[0]+'_withline.png', bbox_inches='tight', pad_inches=0)
def add_noise(img_path='/data/zwx/SMNet/results/OpenMPI/3D/resolutionPhantom/reco_denoised/z_direction',imag_name='z.png'):
    image=Image.open(img_path+'/'+imag_name).convert('L')
    image=np.array(image)/255.0
    print(image.shape)
    noise=np.random.randn(*image.shape)*0.1
    image=image+noise
    image=np.clip(image,0,1)
    #image = Image.fromarray(np.uint8(np.clip(image, 0, 255)))
    #image.save(img_path+'/'+imag_name.split('.')[0]+'_noise.png')
    plt.imsave(img_path+'/'+imag_name.split('.')[0]+'_noise.png',image,cmap='gray')
if __name__=="__main__":
    #resolution_Z()
    #reconstruction('/data/zwx/SMNet/output/adnet_original_32/ADNet_real_denoise_refine/sm_denoised.npy',phantom_name='resolutionPhantom',draw_reolution_figure=True)
    add_line('/data/zwx/SMNet/results/reco_image/',imag_name='5.png')
    add_line('/data/zwx/SMNet/results/reco_image/',imag_name='2.png')
    
    #add_line('/data/zwx/SMNet/results/OpenMPI/3D/resolutionPhantom/reco_noised/z_direction')
    #add_line('/data/zwx/SMNet/results/OpenMPI/3D/resolutionPhantom/reco_traget/z_direction')
    