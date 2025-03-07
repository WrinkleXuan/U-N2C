# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import numpy as np
import torch.distributed as dist
from timm.data import Mixup

from .OpenMPIData import OpenMPIData
from .OS_FFL_MPIData import OSFFLMPIData





def build_loader_train(config):
    config.defrost()
    if config.DATA.DATASET == "OpenMPIData":
        dataset_train =build_dataset(dataset_name=config.DATA.DATASET,
                                     mode=config.MODE,
                                     data_path=config.DATA.DATA_PATH,
                                     data_name=config.DATA.DATA_NAME_TRAIN,
                                     aux_noisy_data_name=config.DATA.AUX_NOISY_DATA_NAME,
                                     aux_clean_data_name=config.DATA.AUX_CLEAN_DATA_NAME,
                                     add_noise=config.DATA.NOISE.ADD_NOISE,
                                     snr=config.DATA.NOISE.SNR,
                                     alpha=config.DATA.NOISE.ALPHA,
                                     noise_type=config.DATA.NOISE.NOISE_TYPE,
                                     output_size=(config.DATA.IMG_SIZE,config.DATA.IMG_SIZE)
                                     )
    elif config.DATA.DATASET=='OS_FFL_MPIData':
        
        dataset_train = build_dataset(dataset_name=config.DATA.DATASET,
                                     mode='train',
                                     data_path=config.DATA.DATA_PATH,
                                     data_name=config.DATA.DATA_NAME_TEST,
                                    output_size=(config.DATA.IMG_SIZE,config.DATA.IMG_SIZE)
                                     )
    
    else:
        raise NotImplementedError("We only support OpenMPIData and OS_FFL for training Now.")
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    if config.DATA.DATASET == "OpenMPIData":
        
        dataset_val = build_dataset(dataset_name=config.DATA.DATASET,
                                     mode='val',
                                     data_path=config.DATA.DATA_PATH,
                                     data_name=config.DATA.DATA_NAME_VAL,
                                     aux_clean_data_name=config.DATA.AUX_CLEAN_DATA_NAME,
                                     add_noise=config.DATA.NOISE.ADD_NOISE,
                                     snr=config.DATA.NOISE.SNR,
                                     alpha=config.DATA.NOISE.ALPHA,
                                     noise_type=config.DATA.NOISE.NOISE_TYPE,
                                      output_size=(config.DATA.IMG_SIZE,config.DATA.IMG_SIZE)
                                     )
    elif config.DATA.DATASET=='OS_FFL_MPIData':
        
        dataset_val = build_dataset(dataset_name=config.DATA.DATASET,
                                     mode='val',
                                     data_path=config.DATA.DATA_PATH,
                                     data_name=config.DATA.DATA_NAME_TEST,
                                    output_size=(config.DATA.IMG_SIZE,config.DATA.IMG_SIZE)
                                     )
    
    else:
        raise NotImplementedError("We only support OpenMPIData and OSFFLMPIData for validation Now.")
   
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)

    sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, shuffle=False)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    
    return dataset_train, dataset_val, data_loader_train, data_loader_val


def build_loader_test(config):
    config.defrost()
    if config.DATA.DATASET == "OpenMPIData":
        dataset_test = build_dataset(dataset_name=config.DATA.DATASET,
                                     mode=config.MODE,
                                     data_path=config.DATA.DATA_PATH,
                                     data_name=config.DATA.DATA_NAME_TEST,
                                     aux_clean_data_name=config.DATA.AUX_CLEAN_DATA_NAME,
                                     add_noise=config.DATA.NOISE.ADD_NOISE,
                                     snr=config.DATA.NOISE.SNR,
                                     alpha=config.DATA.NOISE.ALPHA,
                                     noise_type=config.DATA.NOISE.NOISE_TYPE,
                                    output_size=(config.DATA.IMG_SIZE,config.DATA.IMG_SIZE))
    elif config.DATA.DATASET=='OS_FFL_MPIData':
        
        dataset_test = build_dataset(dataset_name=config.DATA.DATASET,
                                     mode='val',
                                     data_path=config.DATA.DATA_PATH,
                                     data_name=config.DATA.DATA_NAME_TEST,
                                    output_size=(config.DATA.IMG_SIZE,config.DATA.IMG_SIZE)
                                     )
    
    else:
        raise NotImplementedError("We only support OpenMPIData or OS_FFL_MPIData for inference  Now.")
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    
    
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    
    return  dataset_test, data_loader_test

def build_dataset(dataset_name,mode, data_path,data_name,output_size,aux_noisy_data_name=None,aux_clean_data_name=None,add_noise=False,snr=1,alpha=1,noise_type='gaussion_noise'
                  ):
    if dataset_name == 'OpenMPIData':
        if mode=='train':
            is_train=True
            dataset = OpenMPIData(dir_path=data_path,
                                  data_name=data_name,
                                  random_rot_flip=is_train,
                                  aux_noisy_data_name=aux_noisy_data_name,
                                  aux_clean_data_name=aux_clean_data_name,
                                  add_noise=add_noise,
                                  snr=snr,
                                  alpha=alpha,
                                  noise_type=noise_type,
                                  output_size=output_size
                                  )
        elif mode=='val':
            is_train=False
            dataset = OpenMPIData(dir_path=data_path,
                                  data_name=data_name,
                                  mode='train',
                                  random_rot_flip=is_train,
                                  aux_clean_data_name=aux_clean_data_name,
                                  add_noise=add_noise,
                                  snr=snr,
                                  alpha=alpha,
                                  noise_type=noise_type,
                                  output_size=output_size
                                  )
        else:
            is_train=False
            dataset = OpenMPIData(dir_path=data_path,
                                  data_name=data_name,
                                  mode=mode,
                                  random_rot_flip=is_train,
                                  aux_clean_data_name=aux_clean_data_name,
                                  add_noise=add_noise,
                                  snr=snr,
                                  alpha=alpha,
                                  noise_type=noise_type,
                                  output_size=output_size)
    elif dataset_name=='OS_FFL_MPIData':
        dataset = OSFFLMPIData(
            dir_path=data_path,
            data_name=data_name,
            output_size=output_size
        )
    else:
        raise NotImplementedError("We only support OpenMPIData and OS_FFL_MPIData Now.")
    return dataset
