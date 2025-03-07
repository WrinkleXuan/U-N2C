# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
from pathlib import Path
import time
import json
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from config import get_config
from data import build_loader
from utils.logger import create_logger
from datetime import datetime,timedelta

from models import Trainer
from models import Tester,Tester_FFL


def parse_option():
    parser = argparse.ArgumentParser('SMNet training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    # easy config modification
    parser.add_argument('--batch_size', type=int, help="batch size for single GPU")
    parser.add_argument('--epoch', type=int, help="total epochs for training")
    
    parser.add_argument('--data_path', type=str, help='path to dataset')
    parser.add_argument('--data_name', type=str, help='name to file')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation_steps', type=int, help="gradient accumulation steps")
    
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--finetune', action='store_true', help='Network Finetune')
    parser.add_argument('--gpu', type=str, help='gpu_id',default='0,1,2,3')
    
    
    
    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

  
def main(config):
    
    if config.DATA.DATASET=='OpenMPIData':
        if config.MODE=='train':
            dataset, dataset_val, data_loader, data_loader_val = build_loader(config)
        else:
             dataset, data_loader = build_loader(config)
    elif config.DATA.DATASET=='OS_FFL_MPIData':
    
        if config.MODE=='train':
            dataset, dataset_val, data_loader, data_loader_val = build_loader(config)
        else:
             dataset, data_loader = build_loader(config)
    
    else:
        raise NotImplementedError("We only support OpenMPIData and OSFFLMPIData Now.")
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    start_time = time.time()
    if config.MODE=='train':
        model=Trainer(logger=logger,config=config,data_loader_train=data_loader,data_loader_val=data_loader_val)
    
        logger.info("Start training")
        
        for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):         
            model.forward_one_epoch(epoch)
    else:
        if config.DATA.DATASET=='OpenMPIData':
            model=Tester(logger=logger,config=config,data_loader=data_loader)
        elif config.DATA.DATASET=='OS_FFL_MPIData':
       
            model=Tester_FFL(logger=logger,config=config,data_loader=data_loader)
            #model.evaluate_image_reco()
        else:
            raise NotImplementedError("We only support OpenMPIData or OS_FFL_MPIData for inference  Now.")
        logger.info("Start Testing")
        if not config.THROUGHPUT_MODE:
            model.forward_one_epoch(config.TRAIN.START_EPOCH)
        else:
            model.forward_throughput()
  
    total_time = time.time() - start_time

    total_time_str = str(timedelta(seconds=int(total_time)))
    logger.info('Total {} time {}'.format('Training' if config.MODE=='train' else 'Testing',
                                            total_time_str))



if __name__ == '__main__':
    args, config = parse_option()

    #os.environ['CUDA_VISIBLE_DEVICES']='2,3,0,1'
    os.environ['CUDA_VISIBLE_DEVICES']=config.GPU_IDS
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        config.defrost()
        config.LOCAL_RANK=int(os.environ['LOCAL_RANK'])
        config.freeze()
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    now=datetime.now()
    now_formatted=now.strftime("%Y-%m-%d-%H-%M-%S")
    if not config.MODE=='test':
        config.OUTPUT=config.OUTPUT+now_formatted
    
        config.OUTPUT_TRAIN_LOG=os.path.join(config.OUTPUT,'train_log')
        config.OUTPUT_VAL_LOG=os.path.join(config.OUTPUT,'val_log')
        
        
    config.freeze()
    if not config.MODE=='test':
    
        os.makedirs(config.OUTPUT, exist_ok=True)
        Path(config.OUTPUT_TRAIN_LOG).mkdir(exist_ok=True)
        Path(config.OUTPUT_VAL_LOG).mkdir(exist_ok=True)
    
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(),mode=config.MODE, name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0 and config.MODE=='train':
        path = os.path.join(config.OUTPUT, "config.yaml")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)
