# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = "/data/zwx/data/OpenMPIData/SM_Denoising"
_C.DATA.DATA_NAME_TRAIN='calibrations_7.npz'
_C.DATA.DATA_NAME_TEST='calibrations_6.npz'
#_C.DATA.DATA_NAME_TEST='calibrations_5.npz'

_C.DATA.DATA_NAME_VAL='calibrations_5.npz'
_C.DATA.AUX_NOISY_DATA_NAME=["calibrations_8.npz","calibrations_9.npz"]
#_C.DATA.AUX_CLEAN_DATA_NAME=["calibrations_8.npz","calibrations_9.npz","calibrations_10.npz","calibrations_11.npz"]
_C.DATA.AUX_CLEAN_DATA_NAME=["calibrations_10.npz","calibrations_11.npz"]

#NOSIE
_C.DATA.NOISE=CN()
_C.DATA.NOISE.ADD_NOISE=True
_C.DATA.NOISE.NOISE_TYPE='gaussion_noise' # gaussion_noise  poission_gaussion_noise multivariate_gaussion_noise
_C.DATA.NOISE.SNR= 1.
_C.DATA.NOISE.ALPHA= 2.0

# Dataset name
_C.DATA.DATASET = 'OpenMPIData'

# Input image size
_C.DATA.IMG_SIZE = 32
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'ConvNet'
_C.MODEL.DTYPE='Discriminator'
_C.MODEL.MTYPE='PatternNet'

# Model name
_C.MODEL.NAME = 'smnet_tiny_32'
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Dropout rate
_C.MODEL.DROP_RATE = 0.1
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
_C.MODEL.NUMS_NOSIE_MODE=128
_C.MODEL.NUMS_PATTERN_MODE=128

# Swin Transformer parameters



#ConvNet paremeters
_C.MODEL.ConvNet = CN()
_C.MODEL.ConvNet.INPUT_CH = 2
_C.MODEL.ConvNet.BASE_CH = 64
_C.MODEL.ConvNet.NUM_DOWN = 2
_C.MODEL.ConvNet.NUM_RESIDUAL= 4
_C.MODEL.ConvNet.RES_NORM = 'instance'
_C.MODEL.ConvNet.DOWN_NORM = 'instance'
_C.MODEL.ConvNet.UP_NORM = 'layer'
_C.MODEL.ConvNet.MOVING_AVERAGE_RATE=0.999

#TransNet paremeters
_C.MODEL.TransNet = CN()
_C.MODEL.TransNet.INPUT_CH = 2
_C.MODEL.TransNet.EMBED_DIM = 96
_C.MODEL.TransNet.DEPTHS = [2, 2, 6, 2]
_C.MODEL.TransNet.NUM_DOWN = 2
_C.MODEL.TransNet.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.TransNet.MLP_RATIO =4.
_C.MODEL.TransNet.QKV_BIAS = True
_C.MODEL.TransNet.QK_SCALE = None
_C.MODEL.TransNet.MOVING_AVERAGE_RATE=0.999
_C.MODEL.TransNet.NORM='layer_norm'
_C.MODEL.TransNet.ACT='gelu'


_C.MODEL.PATTERN_MEMORY=CN()
_C.MODEL.PATTERN_MEMORY.THRESHOLD = 0.5
_C.MODEL.PATTERN_MEMORY.MOVING_AVERAGE_RATE=0.999
_C.MODEL.PATTERN_MEMORY.K=10



# NlayerDiscriminator parameters
_C.MODEL.DISCRIMINIATOR=CN()
_C.MODEL.DISCRIMINIATOR.IN_CHANS=2
_C.MODEL.DISCRIMINIATOR.EMBED_DIM=96
_C.MODEL.DISCRIMINIATOR.NLAYERS=2
_C.MODEL.DISCRIMINIATOR.NORM_LAYER='instance'
# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 2
_C.TRAIN.WEIGHT_DECAY = 0.05

_C.TRAIN.BASE_LR = 5e-4
#_C.TRAIN.BASE_LR = 5e-3

_C.TRAIN.WARMUP_LR = 5e-7
#_C.TRAIN.WARMUP_LR = 5e-5

_C.TRAIN.MIN_LR = 5e-6
#_C.TRAIN.MIN_LR = 5e-6

# Clip gradient norm
#_C.TRAIN.CLIP_GRAD = 5.0
_C.TRAIN.CLIP_GRAD = None

# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = False
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1


# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
#_C.TRAIN.LR_SCHEDULER.NAME = 'step'

# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# warmup_prefix used in CosineLRScheduler
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True

#_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = False

# [SimMIM] Gamma / Multi steps value, used in MultiStepLRScheduler
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

_C.TRAIN.LOSS = CN()
_C.TRAIN.LOSS.HH=['l1',1.0]

_C.TRAIN.LOSS.LH=['l1',1.0]

_C.TRAIN.LOSS.LL=['l1',0.0]

_C.TRAIN.LOSS.HL=['l1',1.0]

_C.TRAIN.LOSS.HLH=['l1',1.0]
_C.TRAIN.LOSS.LHL=['perceptual',1.0]

_C.TRAIN.LOSS.IDENTITY=['l2',1.0]
_C.TRAIN.LOSS.CC=['l2',1.0]

_C.TRAIN.LOSS.NE=['nll',1.0]

_C.TRAIN.LOSS.CONTRASTIVE=['contrastive',1.0]


_C.TRAIN.LOSS.GAN_TYPE='lsgan'
_C.TRAIN.LOSS.DH=['dh',1.0]
_C.TRAIN.LOSS.DL=['dl',1.0]


# -----------------------------------------------------------------------------
# Visual settings
# -----------------------------------------------------------------------------
_C.VISUAL=CN()
_C.VISUAL.FREQ=50
_C.VISUAL.NAME='visual'
_C.VISUAL.COMPUTATION_GRAPH=False


# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------

# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''

_C.OUTPUT_TRAIN_LOG = ''
_C.OUTPUT_VAL_LOG = ''

# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint (peochs)
_C.SAVE_FREQ = 10
# Frequency to logging info (epochs)
_C.PRINT_FREQ = 1
_C.VAL_FREQ = 5
_C.METRICS=['psnr',
            'nrmse',
            'sbrc',
            'nrmse_reco',
            'psnr_reco',
            'ssim']

# Fixed random seed
_C.SEED = 2024
# Perform Mode (train or test), overwritten by command line argument
_C.MODE ='train'
_C.FINETUNE=False


# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0
_C.GPU_IDS = '0,1,2,3'



def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    # merge from specific arguments
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size

    if _check_args('epoch'):
        config.TRAIN.EPOCHS = args.epoch
    
    if _check_args('finetune'):
        config.FINETUNE=args.finetune
        for k,v in config.TRAIN.LOSS.items():
            if k in ['HLH']:
            #if k in ['HH','Identity']:
            
                v[1]=1.0
            elif k!='GAN_TYPE':
                v[1]=0.
    if _check_args('eval'):
        config.MODE = 'test'
    if _check_args('data_path'):
        config.DATA.DATA_PATH = args.data_path
        if _check_args('data_name'):
            if config.MODE=='train':
                config.DATA.DATA_NAME_TRAIN=args.data_name
            else:
                config.DATA.DATA_NAME_TEST=args.data_name
        print(config.DATA.DATA_PATH+'/'+config.DATA.DATA_NAME)
    
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
    if _check_args('accumulation_steps'):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if config.MODE=='train' and _check_args('output'):
        config.OUTPUT = args.output
    if _check_args('tag'):
        config.TAG = args.tag
    if _check_args('throughput'):
        config.THROUGHPUT_MODE = True
    if _check_args('gpu'):
        config.GPU_IDS = args.gpu

    ## Overwrite optimizer if not None, currently we use it for [fused_adam, fused_lamb]
    if _check_args('optim'):
        config.TRAIN.OPTIMIZER.NAME = args.optim

    # set local rank for distributed training
    #config.LOCAL_RANK = args.local_rank

    # output folder
    if config.MODE=='train':
        config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
