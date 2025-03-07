# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import torch.nn as nn
from .Discriminator import NLayerDiscriminator
from .ConvNet import ConvNet
from .ConvNet import Pattern_Memory_Block
from .TransNet import TransNet
from .Discriminator_Transformer import Discriminator_former
def build_model(model_type,config):
    

    if model_type =='ConvNet':
        model =ConvNet(config.MODEL.ConvNet.INPUT_CH,
                     base_ch=config.MODEL.ConvNet.BASE_CH,
                     num_down=config.MODEL.ConvNet.NUM_DOWN,
                     num_residual=config.MODEL.ConvNet.NUM_RESIDUAL,
                     nums_noise_mode=config.MODEL.NUMS_NOSIE_MODE,
                     res_norm=config.MODEL.ConvNet.RES_NORM,
                     down_norm=config.MODEL.ConvNet.DOWN_NORM,
                     up_norm=config.MODEL.ConvNet.UP_NORM,
                     moving_average_rate=config.MODEL.ConvNet.MOVING_AVERAGE_RATE,
                     )
    elif model_type=='TransNet':
        model= TransNet(
                img_size=config.DATA.IMG_SIZE, 
                in_c=config.MODEL.TransNet.INPUT_CH,
                embed_dim=config.MODEL.TransNet.EMBED_DIM, 
                depths=config.MODEL.TransNet.DEPTHS,
                num_down=config.MODEL.TransNet.NUM_DOWN, 
                num_heads=config.MODEL.TransNet.NUM_HEADS, 
                mlp_ratio=config.MODEL.TransNet.MLP_RATIO, 
                qkv_bias=config.MODEL.TransNet.QKV_BIAS ,
                qk_scale=config.MODEL.TransNet.QK_SCALE, 
                drop_ratio=config.MODEL.DROP_RATE,
                attn_drop_ratio=config.MODEL.DROP_PATH_RATE, 
                drop_path_ratio=config.MODEL.DROP_PATH_RATE,
                norm_layer=config.MODEL.TransNet.NORM,
                act_layer=config.MODEL.TransNet.ACT,
                nums_noise_mode=config.MODEL.NUMS_NOSIE_MODE,
                moving_average_rate=config.MODEL.TransNet.MOVING_AVERAGE_RATE
        )
    elif model_type=='Discriminator':
        if config.MODEL.TYPE=='ConvNet': 
        
            model = NLayerDiscriminator(
                input_nc=config.MODEL.DISCRIMINIATOR.IN_CHANS,
                n_layers=config.MODEL.DISCRIMINIATOR.NLAYERS,
                ndf=config.MODEL.DISCRIMINIATOR.EMBED_DIM,
                norm_layer=config.MODEL.DISCRIMINIATOR.NORM_LAYER
            )
        elif config.MODEL.TYPE=='TransNet':
        
        #    model=Discriminator_former(
        #        image_size=config.DATA.IMG_SIZE,
        #        fmap_dim=config.MODEL.DISCRIMINIATOR.EMBED_DIM,
        #        init_channel=config.MODEL.DISCRIMINIATOR.IN_CHANS,
        #        fmap_max=config.MODEL.TransNet.EMBED_DIM*(2**config.MODEL.TransNet.NUM_DOWN),
        #    )
            model = NLayerDiscriminator(
                input_nc=config.MODEL.DISCRIMINIATOR.IN_CHANS,
                n_layers=config.MODEL.DISCRIMINIATOR.NLAYERS,
                ndf=config.MODEL.DISCRIMINIATOR.EMBED_DIM,
                norm_layer=config.MODEL.DISCRIMINIATOR.NORM_LAYER
            )
        
        else:
            raise NotImplementedError(f"Unkown model: {config.MODEL.TYPE}")
        
    elif model_type == 'PatternNet':
        if config.MODEL.TYPE=='ConvNet': 
            qk_hdim= config.MODEL.ConvNet.BASE_CH*(2**config.MODEL.ConvNet.NUM_DOWN)
            v_hdim= config.MODEL.ConvNet.BASE_CH*(2**config.MODEL.ConvNet.NUM_DOWN)
        elif config.MODEL.TYPE=='TransNet':
            #qk_hdim= config.MODEL.TransNet.EMBED_DIM*(2**len(config.MODEL.TransNet.DEPTHS))
            #v_hdim= config.MODEL.TransNet.EMBED_DIM*(2**len(config.MODEL.TransNet.DEPTHS))
            qk_hdim= config.MODEL.TransNet.EMBED_DIM*(2**config.MODEL.TransNet.NUM_DOWN)
            v_hdim= config.MODEL.TransNet.EMBED_DIM*(2**config.MODEL.TransNet.NUM_DOWN)
        
        else:
            raise NotImplementedError(f"Unkown model: {config.MODEL.TYPE}")
        model=Pattern_Memory_Block(nums_pattern_mode=config.MODEL.NUMS_PATTERN_MODE,
                                   qk_hdim=qk_hdim,
                                   v_hdim=v_hdim,
                                   moving_average_rate=config.MODEL.PATTERN_MEMORY.MOVING_AVERAGE_RATE,
                                   threshold=config.MODEL.PATTERN_MEMORY.THRESHOLD)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
