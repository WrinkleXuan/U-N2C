import math
import torch
import torch.nn as nn
import functools
from copy import deepcopy, copy

import torch.utils
import torch.utils.checkpoint
from .blocks import ConvolutionBlock, ResidualBlock
import torch.nn.functional as F
from einops import rearrange,repeat
import numpy as np
import copy
class Encoder(nn.Module):
    def __init__(self, input_ch, base_ch, num_down, num_residual, res_norm='instance', down_norm='instance'):
        super(Encoder, self).__init__()

        self.conv0 = ConvolutionBlock(
            in_channels=input_ch, out_channels=base_ch, kernel_size=7, stride=1,
            padding=3, pad='reflect', norm=down_norm, activ='relu')
        
        output_ch = base_ch
        for i in range(1, num_down+1):
            m = ConvolutionBlock(
                in_channels=output_ch, out_channels=output_ch * 2, kernel_size=4,
                stride=2, padding=1, pad='reflect', norm=down_norm, activ='relu')
            setattr(self, "conv{}".format(i), m)
            output_ch *= 2

        for i in range(num_residual):
            setattr(self, "res{}".format(i),
                ResidualBlock(output_ch, pad='reflect', norm=res_norm, activ='relu'))

        self.layers = [getattr(self, "conv{}".format(i)) for i in range(num_down+1)] + \
            [getattr(self, "res{}".format(i)) for i in range(num_residual)]
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, output_ch, base_ch, num_up, num_residual, res_norm='instance', up_norm='layer'):
        super(Decoder, self).__init__()
        input_ch = base_ch * 2 ** num_up
        input_chs = []

        for i in range(num_residual):
            setattr(self, "res{}".format(i),
                ResidualBlock(input_ch, pad='reflect', norm=res_norm, activ='lrelu'))
            input_chs.append(input_ch)

        for i in range(num_up):
            m = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                ConvolutionBlock(
                    in_channels=input_ch, out_channels=input_ch // 2, kernel_size=5,
                    stride=1, padding=2, pad='reflect', norm=up_norm, activ='lrelu'))
            setattr(self, "conv{}".format(i), m)
            input_chs.append(input_ch)
            input_ch //= 2

        m = ConvolutionBlock(
            in_channels=base_ch, out_channels=output_ch, kernel_size=7,
            stride=1, padding=3, pad='reflect', norm='none', activ='tanh')
        setattr(self, "conv{}".format(num_up), m)
        input_chs.append(base_ch)
        
        self.layers = [getattr(self, "res{}".format(i)) for i in range(num_residual)] + \
            [getattr(self, "conv{}".format(i)) for i in range(num_up + 1)]

        
    def forward(self, x):
        m= len(self.layers)
        
        for i in range(m):
            x = self.layers[i](x)

        
        return x

class LinearProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_,1,1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1] 
        return q,k,v

    def flops(self, q_L, kv_L=None): 
        kv_L = kv_L or q_L
        flops = q_L*self.dim*self.inner_dim+kv_L*self.dim*self.inner_dim*2
        return flops 

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        ##B,C,H,W=x.size()
        #x=x.reshape(B,-1)

        x = self.fc1(x)
        
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        #x=x.reshape(B,-1,H,W)
        return x
    
class Attention(nn.Module):
    def __init__(self, dim,num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
            
        #self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        
        #self.attn_drop = nn.Dropout(attn_drop)
        #self.proj = nn.Linear(dim, dim)
        #self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None):
        x = x.flatten(2).transpose(1, 2).contiguous()  # B H*W C
        
        B_, N, C = x.shape
        q=x 
        k=attn_kv 
        v=attn_kv 
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'

    def flops(self, q_num, kv_num):
        # calculate flops for 1 window with token length of N
        # print(N, self.dim)
        flops = 0
        # N = self.win_size[0]*self.win_size[1]
        # nW = H*W/N
        # qkv = self.qkv(x)
        # flops += N * self.dim * 3 * self.dim
        #flops += self.qkv.flops(q_num, kv_num)
        # attn = (q @ k.transpose(-2, -1))

        flops += self.num_heads * q_num * (self.dim // self.num_heads) * kv_num
        #  x = (attn @ v)
        flops += self.num_heads * q_num * (self.dim // self.num_heads) * kv_num
        
        # x = self.proj(x)
        flops += q_num * self.dim * self.dim
        print("MCA:{%.2f}"%(flops/1e9))
        return flops

class Memory_Block(nn.Module):        
    def __init__(self, k_hdim,v_hdim, nums_noise_mode, moving_average_rate=0.999):
        super().__init__()
        
        self.k_c = k_hdim
        self.v_c = v_hdim
        self.nums = nums_noise_mode
        
        self.moving_average_rate = moving_average_rate
        
        #self.noise_feature = nn.Parameter(torch.Tensor(nums_noise_mode, k_hdim),requires_grad=False) #k
        self.noise_feature = nn.Parameter(torch.Tensor(nums_noise_mode, k_hdim),requires_grad=False) #k
        
        #self.noise_feature=nn.Embedding(nums_noise_mode, k_hdim)
        self.register_parameter('noise_feature',self.noise_feature)
        
        self.std = nn.Parameter(torch.Tensor(nums_noise_mode, v_hdim),requires_grad=True) #v 
        self.register_parameter('std',self.std)
               
    def update(self, x, score, m=None):
        '''
            x: (n, c)
            e: (k, c)
            score: (n, k)
        '''
        if m is None:
            m = self.noise_feature.data
        x = x.detach()
        embed_ind = torch.max(score, dim=1)[1] # (n, )
        embed_onehot = F.one_hot(embed_ind, self.nums).type(x.dtype) # (n, k)        
        embed_onehot_sum = embed_onehot.sum(0)
        
        embed_sum = x.transpose(0, 1) @ embed_onehot # (c, k)
        embed_mean = embed_sum / (embed_onehot_sum + 1e-6)
        new_data = m * self.moving_average_rate + embed_mean.t() * (1 - self.moving_average_rate)
        if self.training:
            self.noise_feature.data = new_data
        return new_data
                
    def forward(self, x, update_flag=True):
        '''
          x: (b, c, h, w)
          embed: (k, c)
        '''
        
        b, c, h, w = x.size()        
        assert c == self.k_c
        nums, c = self.nums, self.k_c
        
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, c) # (n, c)
        
        m = self.noise_feature.data # (k, c)
                
        xn = F.normalize(x, dim=1) # (n, c)
        mn = F.normalize(m, dim=1) # (k, c)
        score = torch.matmul(xn, mn.t()) # (n, k)
        
        if update_flag and self.training:
            m = self.update(x, score, m)
            mn = F.normalize(m, dim=1) # (k, c)
            score = torch.matmul(xn, mn.t()) # (n, k)
            
        soft_label = F.softmax(score, dim=1)
        out = torch.matmul(soft_label, self.std) # (n, c)
        #out = out.view(b, h, w, self.v_c).permute(0, 3, 1, 2)
        out = out.view(b, h, w, self.v_c)
                                
        return out, score
        #return out
class Pattern_Memory_Block(nn.Module):        
    def __init__(self, qk_hdim,v_hdim, nums_pattern_mode, threshold=0.5,moving_average_rate=0.999,k=10):
        super().__init__()
        
        self.k_c = qk_hdim
        self.v_c = v_hdim
        self.nums = nums_pattern_mode
        self.k=k
        self.moving_average_rate = moving_average_rate
        self.threshold=threshold
        self.position_embed= AdaptivePositionEmbedding(max_length=37,embed_dim=qk_hdim)
        #self.frequency_embed=Mlp(in_features=1,out_features=qk_hdim)
        self.frequency_embed=nn.Linear(1,qk_hdim)
        self.parameter_feature = nn.Parameter(torch.Tensor(nums_pattern_mode, qk_hdim),requires_grad=False) #qk
        self.register_parameter('parameter_feature',self.parameter_feature)
        
        self.pattern = nn.Parameter(torch.Tensor(nums_pattern_mode, v_hdim),requires_grad=False) #v 
        self.register_parameter('pattern',self.pattern)
        self.T=nn.Parameter(torch.tensor(1.),requires_grad=True )
        
        #self.out_proj=Mlp(in_features=v_hdim*2,out_features=v_hdim)
        self.proj=Mlp(in_features=qk_hdim,out_features=qk_hdim)
        #self.proj=nn.Linear(in_features=qk_hdim,out_features=qk_hdim)
        
        self.age=torch.zeros(nums_pattern_mode)
        
        nn.init.zeros_(self.pattern.data)
        nn.init.zeros_(self.parameter_feature.data)
        #nn.init.trunc_normal_(self.pattern.data,std=0.02)
        
        nn.init.trunc_normal_(self.position_embed.pos_embed.data,std=0.02)
        
        self.apply(self._init_weights)
        
    def _init_weights(self,m):
        """
        ViT weight initialization
        :param m: module
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
    
    
    
    def reading(self,x,freq_embed,pos_embed):          
            

        
        q=self.proj(freq_embed+pos_embed)
        qn=F.normalize(q,dim=1)
        
        m,v=self.memory_update(x,q)
        mn=F.normalize(m,dim=1)
        score_para = torch.matmul(qn, mn.t())# (n, k)

        
        
        label=torch.max(score_para,dim=1)[1]
        label= F.one_hot(label, self.nums).type(score_para.dtype)  
        out = torch.matmul(label, v) # (n, c)
        return out
    
    def self_supervised_loss(self,x,freq,depths_pos=None):
        v=self.pattern.data
        vn=F.normalize(v,dim=1)
        m=self.parameter_feature.data
        mn = F.normalize(m, dim=1) # (k, c)
        
        # score from x
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, c) # (n, c)
        
        xn=F.normalize(x,dim=1)
        score=torch.matmul(xn,vn.t())

        label_e = torch.max(score, dim=1)[1] # (n, )
        freq_embed=self.frequency_embed(freq)
        
        freq_embed=repeat(freq_embed,'b c->(b n) c',n=h*w)
        
        pos_embed=self.position_embed((h,w),depths_pos)
        
        
        #for pos in pos_embed:
        #with torch.no_grad():
        q=self.proj(freq_embed+pos_embed)
        qn=F.normalize(q,dim=1)
        #label_x=torch.max(score, dim=1)[1]
        #logits_e=torch.matmul(qn,mn.t())*torch.exp(self.T)
        
        
        logits_e=torch.matmul(qn,mn.t())
        #logits_e=torch.matmul(qn,mn.t())/self.T
        
        #soft_label = F.softmax(logits_e, dim=1)
        
        #label=torch.max(logits_e,dim=1)[1]
        #label= F.one_hot(label, self.nums).type(soft_label.dtype) -soft_label.detach()+soft_label   
        label = F.gumbel_softmax(logits_e, tau=self.T, hard=True)

        out = torch.matmul(label, v) # (n, c)
        loss_identity= F.mse_loss(out,x)
        loss_contrastive=F.cross_entropy(logits_e/self.T,label_e)

        #logits_x=torch.matmul(xn,vn.t())*torch.exp(t)
        #loss_x=nn.CrossEntropyLoss()(logits_x,label_x)
        #loss+=(loss_e+loss_x)/2
        #loss+=loss_e
        #loss/=len(pos_embed)
    
        # score from embed
        return loss_identity,loss_contrastive
          
    def memory_update(self,x,q):
        
        b, c, h, w = x.size()        
        
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, c) # (n, c)
        xn=F.normalize(x,dim=1)
    
        qn=F.normalize(q,dim=1)
            
        #for pos in pos_embed:
        parameters = copy.deepcopy(self.parameter_feature.data) # (k, c)
        m = self.parameter_feature.data # (k, c)
        
        mn = F.normalize(m, dim=1) # (k, c)
    
        
        score_parameters=torch.matmul(qn,mn.t())
        top1_score,top1_index=torch.max(score_parameters, dim=1)
        pattern=copy.deepcopy(self.pattern.data)
        v=self.pattern.data[top1_index]
        vn=F.normalize(v,dim=1)

        score_pattern=torch.sum(xn*vn,dim=1)
        memory_mask= score_pattern > self.threshold
        self.age=self.age+1.0

        ### case1
        case_index=top1_index[memory_mask]
        embed_onehot = F.one_hot(case_index, self.nums).type(x.dtype) # (n, k)        
        embed_onehot_sum = embed_onehot.sum(0)
        embed_sum = q[memory_mask].transpose(0, 1) @ embed_onehot # (c, k)
        embed_mean = embed_sum[:,embed_onehot_sum!=0] / (embed_onehot_sum [embed_onehot_sum!=0])
        parameters[embed_onehot_sum!=0] = parameters[embed_onehot_sum!=0] * self.moving_average_rate + embed_mean.t() * (1 - self.moving_average_rate)
        if self.training:
            self.parameter_feature.data[embed_onehot_sum!=0]=parameters[embed_onehot_sum!=0]
            self.age[embed_onehot_sum!=0]=0.
        
        ### case 2
        
        memory_mask = torch.logical_not(memory_mask)

        case_index = top1_index[memory_mask] if len(top1_index[memory_mask])<self.nums else torch.topk(top1_score[memory_mask],k=self.k,dim=0,largest=False)[1]
        #print(len(case_index))
        #embed_onehot = F.one_hot(case_index, self.nums).type(x.dtype) # (n, k)        
        #embed_onehot_sum = embed_onehot.sum(0)

        #embed_sum = x[memory_mask].transpose(0, 1) @ embed_onehot # (c, k)        
        #embed_mean = embed_sum[:,embed_onehot_sum!=0] / (embed_onehot_sum [embed_onehot_sum!=0])
        #pattern[embed_onehot_sum!=0] = pattern[embed_onehot_sum!=0] *self.moving_average_rate+ embed_mean.t() * (1 - self.moving_average_rate)
        #pattern[embed_onehot_sum!=0] = embed_mean.t() 
        
        
        old_index=torch.topk(self.age,len(case_index),dim=0)[1]
        if len(top1_index[memory_mask])<self.nums: 
            parameters[old_index]=q[memory_mask]
            pattern[old_index]=x[memory_mask]
        else:
            parameters[old_index]=q[memory_mask][case_index]
            pattern[old_index]=x[memory_mask][case_index]
            
        if self.training:
            self.parameter_feature.data[old_index]=parameters[old_index]
            #self.pattern.data[embed_onehot_sum!=0]=pattern[embed_onehot_sum!=0]
            self.pattern.data[old_index]=pattern[old_index]
            self.age[old_index]=0.
            
        return parameters,pattern
        
            
        
            
    def _forward(self, x, freq, depths_pos, channel):
        '''
          x: (b, c, h, w)
          embed: (k, c)
        '''
        
        b, c, h, w = x.size()        
        assert c == self.v_c
        nums, c = self.nums, self.v_c

        
        #with torch.no_grad():
       
        freq_embed=self.frequency_embed(freq)
            
        freq_embed=repeat(freq_embed,'b c->(b n) c',n=h*w)
        pos_embed=self.position_embed((h,w),depths_pos)
        #channel=repeat(channel,'b c->(b n) c',n=h*w)
        
        
        pattern=self.reading(x,freq_embed,pos_embed)
        #out=self.out_proj(torch.cat((pattern,x),dim=-1))
        out = pattern.view(b, h, w, self.v_c).permute(0, 3, 1, 2)
        #x   = x.view(b, h, w, self.v_c).permute(0, 3, 1, 2)
        
        #return self.conv_block(torch.cat((out,x),dim=1))                        
        return out
    
    def forward(self,x, freq=None, depths_pos=None, channel=None,ues_checkpoint=False):
        if freq is None:
            freq=torch.zeros((x.size(0),1),device=x.device)
        if depths_pos is None:
            depths_pos=torch.zeros((x.size(0),1),device=x.device,dtype=torch.long)
        if channel is None:
            channel=torch.zeros((x.size(0),1),device=x.device,dtype=torch.long)    
        if ues_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward,x, freq, depths_pos, channel)
        else:
            return self._forward(x, freq, depths_pos, channel)
        

class AdaptivePositionEmbedding(nn.Module):
    def __init__(self,max_length,embed_dim) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(torch.Tensor(1,max_length, embed_dim))
        self.embed_dim=embed_dim
    def forward(self,size,z_direction_position):
        # the size of  (x y z)
        if size[0]<self.pos_embed.size(-2):
            rows_pos_embed=nn.AdaptiveAvgPool1d(size[0])(self.pos_embed.transpose(-2,-1)).transpose(-2,-1)
            rows_direction=repeat(rows_pos_embed,'b l d-> b (n l) d',n=size[0])
        else:
            rows_direction=repeat(self.pos_embed,'b l d-> b (n l) d',n=size[0])
        rows_direction=rows_direction.expand(z_direction_position.size(0),-1,-1).reshape(-1,self.embed_dim)
        if size[1]<self.pos_embed.size(-2):
        
            columns_pos_embed=nn.AdaptiveAvgPool1d(size[1])(self.pos_embed.transpose(-2,-1)).transpose(-2,-1)
            columns_direction=repeat(columns_pos_embed,'b l d-> b (l n) d',n=size[1])
        else:
            columns_direction=repeat(self.pos_embed,'b l d-> b (l n) d',n=size[1])
        columns_direction=columns_direction.expand(z_direction_position.size(0),-1,-1).reshape(-1,self.embed_dim)
        
        depths_direction=repeat(self.pos_embed,'b l d-> (n b) l d',n=z_direction_position.size(0))
        z_direction_position=z_direction_position.unsqueeze(2).expand(z_direction_position.size(0),z_direction_position.size(1),self.pos_embed.size(-1))
        
        depths_direction=torch.gather(depths_direction,1,z_direction_position)
        depths_direction=depths_direction.expand(-1,size[0]*size[1],-1).reshape(-1,self.embed_dim)
        
        return depths_direction+rows_direction+columns_direction  # b n c ,b n c, b n c 
        
        
class ConvNet(nn.Module):
    """
    Image with artifact is denoted as low quality image
    Image without artifact is denoted as high quality image
    """

    def __init__(self,input_ch=1, base_ch=64, num_down=2, num_residual=4,
        nums_noise_mode=10,
        res_norm='instance', down_norm='instance', up_norm='layer',moving_average_rate=0.999):
        super(ConvNet, self).__init__()
        
        
        self.encoder_clean = Encoder(input_ch, base_ch, num_down, num_residual, res_norm, down_norm)
        
        self.decoder_clean = Decoder(input_ch, base_ch, num_down, num_residual, res_norm, up_norm)
        
        self.encoder_noisy = Encoder(input_ch, base_ch, num_down, num_residual, res_norm, down_norm)
            #self.decoder_noisy = Decoder(input_ch, base_ch, num_down, num_residual, res_norm, up_norm)
        
        self.noise_memory=Memory_Block(nums_noise_mode=nums_noise_mode,k_hdim=base_ch*(2**num_down),v_hdim=(input_ch*(input_ch+1))//2,moving_average_rate=moving_average_rate)
        
        self.gated_block= nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(in_channels=base_ch*(2**num_down)*2,out_channels=base_ch*(2**num_down),kernel_size=1,stride=1,padding=0),
            nn.Sigmoid())
            #ConvolutionBlock(
            #in_channels=base_ch*(2**num_down)*2, out_channels=base_ch*(2**num_down), kernel_size=1, stride=1,
            #padding=0, pad='reflect', norm=down_norm)
            
        nn.init.zeros_(self.noise_memory.noise_feature.data)
        
        nn.init.ones_(self.noise_memory.std)

        self.apply(self._init_weights)
        
    def _init_weights(self,m):
        """
        ViT weight initialization
        :param m: module
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    
    def denoised_forward(self,x,freq,position,channel,content_memory,update_flag=True):
        x_0=self.encoder_noisy(x)
        content_memory.eval()
        with torch.no_grad():
            content=content_memory(x_0,freq,position,channel)
        score=self.gated_block(torch.cat((x_0,content),dim=1))
        content=score*content+(1-score)*x_0
        x_denoised=self.decoder_clean(content)

        A,_=self.noise_memory(x_0-content,update_flag=update_flag)
        B,H,W,C=A.shape
        N=math.floor((math.sqrt(C*2)))
        sampler=torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(N),torch.eye(N))
        samples=sampler.sample((x.size(0),x.size(-2),x.size(-1),)).to(device=A.device)
        
        index=torch.LongTensor([0,1,3]).to(device=A.device)
        
        tmp=torch.zeros((B,H,W,N**2),device=A.device)
        tmp=tmp.index_copy_(3,index,A)
        tmp=tmp.permute(0,3,1,2)
        
            
        tmp=F.interpolate(tmp,size=(x.size(-2),x.size(-1)),mode='nearest')
        A=rearrange(tmp,'b (n1 n2) h w->b h w n1 n2',n1=N)
        
        noise_std=A.transpose(3,4)@A
        
        noise_map=(A.transpose(3,4)@samples.unsqueeze(-1)).squeeze(-1).permute(0,3,1,2)
        
        return x_denoised,content,noise_map,noise_std

    def clean_decoder(self,x_0):
        return self.decoder_clean(x_0)
    def clean_forward(self,x):
        
        x_0=self.encoder_clean(x)
            
        x_reco=self.decoder_clean(x_0)
        return x_reco,x_0
    
    
if __name__=="__main__":
    import torch
    import torch.nn as nn

    # 创建一个 NLayerDiscriminator 实例
    input_nc = 2  # 输入图像的通道数
    ndf = 64  # 最后一层的过滤器数量
    n_layers = 2  # 网络的卷积层数量
    norm_layer = nn.InstanceNorm2d  # 使用的标准化层
    generator=ConvNet(input_ch=input_nc)

    # 创建一个输入图像
    input_image = torch.randn(128, input_nc, 32, 32,requires_grad=True)  # 例如，一个大小为 128x128 的输入图像
    y=torch.randn(128,input_nc,32,32,requires_grad=True)
    x_denoised,_,x_reco,y_reco,y_noise=generator.forward(input_image,y)
    # 前向传播

    
    # 输出和梯度计算示例
    print(x_denoised.size())
    print(x_reco.size())
    print(y_reco.size())
    print(y_noise.size())
