import math
import torch
import torch.nn as nn

import torch.utils
import torch.utils.checkpoint
from .blocks import activ_dict,norm_dict
import torch.nn.functional as F
from einops import rearrange
from .blocks import ResidualBlock

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor



class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class Input_Projection(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1,padding=1, norm_layer=None,act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,),
            act_layer(inplace=True) 
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*W*self.in_channel*self.out_channel*3*3

        if self.norm is not None:
            flops += H*W*self.out_channel 
        print("Input_proj:{%.2f}"%(flops/1e9))
        return flops
    
class Output_Projection(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None,act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size//2),
        )
        if act_layer is not None:
            self.proj.add_module('activation',act_layer(inplace=True) if act_layer!=nn.Tanh else act_layer())
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*W*self.in_channel*self.out_channel*3*3

        if self.norm is not None:
            flops += H*W*self.out_channel 
        print("Output_proj:{%.2f}"%(flops/1e9))
        return flops





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
    

class Attention(nn.Module):
    def __init__(self, dim,num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
            
        self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        
        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x,attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))


        if mask is not None:
            nW = mask.shape[0]
            # mask = repeat(mask, 'nW m n -> nW m (n d)',d = ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
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
        flops += self.qkv.flops(q_num, kv_num)
        # attn = (q @ k.transpose(-2, -1))

        flops += self.num_heads * q_num * (self.dim // self.num_heads) * kv_num
        #  x = (attn @ v)
        flops += self.num_heads * q_num * (self.dim // self.num_heads) * kv_num
        
        # x = self.proj(x)
        flops += q_num * self.dim * self.dim
        print("MCA:{%.2f}"%(flops/1e9))
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
    
class TransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 ):
        super(TransformerBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop_ratio, proj_drop=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
        #self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,  drop=drop_ratio)

    def forward(self, x):
        #x=self.norm1(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

class BaisicSMNetLayer(nn.Module):
    def __init__(self,dim,depth,num_heads,mlp_ratio,qk_scale,qkv_bias,drop_ratio,drop_path_ratio,
                 attn_drop_ratio,norm_layer,act_layer,residual=False):
        super().__init__()
        self.blocks=nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qk_scale=qk_scale,
                qkv_bias=qkv_bias,
                drop_ratio=drop_ratio,
                drop_path_ratio=drop_path_ratio[i],
                attn_drop_ratio=attn_drop_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
                )
            for i in range(depth)])
    def forward(self,x):
        for blk in self.blocks:
            x=blk(x)
        return x
    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops

class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1,2).contiguous()  # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H/2*W/2*self.in_channel*self.out_channel*4*4
        print("Downsample:{%.2f}"%(flops/1e9))
        return flops

# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
            #nn.Upsample(scale_factor=2, mode="nearest"),
            #nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        )
        self.in_channel = in_channel
        self.out_channel = out_channel
        
    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1,2).contiguous() # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*2*W*2*self.in_channel*self.out_channel*2*2 
        print("Upsample:{%.2f}"%(flops/1e9))
        return flops
    
class Encoder(nn.Module):
    def __init__(self,
                img_size,
                in_c,
                embed_dim,
                depths,
                num_down,
                num_heads,mlp_ratio,qk_scale,qkv_bias,drop_ratio,
                attn_drop_ratio,
                act_layer,
                norm_layer,
                dpr
                ):
        super(Encoder, self).__init__()
        
        #self.norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        #self.act_layer = act_layer or nn.GELU
        self.num_layers = len(depths)
        self.num_down=num_down
        #dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, sum(depths))]  # stochastic depth decay rule
       
        self.input_proj = Input_Projection(in_channel=in_c, out_channel=embed_dim, kernel_size=3, stride=1, padding=1,act_layer=nn.LeakyReLU)
        
        self.pos_embed = nn.Parameter(torch.Tensor(1, img_size*img_size, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        self.encoder_layers=nn.ModuleList()
        for i in range(0,self.num_layers):
            if i<self.num_down:
                layer=BaisicSMNetLayer(dim=embed_dim*2**i,
                            depth=depths[i],
                            num_heads=num_heads[i],
                            mlp_ratio=mlp_ratio,
                            qk_scale=qk_scale,
                            qkv_bias=qkv_bias,
                            drop_ratio=drop_ratio,
                            drop_path_ratio= dpr[sum(depths[:i]):sum(depths[:i+1])],
                            attn_drop_ratio=attn_drop_ratio,
                            norm_layer=norm_layer,
                            act_layer=act_layer
                            )
                self.encoder_layers.append(layer)
            
                dowsample=Downsample(embed_dim*2**i,embed_dim*2**(i+1))
                self.encoder_layers.append(dowsample)
            else:
                layer=BaisicSMNetLayer(dim=embed_dim*2**num_down,
                            depth=depths[i],
                            num_heads=num_heads[i],
                            mlp_ratio=mlp_ratio,
                            qk_scale=qk_scale,
                            qkv_bias=qkv_bias,
                            drop_ratio=drop_ratio,
                            drop_path_ratio= dpr[sum(depths[:i]):sum(depths[:i+1])],
                            attn_drop_ratio=attn_drop_ratio,
                            norm_layer=norm_layer,
                            act_layer=act_layer,
                            )
                self.encoder_layers.append(layer)
        self.bottleneck_proj = Output_Projection(in_channel=embed_dim*2**num_down, out_channel=embed_dim*2**num_down, kernel_size=3, stride=1,act_layer=nn.LeakyReLU,norm_layer=nn.InstanceNorm2d)
        
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x=self.pos_drop(self.input_proj(x)+self.pos_embed)
        #x=self.input_proj(x)
        
        for layer in self.encoder_layers:
            x = layer(x)
        
        x=self.bottleneck_proj(x)
        #B, L, C = x.shape
        #H = int(math.sqrt(L))
        #W = int(math.sqrt(L))
        #x = x.transpose(1, 2).view(B, C, H, W)
        
        return x


class Decoder(nn.Module):
    def __init__(self,
                in_c,
                embed_dim,
                depths,
                num_down,
                num_heads,mlp_ratio,qk_scale,qkv_bias,drop_ratio,
                attn_drop_ratio,
                act_layer,
                norm_layer,
                dpr):
        super(Decoder, self).__init__()
        #dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, sum(depths))]  # stochastic depth decay rule
        #self.norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        #self.act_layer = act_layer or nn.GELU
        self.num_layers = len(depths)
        self.num_down=num_down
        self.decoder_layers=nn.ModuleList()
        self.bottleneck_proj = Input_Projection(in_channel=embed_dim*2**num_down, out_channel=embed_dim*2**num_down, kernel_size=3, stride=1, padding=1,act_layer=nn.LeakyReLU,norm_layer=nn.InstanceNorm2d)
        
        for i in range(self.num_layers,0,-1):
            if i>self.num_down:
                layer=BaisicSMNetLayer(dim=embed_dim*2**num_down,
                            depth=depths[i-1],
                            num_heads=num_heads[i-1],
                            mlp_ratio=mlp_ratio,
                            qk_scale=qk_scale,
                            qkv_bias=qkv_bias,
                            drop_ratio=drop_ratio,
                            drop_path_ratio= dpr[sum(depths[:i-1]):sum(depths[:i])],
                            attn_drop_ratio=attn_drop_ratio,
                            norm_layer=norm_layer,
                            act_layer=act_layer,
                            )
                self.decoder_layers.append(layer)
            else:
                upsample=Upsample(embed_dim*2**i,embed_dim*2**(i-1))
                self.decoder_layers.append(upsample)

                layer=BaisicSMNetLayer(dim=embed_dim*2**(i-1),
                            depth=depths[i-1],
                            num_heads=num_heads[i-1],
                            mlp_ratio=mlp_ratio,
                            qk_scale=qk_scale,
                            qkv_bias=qkv_bias,
                            drop_ratio=drop_ratio,
                            drop_path_ratio= dpr[sum(depths[:i-1]):sum(depths[:i])],
                            attn_drop_ratio=attn_drop_ratio,
                            norm_layer=norm_layer,
                            act_layer=act_layer
                            )
                self.decoder_layers.append(layer)
                
            
        self.output_proj = Output_Projection(in_channel=embed_dim, out_channel=in_c, kernel_size=3, stride=1,act_layer=nn.Tanh)

    def forward(self, x):
        x=self.bottleneck_proj(x)
        
        #x = x.flatten(2).transpose(1, 2).contiguous()  # B H*W C
        
        for layer in self.decoder_layers:
            x = layer(x)
        x=self.output_proj(x)
        return x

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
    

        
class TransNet(nn.Module):
    """
    Image with artifact is denoted as low quality image
    Image without artifact is denoted as high quality image
    """

    def __init__(self,
                 img_size=32, 
                 in_c=1,
                embed_dim=96, 
                depths=[2, 3, 2],
                num_down=2, 
                num_heads=[3, 6, 12], 
                mlp_ratio=4.0, 
                qkv_bias=True,
                qk_scale=None, 
                drop_ratio=0.,
                attn_drop_ratio=0., 
                drop_path_ratio=0.,
                norm_layer='layer',
                act_layer='gelu',
                nums_noise_mode=10,
                moving_average_rate=0.999):
        super(TransNet, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, sum(depths))]  # stochastic depth decay rule
        norm_layer = norm_dict[norm_layer]
        act_layer = activ_dict[act_layer]
        #norm_layer or partial(nn.LayerNorm, eps=1e-6)
        #act_layer = act_layer or nn.GELU
        
        self.encoder_clean = Encoder(img_size=img_size,
                                     in_c=in_c, 
                                     embed_dim=embed_dim, 
                                     depths=depths,
                                     num_down=num_down,
                                     num_heads=num_heads,
                                     mlp_ratio=mlp_ratio,
                                     qk_scale=qk_scale,
                                     qkv_bias=qkv_bias,
                                     drop_ratio=drop_ratio,
                                     attn_drop_ratio=attn_drop_ratio,
                                     act_layer=act_layer,
                                     norm_layer=norm_layer,
                                     dpr=dpr)
        
        self.decoder_clean = Decoder(in_c=in_c, 
                                     embed_dim=embed_dim, 
                                     depths=depths,
                                     num_down=num_down,
                                     num_heads=num_heads,
                                     mlp_ratio=mlp_ratio,
                                     qk_scale=qk_scale,
                                     qkv_bias=qkv_bias,
                                     drop_ratio=drop_ratio,
                                     attn_drop_ratio=attn_drop_ratio,
                                     act_layer=act_layer,
                                     norm_layer=norm_layer,
                                     dpr=dpr)
        
        self.encoder_noisy = Encoder(img_size=img_size,
                                     in_c=in_c, 
                                     embed_dim=embed_dim, 
                                     depths=depths,
                                     num_down=num_down,
                                     num_heads=num_heads,
                                     mlp_ratio=mlp_ratio,
                                     qk_scale=qk_scale,
                                     qkv_bias=qkv_bias,
                                     drop_ratio=drop_ratio,
                                     attn_drop_ratio=attn_drop_ratio,
                                     act_layer=act_layer,
                                     norm_layer=norm_layer,
                                     dpr=dpr)
            #self.decoder_noisy = Decoder(input_ch, base_ch, num_down, num_residual, res_norm, up_norm)
        
        #self.noise_memory=Memory_Block(nums_noise_mode=nums_noise_mode,k_hdim=embed_dim*(2**len(depths)),v_hdim=(in_c*(in_c+1))//2,moving_average_rate=moving_average_rate)
        self.noise_memory=Memory_Block(nums_noise_mode=nums_noise_mode,k_hdim=embed_dim*(2**num_down),v_hdim=(in_c*(in_c+1))//2,moving_average_rate=moving_average_rate)
        
        self.gated_block= nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(in_channels=embed_dim*(2**num_down)*2,out_channels=embed_dim*(2**num_down),kernel_size=1,stride=1,padding=0),
            nn.Sigmoid())
            
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
        #B,L,C=x_0.shape
        #H = int(math.sqrt(L))
        #W = int(math.sqrt(L))
        #x_0 = x_0.transpose(1, 2).contiguous().view(B, C, H, W)
        
        content_memory.eval()
        with torch.no_grad():
            content=content_memory(x_0,freq,position,channel)
        
        score=self.gated_block(torch.cat((x_0,content),dim=1))
        content=score*content+(1-score)*x_0
        
        #content=x_0
        x_denoised=self.decoder_clean(content)

        A,_=self.noise_memory(x_0-content,update_flag=update_flag)
        #A,_=self.noise_memory(content,update_flag=update_flag)
        
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
        #B,L,C=x_0.shape
        #H = int(math.sqrt(L))
        #W = int(math.sqrt(L))
        x_reco=self.decoder_clean(x_0)
        #return x_reco,x_0.transpose(1, 2).contiguous().view(B, C, H, W)
        return x_reco,x_0

    
if __name__=="__main__":
    import torch
    import torch.nn as nn

    # 创建一个 NLayerDiscriminator 实例
    input_nc = 2  # 输入图像的通道数
    ndf = 64  # 最后一层的过滤器数量
    n_layers = 2  # 网络的卷积层数量
    norm_layer = nn.InstanceNorm2d  # 使用的标准化层
    generator=TransNet(input_ch=input_nc)

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
