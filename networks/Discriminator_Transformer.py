from math import log2,sqrt,log,pi
from functools import partial
import torch
from torch import nn, einsum
from einops import rearrange, reduce,repeat
from einops.layers.torch import Rearrange


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def leaky_relu(p = 0.1):
    return nn.LeakyReLU(p)

def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

def get_sin_cos(seq):
    n = seq.shape[0]
    x_sinu = repeat(seq, 'i d -> i j d', j = n)
    y_sinu = repeat(seq, 'j d -> i j d', i = n)

    sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim = -1)
    cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim = -1)

    sin, cos = map(lambda t: rearrange(t, 'i j d -> (i j) d'), (sin, cos))
    sin, cos = map(lambda t: repeat(t, 'n d -> () () n (d j)', j = 2), (sin, cos))
    return sin, cos

def is_power_of_two(val):
    return log2(val).is_integer()


def upsample(scale_factor = 2):
    return nn.Upsample(scale_factor = scale_factor)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)

        if isinstance(out, tuple):
            out, latent = out
            ret = (out + x, latent)
            return ret

        return x + out

class SumBranches(nn.Module):
    def __init__(self, branches):
        super().__init__()
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
        return sum(map(lambda fn: fn(x), self.branches))

# attention and transformer modules

class ChanNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn, dim_context = None):
        super().__init__()
        self.norm = ChanNorm(dim)
        self.norm_context = ChanNorm(dim_context) if exists(dim_context) else None
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs.pop('context')
            context = self.norm_context(context)
            kwargs.update(context = context)

        return self.fn(x, **kwargs)
    
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, downsample_keys = 1):
        super().__init__()
        self.dim = dim
        self.downsample_keys = downsample_keys

    def forward(self, q, k):
        device, dtype, n = q.device, q.dtype, int(sqrt(q.shape[-2]))

        seq = torch.linspace(-1., 1., steps = n, device = device)
        seq = seq.unsqueeze(-1)

        scales = torch.logspace(0., log(10 / 2) / log(2), self.dim // 4, base = 2, device = device, dtype = dtype)
        scales = scales[(*((None,) * (len(seq.shape) - 1)), Ellipsis)]

        seq = seq * scales * pi

        x = seq
        y = seq

        y = reduce(y, '(j n) c -> j c', 'mean', n = self.downsample_keys)

        q_sin, q_cos = get_sin_cos(x)
        k_sin, k_cos = get_sin_cos(y)
        q = (q * q_cos) + (rotate_every_two(q) * q_sin)
        k = (k * k_cos) + (rotate_every_two(k) * k_sin)
        return q,k
     
class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding = 0, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

def FeedForward(dim, mult = 4, kernel_size = 3, bn = False):
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(dim, dim * mult * 2, 1),
        nn.GLU(dim = 1),
        nn.BatchNorm2d(dim * mult) if bn else nn.Identity(),
        DepthWiseConv2d(dim * mult, dim * mult * 2, kernel_size, padding = padding),
        nn.GLU(dim = 1),
        nn.Conv2d(dim * mult, dim, 1)
    )

# sinusoidal embedding

class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        dim //= 2
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        h = torch.linspace(-1., 1., x.shape[-2], device = x.device).type_as(self.inv_freq)
        w = torch.linspace(-1., 1., x.shape[-1], device = x.device).type_as(self.inv_freq)
        sinu_inp_h = torch.einsum('i , j -> i j', h, self.inv_freq)
        sinu_inp_w = torch.einsum('i , j -> i j', w, self.inv_freq)
        sinu_inp_h = repeat(sinu_inp_h, 'h c -> () c h w', w = x.shape[-1])
        sinu_inp_w = repeat(sinu_inp_w, 'w c -> () c h w', h = x.shape[-2])
        sinu_inp = torch.cat((sinu_inp_w, sinu_inp_h), dim = 1)
        emb = torch.cat((sinu_inp.sin(), sinu_inp.cos()), dim = 1)
        return emb

# classes

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        fmap_size = None,
        dim_out = None,
        kv_dim = None,
        heads = 8,
        dim_head = 64,
        q_kernel_size = 1,
        kv_kernel_size = 3,
        out_kernel_size = 1,
        q_stride = 1,
        include_self = False,
        downsample = False,
        downsample_kv = 1,
        bn = False,
        latent_dim = None
    ):
        super().__init__()
        self.sinu_emb = FixedPositionalEmbedding(dim)

        inner_dim = dim_head *  heads
        kv_dim = default(kv_dim, dim)
        dim_out = default(dim_out, dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        q_padding = q_kernel_size // 2
        kv_padding = kv_kernel_size // 2
        out_padding = out_kernel_size // 2

        q_conv_params = (1, 1, 0)

        self.to_q = nn.Conv2d(dim, inner_dim, *q_conv_params, bias = False)

        if downsample_kv == 1:
            kv_conv_params = (3, 1, 1)
        elif downsample_kv == 2:
            kv_conv_params = (3, 2, 1)
        elif downsample_kv == 4:
            kv_conv_params = (7, 4, 3)
        else:
            raise ValueError(f'invalid downsample factor for key / values {downsample_kv}')

        self.to_k = nn.Conv2d(kv_dim, inner_dim, *kv_conv_params, bias = False)
        self.to_v = nn.Conv2d(kv_dim, inner_dim, *kv_conv_params, bias = False)

        self.bn = bn
        if self.bn:
            self.q_bn = nn.BatchNorm2d(inner_dim) if bn else nn.Identity()
            self.k_bn = nn.BatchNorm2d(inner_dim) if bn else nn.Identity()
            self.v_bn = nn.BatchNorm2d(inner_dim) if bn else nn.Identity()

        self.has_latents = exists(latent_dim)
        if self.has_latents:
            self.latent_norm = ChanNorm(latent_dim)
            self.latents_to_qkv = nn.Conv2d(latent_dim, inner_dim * 3, 1, bias = False)

            self.latents_to_out = nn.Sequential(
                nn.Conv2d(inner_dim, latent_dim * 2, 1),
                nn.GLU(dim = 1),
                nn.BatchNorm2d(latent_dim) if bn else nn.Identity()
            )

        self.include_self = include_self
        if include_self:
            self.to_self_k = nn.Conv2d(dim, inner_dim, *kv_conv_params, bias = False)
            self.to_self_v = nn.Conv2d(dim, inner_dim, *kv_conv_params, bias = False)

        self.mix_heads_post = nn.Parameter(torch.randn(heads, heads))

        out_conv_params = (3, 2, 1) if downsample else q_conv_params

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim_out * 2, *out_conv_params),
            nn.GLU(dim = 1),
            nn.BatchNorm2d(dim_out) if bn else nn.Identity()
        )

        self.fmap_size = fmap_size
        self.pos_emb = RotaryEmbedding(dim_head, downsample_keys = downsample_kv)

    def forward(self, x, latents = None, context = None, include_self = False):
        assert not exists(self.fmap_size) or x.shape[-1] == self.fmap_size, 'fmap size must equal the given shape'

        b, n, _, y, h, device = *x.shape, self.heads, x.device

        has_context = exists(context)
        context = default(context, x)

        q_inp = x
        k_inp = context
        v_inp = context

        if not has_context:
            sinu_emb = self.sinu_emb(context)
            q_inp += sinu_emb
            k_inp += sinu_emb

        q, k, v = (self.to_q(q_inp), self.to_k(k_inp), self.to_v(v_inp))

        if self.bn:
            q = self.q_bn(q)
            k = self.k_bn(k)
            v = self.v_bn(v)

        out_h, out_w = q.shape[-2:]

        split_head = lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = h)

        q, k, v = map(split_head, (q, k, v))

        if not has_context:
            q, k = self.pos_emb(q, k)

        if self.include_self:
            kx = self.to_self_k(x)
            vx = self.to_self_v(x)
            kx, vx = map(split_head, (kx, vx))

            k = torch.cat((kx, k), dim = -2)
            v = torch.cat((vx, v), dim = -2)

        if self.has_latents:
            assert exists(latents), 'latents must be passed in'
            latents = self.latent_norm(latents)
            lq, lk, lv = self.latents_to_qkv(latents).chunk(3, dim = 1)
            lq, lk, lv = map(split_head, (lq, lk, lv))

            latent_shape = lq.shape
            num_latents = lq.shape[-2]

            q = torch.cat((lq, q), dim = -2)
            k = torch.cat((lk, k), dim = -2)
            v = torch.cat((lv, v), dim = -2)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim = -1)
        attn = einsum('b h i j, h g -> b g i j', attn, self.mix_heads_post)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

        if self.has_latents:
            lout, out = out[..., :num_latents, :], out[..., num_latents:, :]
            lout = rearrange(lout, 'b h (x y) d -> b (h d) x y', h = h, x = latents.shape[-2], y = latents.shape[-1])
            lout = self.latents_to_out(lout)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', h = h, x = out_h, y = out_w)
        out = self.to_out(out)

        if self.has_latents:
            return out, lout

        return out

class SimpleDecoder(nn.Module):
    def __init__(
        self,
        *,
        chan_in,
        chan_out = 3,
        num_upsamples = 4,
    ):
        super().__init__()

        layers = nn.ModuleList([])
        final_chan = chan_out
        chans = chan_in

        for ind in range(num_upsamples):
            last_layer = ind == (num_upsamples - 1)
            chan_out = chans if not last_layer else final_chan * 2
            layer = nn.Sequential(
                upsample(),
                nn.Conv2d(chans, chan_out, 3, padding = 1),
                nn.GLU(dim = 1)
            )
            layers.append(layer)
            chans //= 2

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class Discriminator_former(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        fmap_dim=64,
        fmap_max = 256,
        init_channel = 2,
    ):
        super().__init__()
        assert is_power_of_two(image_size), 'image size must be a power of 2'
        num_layers = int(log2(image_size)) - 2
        #fmap_dim = 96

        self.conv_embed = nn.Sequential(
            nn.Conv2d(init_channel, 32, kernel_size = 4, stride = 2, padding = 1),
            nn.Conv2d(32, fmap_dim, kernel_size = 3, padding = 1)
        )

        image_size //= 2
        self.ax_pos_emb_h = nn.Parameter(torch.randn(image_size, fmap_dim))
        self.ax_pos_emb_w = nn.Parameter(torch.randn(image_size, fmap_dim))

        self.image_sizes = []
        self.layers = nn.ModuleList([])
        fmap_dims = []

        for ind in range(num_layers):
            image_size //= 2
            self.image_sizes.append(image_size)

            fmap_dim_out = min(fmap_dim * 2, fmap_max)

            downsample = SumBranches([
                nn.Conv2d(fmap_dim, fmap_dim_out, 3, 2, 1),
                nn.Sequential(
                    nn.AvgPool2d(2),
                    nn.Conv2d(fmap_dim, fmap_dim_out, 3, padding = 1),
                    leaky_relu()
                )
            ])

            downsample_factor = 2 ** max(log2(image_size) - log2(32), 0)
            attn_class = partial(Attention, fmap_size = image_size, downsample_kv = downsample_factor)

            self.layers.append(nn.ModuleList([
                downsample,
                Residual(PreNorm(fmap_dim_out, attn_class(dim = fmap_dim_out))),
                Residual(PreNorm(fmap_dim_out, FeedForward(dim = fmap_dim_out, kernel_size = (3 if image_size > 4 else 1))))
            ]))

            fmap_dim = fmap_dim_out
            fmap_dims.append(fmap_dim)

        #self.aux_decoder = SimpleDecoder(chan_in = fmap_dims[-2], chan_out = init_channel, num_upsamples = num_layers)

        self.to_logits = nn.Sequential(
            Residual(PreNorm(fmap_dim, Attention(dim = fmap_dim, fmap_size = 2))),
            Residual(PreNorm(fmap_dim, FeedForward(dim = fmap_dim, kernel_size = (3 if image_size > 64 else 1)))),
            nn.Conv2d(fmap_dim, 1, 2),
            Rearrange('b () () () -> b')
        )

    def forward(self, x):
        
        x = self.conv_embed(x)

        ax_pos_emb = rearrange(self.ax_pos_emb_h, 'h c -> () c h ()') + rearrange(self.ax_pos_emb_w, 'w c -> () c () w')
        x += ax_pos_emb

        fmaps = []

        for (downsample, attn, ff), _ in zip(self.layers, self.image_sizes):
            x = downsample(x)
            x = attn(x)
            x = ff(x)

            fmaps.append(x)

        x = self.to_logits(x)


        return x
