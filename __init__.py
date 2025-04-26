import logging
import math
from einops import rearrange
from matplotlib.style import available
from torch import Tensor, nn
import numpy as np
from comfy import model_management
import torch

import comfy.ldm.flux.layers


torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

torch.backends.cudnn.enabled = False

import comfy.ldm.flux
import comfy.ldm.modules.attention
import comfy.ldm.modules.diffusionmodules.model

from comfy.ldm.modules.diffusionmodules.model import (
    Normalize,
    ResnetBlock,
    Upsample,
    nonlinearity,
)
from comfy.ldm.modules.attention import (
    attention_sub_quad,
    attention_pytorch,
    attention_split,
    attention_xformers,
)
from comfy.ldm.modules.diffusionmodules.model import (
    normal_attention,
    pytorch_attention,
    xformers_attention,
)

import comfy.ops

ops = comfy.ops.disable_weight_init

import torch
import os

os.add_dll_directory(os.path.join(os.environ["HIP_PATH"], "bin"))

HIP_VER = 57

if "6.2" in os.environ["HIP_PATH"]:
    from .hip62 import flash_attn_wmma
    from .hip62 import ck_fttn_pyb

    HIP_VER = 62
    print("Detected HIP Version:6.2")
else:
    from .hip57 import flash_attn_wmma
    from .hip57 import ck_fttn_pyb

    HIP_VER = 57
    print("Detected HIP Version:5.7")

select_attention_algorithm = None
select_attention_vae_algorithm = None


def rocm_fttn(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False):

    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )

    if mask is not None:
        # add a batch dimension if there isn't already one
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        # add a heads dimension if there isn't already one
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

    if dim_head <= 128 and HIP_VER == 62:
        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))
        out = ck_fttn_pyb.fwd(
            q, k, v, None, 0, q.shape[-1] ** -0.5, False, False, None
        )[0]
        if not skip_output_reshape:
            out = out.reshape(b, -1, heads * dim_head)
        return out

    # print(q.shape, k.shape, v.shape) # [1, 24, 6400, 128] flux
    dtype = q.dtype
    q, k, v = map(
        lambda t: t.to(torch.float16),
        (q, k, v),
    )

    Br = 64
    Bc = 256
    if dim_head >= 272:
        Br = 32
        Bc = 128

    out = flash_attn_wmma.forward(q, k, v, Br, Bc, False, dim_head**-0.5, False)[0]
    if not skip_output_reshape:
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    return out.to(dtype)

def rocm_fttn_vae(q, k, v, *args, **kwargs):

    B, C, H, W = q.shape
    q, k, v = map(
        lambda t: t.view(B, 1, C, -1).transpose(2, 3),
        (q, k, v),
    )

    d_q = q.shape[-1]

    Br = 64
    Bc = 256
    if d_q >= 272:
        Br = 32
        Bc = 128

    o = flash_attn_wmma.forward(q, k, v, Br, Bc, False, d_q**-0.5, False)[0]
    return o.transpose(2, 3).reshape(B, C, H, W)

class AttnBlock_hijack(nn.Module):
    def __init__(self, in_channels, conv_op=ops.Conv2d):
        global select_attention_vae_algorithm
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = conv_op(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = conv_op(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = conv_op(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = conv_op(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

        self.optimized_attention = select_attention_vae_algorithm

    def forward(self, x):
        global select_attention_vae_algorithm
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        if self.optimized_attention != select_attention_vae_algorithm:
            self.optimized_attention = select_attention_vae_algorithm

        h_ = self.optimized_attention(q, k, v)

        h_ = self.proj_out(h_)

        return x + h_


class Decoder_hijack(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 conv_out_op=ops.Conv2d,
                 resnet_op=ResnetBlock,
                 attn_op=AttnBlock_hijack,
                 conv3d=False,
                 time_compress=None,
                **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        if conv3d:
            conv_op = VideoConv3d
            conv_out_op = VideoConv3d
            mid_attn_conv_op = ops.Conv3d
        else:
            conv_op = ops.Conv2d
            mid_attn_conv_op = ops.Conv2d

        # compute block_in and curr_res at lowest res
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        logging.debug("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = conv_op(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = resnet_op(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       conv_op=conv_op)
        self.mid.attn_1 = attn_op(block_in, conv_op=mid_attn_conv_op)
        self.mid.block_2 = resnet_op(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       conv_op=conv_op)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(resnet_op(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,
                                         conv_op=conv_op))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(attn_op(block_in, conv_op=conv_op))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                scale_factor = 2.0
                if time_compress is not None:
                    if i_level > math.log2(time_compress):
                        scale_factor = (1.0, 2.0, 2.0)

                up.upsample = Upsample(block_in, resamp_with_conv, conv_op=conv_op, scale_factor=scale_factor)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = conv_out_op(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, **kwargs):
        self.last_z_shape = z.shape
        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb, **kwargs)
        h = self.mid.attn_1(h, **kwargs)
        h = self.mid.block_2(h, temb, **kwargs)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb, **kwargs)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, **kwargs)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h, **kwargs)
        if self.tanh_out:
            h = torch.tanh(h)
        return h

def make_attn(in_channels, attn_type="vanilla", conv_op=ops.Conv2d):
    print("  AttnBlock_hijack")
    return AttnBlock_hijack(in_channels, conv_op=conv_op)


# comfy.ldm.modules.diffusionmodules.model.make_attn = make_attn
# comfy.ldm.modules.diffusionmodules.model.AttnBlock = AttnBlock_hijack
# comfy.ldm.modules.diffusionmodules.model.Decoder = Decoder_hijack
# comfy.ldm.modules.attention.optimized_attention = select_attention_algorithm
# comfy.ldm.flux.math.optimized_attention = select_attention_vae_algorithm

# setattr(comfy.ldm.modules.attention,"optimized_attention",select_attention_algorithm)
# setattr(comfy.ldm.modules.diffusionmodules.model, "make_attn", make_attn)
# setattr(comfy.ldm.modules.diffusionmodules.model, "AttnBlock", AttnBlock_hijack)
# setattr(comfy.ldm.modules.diffusionmodules.model, "Decoder", Decoder_hijack)


class AttnOptSelector:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        available_attns = []
        available_attns.append("Flash-Attention-v2")
        # if model_management.xformers_enabled():
        available_attns.append("xformers")
        available_attns.append("pytorch")
        available_attns.append("split")
        available_attns.append("sub-quad")

        return {
            "required": {
                "sampling_attention": (available_attns,),
                "vae_attention": (available_attns,),
                "Model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("MODEL",)

    FUNCTION = "test"
    OUTPUT_NODE = True

    CATEGORY = "_for_testing"

    def test(self, sampling_attention, vae_attention, Model):
        global select_attention_algorithm, select_attention_vae_algorithm
        print("  Select optimized attention:", sampling_attention, vae_attention)
        if sampling_attention == "xformers":
            select_attention_algorithm = attention_xformers
        elif sampling_attention == "pytorch":
            select_attention_algorithm = attention_pytorch
        elif sampling_attention == "split":
            select_attention_algorithm = attention_split
        elif sampling_attention == "sub-quad":
            select_attention_algorithm = attention_sub_quad
        elif sampling_attention == "Flash-Attention-v2":
            select_attention_algorithm = rocm_fttn

        if vae_attention == "xformers":
            select_attention_vae_algorithm = xformers_attention
        elif vae_attention == "pytorch":
            select_attention_vae_algorithm = pytorch_attention
        elif vae_attention == "split":
            select_attention_vae_algorithm = normal_attention
        elif vae_attention == "sub-quad":
            select_attention_vae_algorithm = normal_attention
        elif vae_attention == "Flash-Attention-v2":
            select_attention_vae_algorithm = rocm_fttn_vae

        comfy.ldm.modules.diffusionmodules.model.make_attn = make_attn
        comfy.ldm.modules.diffusionmodules.model.AttnBlock = AttnBlock_hijack
        comfy.ldm.modules.diffusionmodules.model.Decoder = Decoder_hijack
        comfy.ldm.modules.attention.optimized_attention = select_attention_algorithm

        comfy.ldm.flux.math.optimized_attention = select_attention_algorithm
        setattr(comfy.ldm.flux.math, "optimized_attention", select_attention_algorithm)

        setattr(
            comfy.ldm.modules.attention,
            "optimized_attention",
            select_attention_algorithm,
        )
        setattr(comfy.ldm.modules.diffusionmodules.model, "make_attn", make_attn)
        setattr(comfy.ldm.modules.diffusionmodules.model, "AttnBlock", AttnBlock_hijack)
        setattr(comfy.ldm.modules.diffusionmodules.model, "Decoder", Decoder_hijack)

        return (Model, sampling_attention + vae_attention)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {"AttnOptSelector": AttnOptSelector}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {"AttnOptSelector": "optimized attention selector"}
