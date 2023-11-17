from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import functools,cv2
from PIL import Image

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, feature=None):
        if feature != None:             # features to concatenate
            for layer in self:
                if isinstance(layer, TimestepBlock):
                    x = layer(x, emb, feature)
                else:
                    x = layer(x, feature)
            return x
        
        else:
            for layer in self:
                if isinstance(layer, TimestepBlock):
                    x = layer(x, emb)
                else:
                    x = layer(x)
            return x



class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self,dims=2):
        super().__init__()
        # self.channels = channels
        # self.out_channels = out_channels or channels
        # self.use_conv = use_conv
        self.dims = dims
        # if use_conv:
        #     self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        #assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        # if self.use_conv:
        #     x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, dims=2):
        super().__init__()
        # self.channels = channels
        # self.out_channels = out_channels or channels
        # self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        # if use_conv:
        #     self.op = conv_nd(
        #         dims, self.channels, self.out_channels, 3, stride=stride, padding=1
        #     )
     
        # assert self.channels == self.out_channels
        self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        # assert x.shape[1] == self.channels
        return self.op(x)

class Guided_ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        guidance_ch = None
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.guidance_ch = guidance_ch

        # [[2,3], [3,4], [4,4], [4,4]]

        self.top_injection_layer = conv_nd(dims, 4*self.guidance_ch, self.channels, kernel_size=1,padding=0)
        self.bottom_injection_layer = conv_nd(dims, 4*self.guidance_ch, self.out_channels, kernel_size=1, padding=0)

        self.in_layers = nn.Sequential(
            normalization(self.channels),
            nn.SiLU(),
            conv_nd(dims, self.channels, self.out_channels, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                self.emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        if self.dropout > 0:
            self.out_layers = nn.Sequential(
                normalization(self.out_channels),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1),
            )
        else:
            self.out_layers = nn.Sequential(
                normalization(self.out_channels),
                nn.SiLU(),
                # nn.Dropout(p=dropout),
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1),
            )      


        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb,feature):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb,feature), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb, feature):

        # TODO Conv 1x1을 이용해서 x와 guidance feature 주입 x + f  
        top_feature = self.top_injection_layer(feature)
        bottom_feature = self.bottom_injection_layer(feature)

        h = x + top_feature
        h = self.in_layers(h)
        h = h + bottom_feature

        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]  # normalization, block
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out + bottom_feature # TODO Conv 1x1을 이용해서 x와 guidance feature 주입 x + f  [B,C/k, H/k, W/k] 
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(self.channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(), 
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        if self.dropout>0:
            self.out_layers = nn.Sequential(
                normalization(self.out_channels),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1),
            )
        else:
            self.out_layers = nn.Sequential(
                normalization(self.out_channels),
                nn.SiLU(),
                # nn.Dropout(p=dropout),
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1),    
            )       

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class MSG_Resblock(nn.Module):
    def __init__(
        self,
        channels,
        out_channels=None,
        use_conv=False,
        dims=2,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv


        self.layers = nn.Sequential(
            normalization(self.channels),
            nn.SiLU(),
            conv_nd(dims, self.channels, self.out_channels, 3, padding=1),
            normalization(self.out_channels),
            nn.SiLU(),
            conv_nd(dims,self.out_channels,self.out_channels,3,padding=1)
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, self.channels, self.out_channels, kernel_size=1)

    def forward(self, x):

        h = self.layers(x)

        return self.skip_connection(x) + h



class Multiscale_Structure_Guidance(nn.Module):
    
    '''
    input y: blur image (h,w,3)

    1. RGB to Grayscale
    2. Downsample
    3. conv 3x3
    4. ResBlock (1d,2d,3d,4d)  without timeembeding
    5. conv 3x3

    output x': grayscale image (h,w,1)
    
    '''
    def __init__(self, scale, guidance_ch):
        super(Multiscale_Structure_Guidance, self).__init__()

        self.scale = scale
        self.guidance_ch = guidance_ch  # S:32, L:64

        # downscale operator
        self.down_scale = avg_pool_nd(dims =2, kernel_size = self.scale, stride = self.scale )
        
        # 3x3 convolution
        self.conv1 = nn.Conv2d(1, self.guidance_ch, kernel_size=3, padding=1)
        
        # ResBlock (1d,2d,3d,4d). 
        self.resblock1 = MSG_Resblock(channels = self.guidance_ch, out_channels=self.guidance_ch)
        self.resblock2 = MSG_Resblock(channels = self.guidance_ch, out_channels= 2 * self.guidance_ch)
        self.resblock3 = MSG_Resblock(channels = 2 * self.guidance_ch, out_channels= 3 * self.guidance_ch)
        self.resblock4 = MSG_Resblock(channels = 3 * self.guidance_ch, out_channels= 4 * self.guidance_ch)

        
        # 3x3 convolution to get back to 1 channel
        self.conv2 = nn.Conv2d(4*self.guidance_ch, 1, kernel_size=3, padding=1)
        
    def forward(self, y):

        # Convert to grayscale using torchvision
        x_gray = TF.rgb_to_grayscale(y)  # h x w x 1
        
        
        #  Downsample and add small gaussian noise
        x_ds = self.down_scale(x_gray)   # h/n x w/n x 1

        sigma = 0.01
        noise = th.randn_like(x_ds) * sigma
        x_ds = x_ds + noise

        # ##################################################  debug
        # x_start = x_ds.squeeze(0)  # 배치 차원 제거
        # x_start = x_start.mul(255).byte()
        # x_start = x_start.permute(1, 2, 0)  # CxHxW to HxWxC
        # x_start = x_start.squeeze(-1)
        # image = Image.fromarray(x_start.cpu().numpy(), 'L')
        # image.save('original.jpg')

        # x_start = x_1.squeeze(0)  # 배치 차원 제거
        # x_start = x_start.mul(255).byte()
        # x_start = x_start.permute(1, 2, 0)  # CxHxW to HxWxC
        # x_start = x_start.squeeze(-1)
        # image = Image.fromarray(x_start.cpu().numpy(), 'L')
        # image.save('operator1.jpg')

        # sigma = 0.01
        # noise = th.randn_like(x_ds) * sigma
        # x_2 = x_ds + noise

        # x_start = x_2.squeeze(0)  # 배치 차원 제거
        # x_start = x_start.mul(255).byte()
        # x_start = x_start.permute(1, 2, 0)  # CxHxW to HxWxC
        # x_start = x_start.squeeze(-1)
        # image = Image.fromarray(x_start.cpu().numpy(), 'L')
        # image.save('operator2.jpg')

        ######################################################

        
        x = self.conv1(x_ds)              # h/n x w/n x c(guidance channel)
        
        
        x = self.resblock1(x)             # h/n x w/n x d
        x = self.resblock2(x)             # h/n x w/n x 2d
        x = self.resblock3(x)             # h/n x w/n x 3d
        feature = self.resblock4(x)       # h/n x w/n x 4d
        
        
        x_prime = self.conv2(feature)     # h x w x 1

        return x_prime,feature
    


class UNet_MSG(nn.Module):
    """
    The full UNet_MSG model with timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param guidance_channels: s:32, L:64
    :param learn_sigma:

    """

    def __init__(
        self,
        in_channels,
        guidance_channels,
        dropout=0,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        use_scale_shift_norm=False,
        learn_sigma = False
    ):
        super().__init__()

        print(f'guidance channel:{guidance_channels}')

        self.out_channels = 3 if not learn_sigma else 6
        print(f'learn sigma is {learn_sigma}, out_channel is {self.out_channels}')
        print(f'use_scale_shift_norm is {use_scale_shift_norm}')
        print(f'drop out {dropout}')

        self.in_channels = in_channels
        self.guidance_ch = guidance_channels
        self.dropout = dropout
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        time_embed_dim = self.guidance_ch * 4
        self.time_embed = nn.Sequential(
            linear(self.guidance_ch, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # 1st, 2nd blocks in UNet
        self.preprocess_blocks = nn.ModuleList([])
        self.preprocess_blocks.append(TimestepEmbedSequential(conv_nd(dims, in_channels, self.guidance_ch, 3, padding=1)))
        self.preprocess_blocks.append(TimestepEmbedSequential(ResBlock(channels=self.guidance_ch,
                                              emb_channels = time_embed_dim,
                                              out_channels= 2 * self.guidance_ch,
                                              use_scale_shift_norm=use_scale_shift_norm,
                                              dropout=dropout)))

        # 3rd ~ 9rd blocks in UNet
        self.input_blocks=nn.ModuleList([])
        
        self.nums_of_guided_resblock = 4
        self.guidance_dim = [[2,3], [3,4], [4,4], [4,4]]
        self.nums_of_resblock = 3
        self.resblock_dim = [[8,3], [6,2], [4,1]]

        for i in range(self.nums_of_guided_resblock):
            layers=[]
            if i != 3:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(dims=2)))

            guidance_dim = self.guidance_dim[i]
            layers.append(
                Guided_ResBlock(
                    channels=guidance_dim[0] * self.guidance_ch,
                    emb_channels = time_embed_dim,
                    out_channels = guidance_dim[1] * self.guidance_ch,
                    guidance_ch=self.guidance_ch, 
                    use_scale_shift_norm=use_scale_shift_norm,
                    dropout=dropout
                )

            )

            self.input_blocks.append(TimestepEmbedSequential(*layers))


        # 10rd ~ 15rd blocks in UNet
        self.output_blocks = nn.ModuleList([])
        for i in range(self.nums_of_resblock):
            resblock_dim = self.resblock_dim[i]
            layers=[]

            self.output_blocks.append(TimestepEmbedSequential(Upsample(dims=2)))
            layers.append(
                ResBlock(
                    channels = resblock_dim[0] * self.guidance_ch,
                    emb_channels = time_embed_dim,
                    out_channels = resblock_dim[1] * self.guidance_ch,
                    use_scale_shift_norm=use_scale_shift_norm,
                    dropout=dropout
                )
            )
            self.output_blocks.append(TimestepEmbedSequential(*layers))

        # last block in UNet
        self.out = conv_nd(dims,self.guidance_ch, self.out_channels, 3, padding=1)

        # guidance block
        self.MSG_modules = nn.ModuleList([])
        for i in range(1,4):
            scale_k = 2 ** i
            self.MSG_modules.append(Multiscale_Structure_Guidance(scale=scale_k, guidance_ch=self.guidance_ch))


    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        # self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        # self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
        self.preprocess_blocks.apply(convert_module_to_f32)
        self.final_blocks.apply(convert_module_to_f32)
        self.MSG_modules.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None,**kwargs):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: blur image
        :return: an [N x C x ...] Tensor of outputs.
        """
        
        if 'blur' in kwargs.keys():
            y = kwargs['blur']

        blur_y = y.type(self.dtype)  # blur y image
        sharp_x = x.type(self.dtype)

        h = th.cat((blur_y, sharp_x), dim=1)  # concat blur and sharp image

        hs = []
        features = []
        x_primes = []
        emb = self.time_embed(timestep_embedding(timesteps, self.guidance_ch))  # timestep embeding
        
        # Guidance block: extract features
        if len(kwargs) == 0:                                    # train
            for module in self.MSG_modules:                   # guidance features, 1/2, 1/4, 1/8, 1/8
                x_prime,feature = module(blur_y)
                features.append(feature)                      # features will be injected to self.input_blocks
                x_primes.append(x_prime)                      # MSG module's downscaled x'
            features.append(feature)                          # last 1/8 downscale feature

        else:
                                                              # sampling 
            for module in self.MSG_modules:                   # guidance features, 1/2, 1/4, 1/8, 1/8
                x_prime,feature = module(blur_y)
                features.append(feature)                      # features will be inject to self.input_blocks
                x_primes.append(x_prime)                      # MSG module's downscaled x'
            features.append(feature)  
            # self.test_features = features
            # self.test_xprimes = x_primes


        for module in self.preprocess_blocks:             # conv3x3(nn.Module) -> resblock(TimestepBlcok)
            h = module(h, emb)
        hs.append(h)                                      # will be concatenated with Upsample block

        for idx,module in enumerate(self.input_blocks):   # DS->GR-> DS->GR-> DS->GR-> GR
            if isinstance(module[0], Downsample):
                h = module(h,emb)                         # if module is GuidedResblock
            else:
                h = module(h, emb, features.pop(0))               

            if idx ==1 or idx == 3:                       # will be concatenated with Upsample block
                hs.append(h)


        for module in self.output_blocks:                 # Upsample -> concat -> Resblock
            if isinstance(module[0],Upsample):
                h=module(h,emb)
            else:
                h = th.cat([h,hs.pop()], dim=1)
                h = module(h,emb)            
            
        h = self.out(h)    

        return h, x_primes