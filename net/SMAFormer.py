# ------------------------------------------------------------
# Copyright (c) University of Macau,
# Shenzhen Institutes of Advanced Technology，Chinese Academy of Sciences.
# Licensed under the Apache License 2.0 [see LICENSE for details]
# Written by FuChen Zheng(YC379501)
# ------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
# from net.init_weights import init_weights
# from dataset import dataset
from torch.nn import Softmax
# from axial_attention import AxialAttention
from einops import rearrange
import math
from ptflops import get_model_complexity_info


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ResUformer', default=None,
                        help='model name: (default: arch+timestamp)')

    args = parser.parse_args()

    return args


class RowAttention(nn.Module):

    def __init__(self, in_dim, q_k_dim):
        '''
        Parameters
        ----------
        in_dim : int
            channel of input img tensor
        q_k_dim: int
            channel of Q, K vector
        '''
        super(RowAttention, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.in_dim, kernel_size=1)
        self.softmax = Softmax(dim=2)
        self.gamma = nn.Parameter(torch.zeros(1)).to(self.device)

    def forward(self, x):
        '''
        Parameters
        ----------
        x : Tensor
            4-D , (batch, in_dims, height, width) -- (b,c1,h,w)
        '''

        ## c1 = in_dims; c2 = q_k_dim
        b, _, h, w = x.size()

        Q = self.query_conv(x)  # size = (b,c2, h,w)
        K = self.key_conv(x)  # size = (b, c2, h, w)
        V = self.value_conv(x)  # size = (b, c1,h,w)

        Q = Q.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)  # size = (b*h,w,c2)
        K = K.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)  # size = (b*h,c2,w)
        V = V.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)  # size = (b*h, c1,w)

        # size = (b*h,w,w) [:,i,j] 表示Q的所有h的第 Wi行位置上所有通道值与 K的所有h的第 Wj列位置上的所有通道值的乘积，
        # 即(1,c2) * (c2,1) = (1,1)
        row_attn = torch.bmm(Q, K)
        ########
        # 此时的 row_atten的[:,i,0:w]
        # 表示Q的所有h的第 Wi行位置上所有通道值与 K的所有列(0:w)的逐个位置上的所有通道值的乘积
        # 此操作即为 Q的某个（i,j）与 K的（i,0:w）逐个位置的值的乘积，得到行attn
        ########

        # 对row_attn进行softmax
        row_attn = self.softmax(row_attn)  # 对列进行softmax，即[k,i,0:w] ，某一行的所有列加起来等于1，

        # size = (b*h,c1,w) 这里先需要对row_atten进行 行列置换，使得某一列的所有行加起来等于1
        # [:,i,j]即为V的所有行的某个通道上，所有列的值 与 row_attn的行的乘积，即求权重和
        out = torch.bmm(V, row_attn.permute(0, 2, 1))

        # size = (b,c1,h,w)
        out = out.view(b, h, -1, w).permute(0, 2, 1, 3)

        out = self.gamma * out + x

        return out


class ColAttention(nn.Module):

    def __init__(self, in_dim, q_k_dim):
        '''
        Parameters
        ----------
        in_dim : int
            channel of input img tensor
        q_k_dim: int
            channel of Q, K vector
        '''
        super(ColAttention, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.in_dim, kernel_size=1)
        self.softmax = Softmax(dim=2)
        self.gamma = nn.Parameter(torch.zeros(1)).to(self.device)

    def forward(self, x):
        '''
        Parameters
        ----------
        x : Tensor
            4-D , (batch, in_dims, height, width) -- (b,c1,h,w)
        '''

        ## c1 = in_dims; c2 = q_k_dim
        b, _, h, w = x.size()

        Q = self.query_conv(x)  # size = (b,c2, h,w)
        K = self.key_conv(x)  # size = (b, c2, h, w)
        V = self.value_conv(x)  # size = (b, c1,h,w)

        Q = Q.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)  # size = (b*w,h,c2)
        K = K.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)  # size = (b*w,c2,h)
        V = V.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)  # size = (b*w,c1,h)

        # size = (b*w,h,h) [:,i,j] 表示Q的所有W的第 Hi行位置上所有通道值与 K的所有W的第 Hj列位置上的所有通道值的乘积，
        # 即(1,c2) * (c2,1) = (1,1)
        col_attn = torch.bmm(Q, K)
        ########
        # 此时的 col_atten的[:,i,0:w] 表示Q的所有W的第 Hi行位置上所有通道值与 K的所有W的 所有列(0:h)的逐个位置上的所有通道值的乘积
        # 此操作即为 Q的某个（i,j）与 K的（i,0:h）逐个位置的值的乘积，得到列attn
        ########

        # 对row_attn进行softmax
        col_attn = self.softmax(col_attn)  # 对列进行softmax，即[k,i,0:w] ，某一行的所有列加起来等于1，

        # size = (b*w,c1,h) 这里先需要对col_atten进行 行列置换，使得某一列的所有行加起来等于1
        # [:,i,j]即为V的所有行的某个通道上，所有列的值 与 col_attn的行的乘积，即求权重和
        out = torch.bmm(V, col_attn.permute(0, 2, 1))

        # size = (b,c1,h,w)
        out = out.view(b, w, -1, h).permute(0, 2, 3, 1)

        out = self.gamma * out + x

        return out


# ------------------------------------------------------------
# Copyright (c) University of Macau,
# Shenzhen Institutes of Advanced Technology，Chinese Academy of Sciences.
# Licensed under the Apache License 2.0 [see LICENSE for details]
# Written by FuChen Zheng(YC379501)
# ------------------------------------------------------------
class Modulator(nn.Module):
    def __init__(self, in_ch, out_ch, with_pos=True):
        super(Modulator, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.rate = [1, 6, 12, 18]
        self.with_pos = with_pos
        self.patch_size = 2
        self.bias = nn.Parameter(torch.zeros(1, out_ch, 1, 1))

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.CA_fc = nn.Sequential(
            nn.Linear(in_ch, in_ch // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // 16, in_ch, bias=False),
            nn.Sigmoid(),
        )

        # Pixel Attention
        self.PA_conv = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        self.PA_bn = nn.BatchNorm2d(in_ch)
        self.sigmoid = nn.Sigmoid()

        # Spatial Attention
        self.SA_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=rate, dilation=rate),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_ch)
            ) for rate in self.rate
        ])
        self.SA_out_conv = nn.Conv2d(len(self.rate) * out_ch, out_ch, 1)

        self.output_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_ch)
        self._init_weights()

        self.pj_conv = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=self.patch_size + 1,
                         stride=self.patch_size, padding=self.patch_size // 2)
        self.pos_conv = nn.Conv2d(self.out_ch, self.out_ch, kernel_size=3, padding=1, groups=self.out_ch, bias=True)
        self.layernorm = nn.LayerNorm(self.out_ch, eps=1e-6)

    def forward(self, x):
        res = x
        pa = self.PA(x)
        ca = self.CA(x)

        # Softmax(PA @ CA)
        pa_ca = torch.softmax(pa @ ca, dim=-1)

        # Spatial Attention
        sa = self.SA(x)

        # (Softmax(PA @ CA)) @ SA
        out = pa_ca @ sa
        out = self.norm(self.output_conv(out))
        out = out + self.bias
        synergistic_attn = out + res
        return synergistic_attn

    # def forward(self, x):
    #     pa_out = self.pa(x)
    #     ca_out = self.ca(x)
    #     sa_out = self.sa(x)
    #     # Concatenate along channel dimension
    #     combined_out = torch.cat([pa_out, ca_out, sa_out], dim=1)
    #
    #     return self.norm(self.output_conv(combined_out))



    def PE(self, x):
        proj = self.pj_conv(x)

        if self.with_pos:
            pos = proj * self.sigmoid(self.pos_conv(proj))

        pos = pos.flatten(2).transpose(1, 2)  # BCHW -> BNC
        embedded_pos = self.layernorm(pos)

        return embedded_pos

    def PA(self, x):
        attn = self.PA_conv(x)
        attn = self.PA_bn(attn)
        attn = self.sigmoid(attn)
        return x * attn

    def CA(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.CA_fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

    def SA(self, x):
        sa_outs = [block(x) for block in self.SA_blocks]
        sa_out = torch.cat(sa_outs, dim=1)
        sa_out = self.SA_out_conv(sa_out)
        return sa_out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    '''
        使用Xavier初始化（也称为Glorot初始化）作为初始化权重的方法。Xavier初始化是一种流行的技术，可帮助更好地初始化神经网络的权重，促进模型的训练效果。
        1.nn.init.xavier_uniform_(m.weight): 
            对于Conv2d层，使用Xavier均匀分布初始化权重，这种初始化方法适用于Sigmoid和Tanh等激活函数。
        2.nn.init.kaiming_normal_(m.weight): 
            对于Conv2d层，使用Kaiming正态分布初始化权重，这种初始化方法适用于ReLU激活函数。它会根据权重张量的形状和激活函数的特性来初始化权重。
        '''
    # def _init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.xavier_uniform_(m.weight)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)


'''
Synergistic Multi-Attention
'''
# ------------------------------------------------------------
# Copyright (c) University of Macau,
# Shenzhen Institutes of Advanced Technology，Chinese Academy of Sciences.
# Licensed under the Apache License 2.0 [see LICENSE for details]
# Written by FuChen Zheng(YC379501)
# ------------------------------------------------------------
class SMA(nn.Module):
    def __init__(self, feature_size, num_heads, dropout):
        super(SMA, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feature_size, num_heads=num_heads, dropout=dropout)
        self.combined_modulator = Modulator(feature_size, feature_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, value, key, query):
        MSA = self.attention(query, key, value)[0]

        # 将输出转换为适合AttentionBlock的输入格式
        batch_size, seq_len, feature_size = MSA.shape
        MSA = MSA.permute(0, 2, 1).view(batch_size, feature_size, int(seq_len**0.5), int(seq_len**0.5))
        # 通过CombinedModulator进行multi-attn fusion
        synergistic_attn = self.combined_modulator.forward(MSA)


        # 将输出转换回 (batch_size, seq_len, feature_size) 格式
        x = synergistic_attn.view(batch_size, feature_size, -1).permute(0, 2, 1)

        return x
class MSA(nn.Module):
    def __init__(self, feature_size, num_heads, dropout):
        super(MSA, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feature_size, num_heads=num_heads, dropout=dropout)
        self.combined_modulator = Modulator(feature_size, feature_size)

    def forward(self, value, key, query):
        attention = self.attention(query, key, value)[0]

        return attention

class E_MLP(nn.Module):
    def __init__(self, feature_size, forward_expansion, dropout):
        super(E_MLP, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_size, forward_expansion * feature_size),
            nn.GELU(),
            nn.Linear(forward_expansion * feature_size, feature_size)
        )
        self.linear1 = nn.Linear(feature_size, forward_expansion * feature_size)
        self.act = nn.GELU()
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(in_channels=forward_expansion * feature_size, out_channels=forward_expansion * feature_size, kernel_size=3, padding=1, groups=1)

        # pixelwise convolution
        self.pixelwise_conv = nn.Conv2d(in_channels=forward_expansion * feature_size, out_channels=forward_expansion * feature_size, kernel_size=3, padding=1)

        self.linear2 = nn.Linear(forward_expansion * feature_size, feature_size)

    def forward(self, x):
        b, hw, c = x.size()
        feature_size = int(math.sqrt(hw))

        x = self.linear1(x)
        x = self.act(x)
        x = rearrange(x, 'b (h w) (c) -> b c h w', h=feature_size, w=feature_size)
        x = self.depthwise_conv(x)
        x = self.pixelwise_conv(x)
        x = rearrange(x, 'b c h w -> b (h w) (c)', h=feature_size, w=feature_size)
        out = self.linear2(x)

        return out


class SMAFormerBlock(nn.Module):
    def __init__(self, ch_in, ch_out, heads, dropout, forward_expansion, fusion_gate):
        super(SMAFormerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(ch_out)
        self.norm2 = nn.LayerNorm(ch_out)
        self.MSA = MSA(ch_out, heads, dropout)
        self.synergistic_multi_attention = SMA(ch_out, heads, dropout)
        self.e_mlp = E_MLP(ch_out, forward_expansion, dropout)
        self.fusion_gate = fusion_gate
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, value, key, query, res):
        if self.fusion_gate:
            attention = self.synergistic_multi_attention(query, key, value)
        else:
            attention = self.MSA(query, key, value)
        query = self.dropout(self.norm1(attention + res))
        feed_forward = self.e_mlp(query)
        out = self.dropout(self.norm2(feed_forward + query))
        return out

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, heads, dropout, forward_expansion, num_layers, fusion_gate):
        super(EncoderBlock, self).__init__()
        self.layers = nn.ModuleList([
            SMAFormerBlock(in_ch, out_ch, heads, dropout, forward_expansion, fusion_gate) for _ in range(num_layers)
        ])
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x, res):
        '''[B, H*W, C]'''
        for layer in self.layers:
            x = layer(res, res, x, x)

        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, heads, dropout, forward_expansion, num_layers, fusion_gate):
        super(DecoderBlock, self).__init__()
        self.layers = nn.ModuleList([
            SMAFormerBlock(in_ch, out_ch, heads, dropout, forward_expansion, fusion_gate) for _ in range(num_layers)
        ])
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x, res):
        '''[B, H*W, C]'''
        for layer in self.layers:
            x = layer(res, res, x, x)

        return x



class Upsample_(nn.Module):
    def __init__(self, scale=2):
        super(Upsample_, self).__init__()

        self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale)

    def forward(self, x):
        return self.upsample(x)

class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)

class Upsample_Transpose(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample_Transpose, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)

class Cross_AttentionBlock(nn.Module):
    def __init__(self, input_encoder, input_decoder, output_dim):
        super(Cross_AttentionBlock, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.ReLU(),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
            # nn.MaxPool2d(2, 2),
        )

        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(input_decoder),
            nn.ReLU(),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
        )

        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2

class SMAFormer(nn.Module):
    def __init__(self, args):
        super(SMAFormer, self).__init__()
        self.args = args
        in_channels = 3
        n_classes = 3
        patch_size = 2
        filters = [16, 32, 64, 128, 256, 512]
        encoder_layer = 1
        decoder_layer = 1
        self.patch_size = patch_size
        self.filters = filters
        #layer 1 + embedding
        # Licensed under the Apache License 2.0 [see LICENSE for details]
        # Written by FuChen Zheng
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1)
        )

        self.patch_embedding1 = Modulator(in_ch=filters[0], out_ch=filters[1])

        self.EncoderBlock1 = EncoderBlock(in_ch=filters[1], out_ch=filters[1], heads=8, dropout=0.1, forward_expansion=2,
                                                    num_layers=encoder_layer, fusion_gate=True)

        self.residual_conv1 = ResidualConv(filters[1], filters[2], 2, 1)

        self.patch_embedding2 = Modulator(in_ch=filters[2], out_ch=filters[3])

        self.EncoderBlock2 = EncoderBlock(in_ch=filters[3], out_ch=filters[3], heads=8, dropout=0.1, forward_expansion=2,
                                                    num_layers=encoder_layer, fusion_gate=True)

        self.residual_conv2 = ResidualConv(filters[3], filters[4], 2, 1)

        self.patch_embedding3 = Modulator(in_ch=filters[4], out_ch=filters[5])

        self.EncoderBlock3 = EncoderBlock(in_ch=filters[5], out_ch=filters[5], heads=8, dropout=0.1, forward_expansion=2,
                                                    num_layers=encoder_layer, fusion_gate=True)

        self.EncoderBlock4 = EncoderBlock(in_ch=filters[5], out_ch=filters[5], heads=8, dropout=0.1, forward_expansion=2,
                                                    num_layers=encoder_layer, fusion_gate=True)

        self.DecoderBlock1 = DecoderBlock(in_ch=filters[5], out_ch=filters[5], heads=8, dropout=0.1, forward_expansion=2,
                                                    num_layers=decoder_layer, fusion_gate=True)

        self.upsample = Upsample_(2)
        self.upsample_transpose1 = Upsample_Transpose(filters[5], filters[4], kernel=2, stride=2)
        self.DecoderBlock2 = DecoderBlock(in_ch=filters[5], out_ch=filters[5], heads=8, dropout=0., forward_expansion=2,
                                                    num_layers=decoder_layer, fusion_gate=True)

        self.upsample_transpose2 = Upsample_Transpose(filters[5], filters[4], kernel=2, stride=2)
        self.upsample_transpose3 = Upsample_Transpose(filters[4]+filters[3], filters[3], kernel=1, stride=1)
        self.DecoderBlock3 = DecoderBlock(in_ch=filters[3], out_ch=filters[3], heads=8, dropout=0.1, forward_expansion=2,
                                                    num_layers=decoder_layer, fusion_gate=True)

        self.upsample_transpose4 = Upsample_Transpose(filters[3], filters[2], kernel=2, stride=2)

        self.upsample_transpose5 = Upsample_Transpose(filters[3], filters[2], kernel=2, stride=2)
        self.DecoderBlock4 = DecoderBlock(in_ch=filters[2], out_ch=filters[2], heads=8, dropout=0.1, forward_expansion=2,
                                                    num_layers=decoder_layer, fusion_gate=True)
        self.adjust = Upsample_Transpose(filters[1], filters[2], kernel=1, stride=1)
        self.upsample_transpose6 = Upsample_Transpose(filters[2], filters[1], kernel=2, stride=2)
        self.output_layer1 = nn.Sequential(nn.Conv2d(filters[1], filters[0], 1))
        self.output_layer2 = nn.Sequential(nn.Conv2d(filters[0], n_classes, 1))

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x) #[16,3,512x2]->[16,16,512x2]

        x2 = self.patch_embedding1.PE(x1) #[16,16,512,512]->[16,256*256,32]
        e1 = self.EncoderBlock1(x2, x2) #[16,256*256,32]
        b, num_patch, c = e1.size()
        x2 = e1.view(b, c, int(num_patch ** 0.5), int(num_patch ** 0.5))  # 调整回卷积形状 [16,256*256,32]->[16,32,256,256]
        x2 = self.residual_conv1(x2) #[16,32,256,256]->[16,64,128,128]

        x3 = self.patch_embedding2.PE(x2) #[16,64,128,128]->[16,64*64,128]
        e2 = self.EncoderBlock2(x3, x3) #[16,64*64,128]
        b, num_patch, c = e2.size()
        e2 = e2.view(b, c, num_patch //self.filters[2], num_patch //self.filters[2]) #调整回卷积形状[16,64*64,128]->[16,128,64,64]
        x3 = self.residual_conv2(e2) #[16,128,64,64]->[16,256,32,32]

        x4 = self.patch_embedding3.PE(x3) #[16,256,32,32]->[16,16*16,512]
        e3 = self.EncoderBlock3(x4, x4) #[16,16*16,512]

        e4 = self.EncoderBlock4(e3, e3)  # [16, 16*16(h*w), c=512]

        '''Decoder'''
        x5 = self.DecoderBlock1(e4, e4)  #[16,512,16,16]
        b, hw, c = x5.size()
        h = w = int(hw ** 0.5)
        x5 = x5.contiguous().permute(0, 2, 1).view(b, c, h, w)  #调整回卷积形状 #[16, 16*16(h*w), c=512]->[16,512,16,16]
        x6 = self.upsample_transpose1(x5)   #[16,512,16,16]->[16,256,32,32]
        x6 = torch.cat([x6, x3], dim=1) #[16,256+256,32,32]
        b, c, h, w = x6.size()
        x6 = x6.view(b, c, h * w).contiguous().permute(0, 2, 1) ##[16,32*32,512]
        b, num_patch, c = e3.size()
        e3 = e3.view(b, c, int(num_patch ** 0.5), int(num_patch ** 0.5))  # [16,16*16,512]->[16,512,16,16]
        e3 = self.upsample(e3) #[16,512,16,16]->[16,512,32,32]
        b, c, h, w = e3.size()
        e3 = e3.view(b, c, h * w).contiguous().permute(0, 2, 1)  ##[16,32*32,512]
        x6 = self.DecoderBlock2(x6, e3) #[16,32*32,512]skip_con[16,32*32,512]
        b, hw, c = x6.size()
        h = w = int(hw ** 0.5)
        x6 = x6.permute(0, 2, 1).contiguous().view(b, c, h, w)  #[16,32*32,512]->[16,512,32,32]

        x7 = self.upsample_transpose2(x6) #[16, 512, 32, 32]->[16, 256, 64, 64]
        x7 = torch.cat([x7, e2], dim=1) #[16, 256, 64, 64]+[16,128,64,64]->[16,256+128,64,64]
        x7 = self.upsample_transpose3(x7)   #[16,256+128,64,64]->[16,128,64,64]
        b, c, h, w = x7.size()
        x7 = x7.view(b, c, h * w).contiguous().permute(0, 2, 1)  #[16,128,64,64]->[16,64*64,128]
        b, c, h, w = e2.size()
        e2 = e2.view(b, c, h * w).contiguous().permute(0, 2, 1)  ##[16,64*64,128]
        x7 = self.DecoderBlock3(x7, e2) #[16,64*64,128]connect[16,64*64,128]
        b, hw, c = x7.size()
        h = w = int(hw ** 0.5)
        x7 = x7.permute(0, 2, 1).contiguous().view(b, c, h, w)  #[16,64*64,128]->[16,128,64,64]

        x8 = self.upsample_transpose4(x7)   #[16,128,64,64]->[16,64,128,128]
        x8 = torch.cat([x8, x2], dim=1) #[16,64,128,128]+[16,64,128,128]->[16,64+64,128,128]
        x8 = self.upsample_transpose5(x8)   #[16,64+64,128,128]->[16,64,256,256]
        b, c, h, w = x8.size()
        x8 = x8.view(b, c, h * w).contiguous().permute(0, 2, 1)  #[16,64,256,256]->[16,256*256,64]
        b_e1, hw_e1, c_e1 = e1.size()
        h_e1 = w_e1 = int(hw_e1 ** 0.5)
        e1 = e1.permute(0, 2, 1).contiguous().view(b_e1, c_e1, h_e1, w_e1)  # [16,256*256,32]->[16,32,256,256]#
        e1 = self.adjust(e1)    # [16,32,256,256]->[16,64,256,256]
        b_e1, c_e1, h_e1, w_e1 = e1.size()
        e1 = e1.view(b_e1, c_e1, h_e1 * w_e1).contiguous().permute(0, 2, 1)  #[16,64,256,256]->[16,256*256,64]
        x8 = self.DecoderBlock4(x8, e1)
        b, hw, c = x8.size()
        h = w = int(hw ** 0.5)
        x8 = x8.permute(0, 2, 1).contiguous().view(b, c, h, w)  #[16,256*256,64]->[16,64,256,256]
        x8 = self.upsample_transpose6(x8)   #[16,64,256,256]->[16,32,512,512]

        out = self.output_layer1(x8)    #[16,32,512,512]->[16,16,512,512]
        out = self.output_layer2(out) #[16,16,512,512]->[16,num_classes,512,512]

        return out





'''
exp1: se block -> Fusion 3 attn
success

exp2: aspp -> Fusion 3 attn
success, but the aspp convergent in epoch 5, this attn fusion only convergent after epoch 12
also, cost way too many calculation resources


exp3: delete CA:
without CA, the convergent speed almost same, 
but in the epoch 12:  
with CA val_dice2=11%
without CA val_dice2=8% 

exp4: replace Cross attn block
success

exp5: 1 layer ViT -> 8 layer ViT
success, convergent speed little slower than CNN

exp6: ViT -> SMAFormer
success, but attn fusion caused convergent speed slower than CNN

exp7(SMAFormerV2): replace all SE, ASPP to Embedded_modulator.Synergistic_attn(MSA+(PA@CA@SA))
but no tranposeConv(linear upsample+resConv)
success

exp8(SMAFormerV3): replace all (ResConv & linear sample) -> ConvTranspose to upsampling
success

'''
