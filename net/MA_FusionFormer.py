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
from net.init_weights import init_weights
from dataset import dataset
from torch.nn import Softmax
from axial_attention import AxialAttention
from einops import rearrange
import math

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


class Patch_Position_Embedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size, img_size):
        super(Patch_Position_Embedding, self).__init__()
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.img_size = img_size

        # Projection layer
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size+1, stride=patch_size, padding=patch_size //2)

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))

        # Calculate number of patches
        num_patches = (img_size // patch_size) ** 2

        # Positional embeddings for patches and class token
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, emb_size))

    def forward(self, x):
        b = x.shape[0]  # Get batch size

        # Project to emb_size and flatten the patches
        x = self.projection(x).flatten(2).transpose(1, 2)  # [B, C, H, W] -> [B, N, E]

        # Expand class token for the whole batch
        cls_tokens = self.cls_token.expand(b, -1, -1)
        # print(cls_tokens.shape)
        # print(x.shape)
        # exit()
        # Concatenate the class token with the patch embeddings
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, E]

        # Add position embeddings
        x += self.position_embeddings  # [B, N+1, E] += [1, N+1, E]
        print(x.shape)
        return x


class PA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))

class PatchEmbed(nn.Module):
    def __init__(self, in_ch, out_ch, patch_size, with_pos=True):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=patch_size + 1, stride=patch_size, padding=patch_size // 2)

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

        self.norm = nn.LayerNorm(out_ch, eps=1e-6)

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.proj(x)
        if self.with_pos:
            x = self.pos(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        # H, W = H // self.patch_size, W // self.patch_size
        return x
'''
Cross Attention Block
Channel attention + Spatial Attention
Cross Channel Spatial Attention(CCS attention)
Written by FuChen Zheng(YC37950,UofMacau && Chinese Academy of Sciences)
'''
# Licensed under the Apache License 2.0 [see LICENSE for details]
# Written by FuChen Zheng(YC37950,UofMacau && Chinese Academy of Sciences)
# class CCS_attention(nn.Module):
#     def __init__(self, feature_size, reduction=16):
#         super(CCS_attention, self).__init__()
#         self.position_embedding = PatchEmbedding(
#             in_channels=
#         )
#
#         self.spatial_attention = AxialAttention(
#             dim=feature_size[1],
#             dim_index=1,
#
#         )
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#         x1 = self.position_embedding(x).view(h, c)
#         print(x1.shape)
#         exit()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)


class Channel_Attention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Channel_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Spatial_Attention(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):
        super(Spatial_Attention, self).__init__()

        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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


# Transformer编码器块
class TransformerEncoder(nn.Module):
    def __init__(self, feature_size, heads, dropout, forward_expansion):
        super(TransformerEncoder, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feature_size, num_heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(feature_size)
        self.norm2 = nn.LayerNorm(feature_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(feature_size, forward_expansion * feature_size),
            nn.GELU(),
            nn.Linear(forward_expansion * feature_size, feature_size)
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, value, key, query):
        query = self.norm1(query)
        attention = self.attention(query, key, value)[0]
        x = self.dropout(attention) + query
        # use layer norm before resiudal connection
        # Licensed under the Apache License 2.0 [see LICENSE for details]
        # Written by FuChen Zheng
        x = self.norm2(x)
        forward = self.feed_forward(x)
        out = self.dropout(forward) + x
        return out


# Creating the full Transformer Encoder
class TransformerEncoderBlock(nn.Module):
    def __init__(self, feature_size, heads, dropout, forward_expansion, num_layers):
        super(TransformerEncoderBlock, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoder(feature_size, heads, dropout, forward_expansion) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x, x, x)  # 在自注意力中 key, query 和 value 都是相同的输入

        return x

class TransformerDecoder(nn.Module):
    def __init__(self, feature_size, heads, dropout, forward_expansion):
        super(TransformerDecoder, self).__init__()
        self.attention = nn.MultiheadAttention(feature_size, heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(feature_size)
        self.norm2 = nn.LayerNorm(feature_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(feature_size, forward_expansion * feature_size),
            nn.GELU(),
            nn.Linear(forward_expansion * feature_size, feature_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, skip_connection):
        attention = self.attention(query, key, value)[0]
        query = self.dropout(self.norm1(attention + skip_connection))
        forward = self.feed_forward(query)
        out = self.dropout(self.norm2(forward + query))
        return out


# Creating the full Transformer Decoder
class TransformerDecoderBlock(nn.Module):
    def __init__(self, feature_size, heads, dropout, forward_expansion, num_layers):
        super(TransformerDecoderBlock, self).__init__()
        self.layers = nn.ModuleList([
            TransformerDecoder(feature_size, heads, dropout, forward_expansion) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out):
        for layer in self.layers:
            x = layer(enc_out, enc_out, x, x)

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
        # print(x1.shape)
        # print(x2.shape)
        # print(self.conv_encoder(x1).shape)
        # print(self.conv_decoder(x2).shape)
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2

class ResUformer(nn.Module):
    def __init__(self, args):
        super(ResUformer, self).__init__()
        self.args = args
        in_channels = 3
        n_classes = dataset.num_classes
        patch_size = 2
        filters = [16, 32, 64, 128, 256, 512, 1024]
        encoder_layer = 5
        decoder_layer = 5
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
        self.patch_position_embedding1 = Patch_Position_Embedding(
            in_channels=filters[0], patch_size=patch_size,
            emb_size=filters[0], img_size=512
        )

        self.patch_embedding1 = PatchEmbed(in_ch=filters[0], out_ch=filters[1], patch_size=patch_size)

        self.squeeze_excite1 = Channel_Attention(filters[1])

        self.residual_conv1 = ResidualConv(filters[1], filters[2], 2, 1)

        self.patch_embedding2 = PatchEmbed(in_ch=filters[2], out_ch=filters[3], patch_size=patch_size)

        self.squeeze_excite2 = Channel_Attention(filters[3])

        self.residual_conv2 = ResidualConv(filters[3], filters[4], 2, 1)

        self.patch_embedding3 = PatchEmbed(in_ch=filters[4], out_ch=filters[5], patch_size=patch_size)

        self.squeeze_excite3 = Channel_Attention(filters[5])

        self.residual_conv3 = ResidualConv(filters[5], filters[6], 1, 1)

        self.Encoder = TransformerEncoder(feature_size=filters[6], heads=8, dropout=0.,
                                                      forward_expansion=4)
        self.EncoderBlock = TransformerEncoderBlock(feature_size=filters[6], heads=8, dropout=0., forward_expansion=4,
                                                    num_layers=encoder_layer)

        self.FeatureFusion_bridge = Spatial_Attention(filters[6], filters[6])

        self.Decoder = TransformerDecoder(feature_size=filters[6], heads=8, dropout=0.0, forward_expansion=4)

        self.DecoderBlock = TransformerDecoderBlock(feature_size=filters[6], heads=8, dropout=0., forward_expansion=4,
                                                    num_layers=decoder_layer)


        self.attn1 = Cross_AttentionBlock(filters[6], filters[6], filters[6])
        self.upsample1 = Upsample_(2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[6], filters[5], 1, 1)

        self.attn2 = Cross_AttentionBlock(filters[4], filters[5], filters[5])
        self.upsample2 = Upsample_(4)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[5], filters[3], 1, 1)

        self.attn3 = Cross_AttentionBlock(filters[2], filters[3], filters[3])
        self.upsample3 = Upsample_(4)
        self.up_residual_conv3 = ResidualConv(filters[3] + filters[0], filters[1], 1, 1)

        self.FeatureFusion_out = Spatial_Attention(filters[1], filters[0])

        self.output_layer = nn.Sequential(nn.Conv2d(filters[0], n_classes, 1))

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x) #[16,3,512x2]->[16,16,512x2]
        # x2 = self.patch_position_embedding1(x1)

        x2 = self.patch_embedding1(x1) #[16,16,512,512]->[16,256*256,32]
        b, num_patch, c = x2.size()
        x2 = x2.view(b, c, num_patch//self.filters[4], num_patch//self.filters[4])  # 调整回卷积形状 [16,256*256,32]->[16,32,256,256]
        x2_skip = self.squeeze_excite1(x2) #same
        x2 = self.residual_conv1(x2_skip) #[16,32,256,256]->[16,64,128,128]

        x3 = self.patch_embedding2(x2) #[16,64,128,128]->[16,64*64,128]
        b, num_patch, c = x3.size()
        x3 = x3.view(b, c, num_patch //self.filters[2], num_patch //self.filters[2]) #调整回卷积形状[16,64*64,128]->[16,128,64,64]
        x3_skip = self.squeeze_excite2(x3)
        x3 = self.residual_conv2(x3_skip) #[16,128,64,64]->[16,256,32,32]

        x4 = self.patch_embedding3(x3) #[16,256,32,32]->[16,16*16,512]
        b, num_patch, c = x4.size()
        x4 = x4.view(b, c, num_patch//self.filters[0], num_patch//self.filters[0]) #[16,16*16,512]->[16,512,16,16]
        x4_skip = self.squeeze_excite3(x4)
        x4 = self.residual_conv3(x4_skip) #[16,512,16,16]->[16,1024,16,16]

        #Total Transformer Block
        b, c, h, w = x4.size()
        x5 = x4.view(b, c, h * w).permute(0, 2, 1)  # 调整形状以适应Transformer模块
        # x5 = self.Encoder(x5, x5, x5)  # [16, 16*16(h*w), c=1024]
        encoder_out = self.EncoderBlock(x5) # [16, 16*16(h*w), c=1024]
        x5 = encoder_out.permute(0, 2, 1).view(b, c, h, w)  #调整回卷积形状 #[16, 16*16(h*w), c=1024]->[16,1024,16,16]
        x_bridge = self.FeatureFusion_bridge(x5) #same
        b, c, h, w = x_bridge.size()
        x_bridge = x_bridge.view(b, c, h * w).permute(0, 2, 1)
        x5 = self.DecoderBlock(x_bridge, encoder_out)  #input:[x, enc_out], shape:[16, 16*16, 1024]
        x5 = x5.permute(0, 2, 1).view(b, c, h, w)  #调整回卷积形状 #[16, 16*16(h*w), c=1024]->[16,1024,16,16]

        x6 = self.attn1(x4, x5) #same
        x6 = self.upsample1(x6) #[16,1024,16,16]->[16,1024,32,32]
        x6 = torch.cat([x6, x3], dim=1) #[16,1024,32,32]+[16,256,32,32]->[16,1280,32,32]
        x6 = self.up_residual_conv1(x6) #[16,1280,32,32]->[16, 512, 32, 32]

        x7 = self.attn2(x3, x6) #[16,256,32,32][16,512,32,32]->[16,512,32,32]
        x7 = self.upsample2(x7) #[16, 512, 32, 32]->[16, 512, 128, 128]
        x7 = torch.cat([x7, x2], dim=1) #[16, 512, 128, 128]+[16,64,128,128]->[16,64+512,128,128]
        x7 = self.up_residual_conv2(x7) #[16,64+512,128,128]->[16,128,128,128]

        x8 = self.attn3(x2, x7) #[16,64,128,128],[16,128,128,128]->[16,128,128,128]
        x8 = self.upsample3(x8) #[16,128,128,128]->[16,128,512,512]
        x8 = torch.cat([x8, x1], dim=1) #[16,128,512,512]+[16,16,512x2]->[16,128+16,512,512]
        x8 = self.up_residual_conv3(x8) #[16,128+16,512,512]->[16,32,512,512]

        x9 = self.FeatureFusion_out(x8) #[16,32,512,512]->[16,16,512,512]
        out = self.output_layer(x9) #[16,16,512,512]->[16,num_classes,512,512]

        return out


#
# # 检查模型是否能够创建并输出期望的维度
# args = parse_args()
# model = ResUformer(args)
# x = torch.randn(16, 3, 512, 512)  # 假设输入是256x256的RGB图像
# with torch.no_grad():  # 在不计算梯度的情况下执行前向传播
#     out = model(x)
# print(out.shape)  # 输出预期是与分类头的输出通道数匹配的特征图
