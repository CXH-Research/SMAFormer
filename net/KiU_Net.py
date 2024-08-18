import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

import torch
import torch.nn as nn
import torch.nn.functional as F

class kiunet_org(nn.Module):

    def __init__(self,args):
        super(kiunet_org, self).__init__()
        self.args = args
        in_ch = 3
        out_ch = 2
        self.start = nn.Conv2d(in_ch, 3, 3, stride=2, padding=1)

        self.encoder1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.en1_bn = nn.BatchNorm2d(16)
        self.encoder2= nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.en2_bn = nn.BatchNorm2d(32)
        self.encoder3= nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.en3_bn = nn.BatchNorm2d(64)

        self.decoder1 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.de1_bn = nn.BatchNorm2d(32)
        self.decoder2 = nn.Conv2d(32,16, 3, stride=1, padding=1)
        self.de2_bn = nn.BatchNorm2d(16)
        self.decoder3 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.de3_bn = nn.BatchNorm2d(8)

        self.decoderf1 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.def1_bn = nn.BatchNorm2d(32)
        self.decoderf2= nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.def2_bn = nn.BatchNorm2d(16)
        self.decoderf3 =   nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.def3_bn = nn.BatchNorm2d(8)

        self.encoderf1 =   nn.Conv2d(1, 16, 3, stride=1, padding=1)  # First Layer GrayScale Image , change to input channels to 3 in case of RGB
        self.enf1_bn = nn.BatchNorm2d(16)
        self.encoderf2=   nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.enf2_bn = nn.BatchNorm2d(32)
        self.encoderf3 =   nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.enf3_bn = nn.BatchNorm2d(64)

        self.intere1_1 = nn.Conv2d(16,16,3, stride=1, padding=1)
        self.inte1_1bn = nn.BatchNorm2d(16)
        self.intere2_1 = nn.Conv2d(32,32,3, stride=1, padding=1)
        self.inte2_1bn = nn.BatchNorm2d(32)
        self.intere3_1 = nn.Conv2d(64,64,3, stride=1, padding=1)
        self.inte3_1bn = nn.BatchNorm2d(64)

        self.intere1_2 = nn.Conv2d(16,16,3, stride=1, padding=1)
        self.inte1_2bn = nn.BatchNorm2d(16)
        self.intere2_2 = nn.Conv2d(32,32,3, stride=1, padding=1)
        self.inte2_2bn = nn.BatchNorm2d(32)
        self.intere3_2 = nn.Conv2d(64,64,3, stride=1, padding=1)
        self.inte3_2bn = nn.BatchNorm2d(64)

        self.interd1_1 = nn.Conv2d(32,32,3, stride=1, padding=1)
        self.intd1_1bn = nn.BatchNorm2d(32)
        self.interd2_1 = nn.Conv2d(16,16,3, stride=1, padding=1)
        self.intd2_1bn = nn.BatchNorm2d(16)
        self.interd3_1 = nn.Conv2d(64,64,3, stride=1, padding=1)
        self.intd3_1bn = nn.BatchNorm2d(64)

        self.interd1_2 = nn.Conv2d(32,32,3, stride=1, padding=1)
        self.intd1_2bn = nn.BatchNorm2d(32)
        self.interd2_2 = nn.Conv2d(16,16,3, stride=1, padding=1)
        self.intd2_2bn = nn.BatchNorm2d(16)
        self.interd3_2 = nn.Conv2d(64,64,3, stride=1, padding=1)
        self.intd3_2bn = nn.BatchNorm2d(64)

        # self.start = nn.Conv3d(1, 1, 3, stride=1, padding=1)
        self.final = nn.Conv2d(8,out_ch,1,stride=1,padding=0)
        self.fin = nn.Conv2d(out_ch,out_ch,1,stride=1,padding=0)

        self.map4 = nn.Sequential(
            nn.Conv2d(32, 1, 1, 1),
            nn.Upsample(scale_factor=(4, 16, 16), mode='trilinear'),
            nn.Sigmoid()
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(16, 1, 1, 1),
            nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear'),
            nn.Sigmoid()
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(8, 1, 1, 1),
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear'),
            nn.Sigmoid()
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, 1, 1, 1),
            nn.Upsample(scale_factor=(16, 32, 32), mode='trilinear'),
            nn.Sigmoid()
        )

        self.soft = nn.Softmax(dim =1)

    def forward(self, x):
        # print(x.shape)
        outx = self.start(x)
        # print(outx.shape)
        out = F.relu(self.en1_bn(F.max_pool3d(self.encoder1(outx),2,2)))  #U-Net branch
        out1 = F.relu(self.enf1_bn(F.interpolate(self.encoderf1(outx),scale_factor=(0.5,1,1),mode ='trilinear'))) #Ki-Net branch
        tmp = out
        # print(out.shape,out1.shape)
        out = torch.add(out,F.interpolate(F.relu(self.inte1_1bn(self.intere1_1(out1))),scale_factor=(1,0.5,0.5),mode ='trilinear')) #CRFB
        out1 = torch.add(out1,F.interpolate(F.relu(self.inte1_2bn(self.intere1_2(tmp))),scale_factor=(1,2,2),mode ='trilinear')) #CRFB

        u1 = out  #skip conn
        o1 = out1  #skip conn

        out = F.relu(self.en2_bn(F.max_pool3d(self.encoder2(out),2,2)))
        out1 = F.relu(self.enf2_bn(F.interpolate(self.encoderf2(out1),scale_factor=(1,1,1),mode ='trilinear')))
        tmp = out
        # print(out.shape,out1.shape)
        out = torch.add(out,F.interpolate(F.relu(self.inte2_1bn(self.intere2_1(out1))),scale_factor=(0.5,0.25,0.25),mode ='trilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.inte2_2bn(self.intere2_2(tmp))),scale_factor=(2,4,4),mode ='trilinear'))

        u2 = out
        o2 = out1
        out = F.pad(out,[0,0,0,0,0,1])
        # print(out.shape)
        out = F.relu(self.en3_bn(F.max_pool3d(self.encoder3(out),2,2)))
        out1 = F.relu(self.enf3_bn(F.interpolate(self.encoderf3(out1),scale_factor=(1,2,2),mode ='trilinear')))
        # print(out.shape,out1.shape)
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.inte3_1bn(self.intere3_1(out1))),scale_factor=(0.5,0.0625,0.0625),mode ='trilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.inte3_2bn(self.intere3_2(tmp))),scale_factor=(2,16,16),mode ='trilinear'))

        ### End of encoder block

        ### Start Decoder

        out = F.relu(self.de1_bn(F.interpolate(self.decoder1(out),scale_factor=(2,2,2),mode ='trilinear')))  #U-NET
        out1 = F.relu(self.def1_bn(F.max_pool3d(self.decoderf1(out1),2,2))) #Ki-NET
        tmp = out
        # print(out.shape,out1.shape)
        out = torch.add(out,F.interpolate(F.relu(self.intd1_1bn(self.interd1_1(out1))),scale_factor=(2,0.25,0.25),mode ='trilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.intd1_2bn(self.interd1_2(tmp))),scale_factor=(0.5,4,4),mode ='trilinear'))
        # print(out.shape)
        output1 = self.map4(out)
        out = torch.add(out,u2)  #skip conn
        out1 = torch.add(out1,o2)  #skip conn

        out = F.relu(self.de2_bn(F.interpolate(self.decoder2(out),scale_factor=(1,2,2),mode ='trilinear')))
        out1 = F.relu(self.def2_bn(F.max_pool3d(self.decoderf2(out1),1,1)))
        # print(out.shape,out1.shape)
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.intd2_1bn(self.interd2_1(out1))),scale_factor=(1,0.5,0.5),mode ='trilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.intd2_2bn(self.interd2_2(tmp))),scale_factor=(1,2,2),mode ='trilinear'))
        output2 = self.map3(out)
        # print(out1.shape,o1.shape)
        out = torch.add(out,u1)
        out1 = torch.add(out1,o1)

        out = F.relu(self.de3_bn(F.interpolate(self.decoder3(out),scale_factor=(1,2,2),mode ='trilinear')))
        out1 = F.relu(self.def3_bn(F.max_pool3d(self.decoderf3(out1),1,1)))
        # print(out.shape,out1.shape)
        output3 = self.map2(out)


        out = torch.add(out,out1) # fusion of both branches

        out = F.relu(self.final(out))  #1*1 conv

        output4 = F.interpolate(self.fin(out),scale_factor=(4,4,4),mode ='trilinear')
        # print(out.shape)
        # out = self.soft(out)
        # print(output1.shape,output2.shape,output3.shape,output4.shape)
        if self.training is True:
            return output1, output2, output3, output4
        else:
            return output4


import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class RecurrentBlock(nn.Module):
    """
    Recurrent Block
    """

    def __init__(self, ch):
        super(RecurrentBlock, self).__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)
        out = x * y + x
        return out


class AttentionBlock(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        out = x * psi

        return out

#
# class KiU_Net(nn.Module):
#     """
#     KiU-Net Model
#     """
#
#     def __init__(self, args):
#         super(KiU_Net, self).__init__()
#         self.args = args
#         in_channels = 3
#         out_channels = 2
#         # Encoder
#         self.conv1 = ConvBlock(in_channels, 32)
#         self.pool1 = nn.MaxPool2d(kernel_size=2)
#
#         self.conv2 = ConvBlock(32, 64)
#         self.pool2 = nn.MaxPool2d(kernel_size=2)
#
#         self.conv3 = ConvBlock(64, 128)
#         self.pool3 = nn.MaxPool2d(kernel_size=2)
#
#         self.conv4 = ConvBlock(128, 256)
#         self.pool4 = nn.MaxPool2d(kernel_size=2)
#
#         self.conv5 = ConvBlock(256, 512)
#         self.pool5 = nn.MaxPool2d(kernel_size=2)
#
#         # Recurrent Blocks
#         self.recursive_block1 = RecurrentBlock(32)
#         self.recursive_block2 = RecurrentBlock(64)
#         self.recursive_block3 = RecurrentBlock(128)
#         self.recursive_block4 = RecurrentBlock(256)
#         self.recursive_block5 = RecurrentBlock(512)
#
#
#         # Up Convolution
#         self.upconv5 = UpConv(512, 256)
#         self.attention_block5 = AttentionBlock(F_g=256, F_l=256, F_int=128)
#
#         self.upconv4 = UpConv(256 + 128*2, 128)
#         self.attention_block4 = AttentionBlock(F_g=128, F_l=128, F_int=64)
#
#         self.upconv3 = UpConv(128 + 64*2, 64)
#         self.attention_block3 = AttentionBlock(F_g=64, F_l=64, F_int=32)
#
#         self.upconv2 = UpConv(64 + 32*2, 32)
#         self.attention_block2 = AttentionBlock(F_g=32, F_l=32, F_int=16)
#
#         self.upconv1 = UpConv(32 + in_channels, 16)
#         self.attention_block1 = AttentionBlock(F_g=in_channels, F_l=16, F_int=8)
#
#         # Output
#         self.outconv = nn.Conv2d(16, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
#
#     def forward(self, x):
#         # Down Convolution
#         conv1 = self.conv1(x)
#         pool1 = self.pool1(conv1)
#
#         conv2 = self.conv2(pool1)
#         pool2 = self.pool2(conv2)
#
#         conv3 = self.conv3(pool2)
#         pool3 = self.pool3(conv3)
#
#         conv4 = self.conv4(pool3)
#         pool4 = self.pool4(conv4)
#
#         conv5 = self.conv5(pool4)
#         pool5 = self.pool5(conv5)
#
#         # Recurrent Blocks
#         recursive_block1_out = self.recursive_block1(pool1)
#         recursive_block2_out = self.recursive_block2(pool2)
#         recursive_block3_out = self.recursive_block3(pool3)
#         recursive_block4_out = self.recursive_block3(pool4)
#         recursive_block5_out = self.recursive_block3(pool5)
#
#         # Up Convolution
#         cat5 = torch.cat([recursive_block5_out, self.upconv5(recursive_block5_out)], dim=1)
#         cat5_attention_out = self.attention_block5(cat5, conv4)
#
#         cat4 = torch.cat([recursive_block4_out, self.upconv4(cat5_attention_out)], dim=1)
#         cat4_attention_out = self.attention_block4(cat4, conv3)
#
#         cat3 = torch.cat([recursive_block3_out, self.upconv3(cat4_attention_out)], dim=1)
#         cat3_attention_out = self.attention_block3(cat3, self.conv2(conv2))
#
#         cat2 = torch.cat([recursive_block2_out, self.upconv2(cat3_attention_out)], dim=1)
#         cat2_attention_out = self.attention_block2(cat2, self.conv1(conv1))
#
#         cat1 = torch.cat([recursive_block1_out, self.upconv1(cat2_attention_out)], dim=1)
#         cat1_attention_out = self.attention_block1(cat1, x)
#
#         # Output
#         out = self.outconv(cat1_attention_out)
#
#         return out