import math

import torch
from torch import nn
from collections import OrderedDict
from torch.utils.checkpoint import checkpoint


# class UNet(nn.Module): # Max Pooling, BatchNorm & LeakyReLU on demand
#     """
#     The layout of the model has been adapted from https://github.com/mateuszbuda/brain-segmentation-pytorch,
#     using 3-dimensional kernels and other modifications.
#
#
#     MIT License
#
# Copyright (c) 2019 mateuszbuda
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE."""
#     def __init__(self, in_channels, out_channels, features, batchnorm=True, leaky=True, max_pool=True, use_medical_data = False, split_med_channels = False):
#         '''
#         :param in_channels: modalities
#         :param out_channels: output modalitites
#         :param features: size of input (each modality)
#         :param batchnorm: bool
#         :param leaky: bool
#         :param max_pool: bool
#         '''
#         super(UNet, self).__init__()
#         self.batchnorm = batchnorm
#         self.leaky = leaky
#         self.in_channels = in_channels
#         self.use_medical_data = use_medical_data
#         self.split_med_channels = split_med_channels
#         self.seperat_mod_channels = in_channels # channels other than med data
#         self.true_in_channels = in_channels
#         print("channels", self.seperat_mod_channels, self.true_in_channels)
#
#         #medical data
#         if use_medical_data and not split_med_channels:
#             self.encoder1_md = self._block(5, features, name="enc11_md")
#             self.pool1_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#             self.encoder2_md = self._block(features, features * 2, name="enc21_md")
#             self.pool2_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#             self.encoder3_md = self._block(features * 2, features * 4, name="enc31_md")
#             self.pool3_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#             self.encoder4_md = self._block(features * 4, features * 8, name="enc41_md")
#             self.pool4_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#             self.seperat_mod_channels = self.seperat_mod_channels - 5
#             self.true_in_channels = in_channels - 4
#             print("channels", self.seperat_mod_channels, self.true_in_channels)
#
#         if use_medical_data and split_med_channels:
#             self.encoder11_md = self._block(1, features, name="enc11_md")
#             self.pool11_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2,
#                                                                                                  stride=2)
#             self.encoder21_md = self._block(features, features * 2, name="enc21_md")
#             self.pool21_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2,
#                                                                                                  stride=2)
#             self.encoder31_md = self._block(features * 2, features * 4, name="enc31_md")
#             self.pool31_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2,
#                                                                                                  stride=2)
#             self.encoder41_md = self._block(features * 4, features * 8, name="enc41_md")
#             self.pool41_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2,
#                                                                                                  stride=2)
#
#
#
#             self.encoder12_md = self._block(1, features, name="enc12_md")
#             self.pool12_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2,
#                                                                                                  stride=2)
#             self.encoder22_md = self._block(features, features * 2, name="enc22_md")
#             self.pool22_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2,
#                                                                                                  stride=2)
#             self.encoder32_md = self._block(features * 2, features * 4, name="enc32_md")
#             self.pool32_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2,
#                                                                                                  stride=2)
#             self.encoder42_md = self._block(features * 4, features * 8, name="enc42_md")
#             self.pool42_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2,
#                                                                                                 stride=2)
#
#
#
#             self.encoder13_md = self._block(1, features, name="enc13_md")
#             self.pool13_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2,
#                                                                                                  stride=2)
#             self.encoder23_md = self._block(features, features * 2, name="enc23_md")
#             self.pool23_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2,
#                                                                                                  stride=2)
#             self.encoder33_md = self._block(features * 2, features * 4, name="enc33_md")
#             self.pool33_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2,
#                                                                                                  stride=2)
#             self.encoder43_md = self._block(features * 4, features * 8, name="enc43_md")
#             self.pool43_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2,
#                                                                                                  stride=2)
#
#
#
#             self.encoder14_md = self._block(1, features, name="enc14_md")
#             self.pool14_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2,
#                                                                                                  stride=2)
#             self.encoder24_md = self._block(features, features * 2, name="enc24_md")
#             self.pool24_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2,
#                                                                                                  stride=2)
#             self.encoder34_md = self._block(features * 2, features * 4, name="enc34_md")
#             self.pool34_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2,
#                                                                                                  stride=2)
#             self.encoder44_md = self._block(features * 4, features * 8, name="enc44_md")
#             self.pool44_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2,
#                                                                                                  stride=2)
#
#
#
#             self.encoder15_md = self._block(1, features, name="enc15_md")
#             self.pool15_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2,
#                                                                                                  stride=2)
#             self.encoder25_md = self._block(features, features * 2, name="enc25_md")
#             self.pool25_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2,
#                                                                                                  stride=2)
#             self.encoder35_md = self._block(features * 2, features * 4, name="enc35_md")
#             self.pool35_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2,
#                                                                                                  stride=2)
#             self.encoder45_md = self._block(features * 4, features * 8, name="enc45_md")
#             self.pool45_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2,
#                                                                                                  stride=2)
#
#
#             self.seperat_mod_channels = self.seperat_mod_channels - 5
#             print("channels", self.seperat_mod_channels, self.true_in_channels)
#
#
#
#         self.encoder11 = self._block(1, features, name="enc11")
#         self.pool11 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#         self.encoder21 = self._block(features, features * 2, name="enc21")
#         self.pool21 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#         self.encoder31 = self._block(features * 2, features * 4, name="enc31")
#         self.pool31 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#         self.encoder41 = self._block(features * 4, features * 8, name="enc41")
#         self.pool41 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#
#         if self.seperat_mod_channels > 1 :
#             print("built channel2")
#             self.encoder12 = self._block(1, features, name="enc12")
#             self.pool12 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#             self.encoder22 = self._block(features, features * 2, name="enc22")
#             self.pool22 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#             self.encoder32 = self._block(features * 2, features * 4, name="enc32")
#             self.pool32 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#             self.encoder42 = self._block(features * 4, features * 8, name="enc42")
#             self.pool42 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#         if self.seperat_mod_channels > 2:
#             print("built channel3")
#             self.encoder13 = self._block(1, features, name="enc13")
#             self.pool13 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#             self.encoder23 = self._block(features, features * 2, name="enc23")
#             self.pool23 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#             self.encoder33 = self._block(features * 2, features * 4, name="enc33")
#             self.pool33 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#             self.encoder43 = self._block(features * 4, features * 8, name="enc43")
#             self.pool43 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#         if self.seperat_mod_channels > 3:
#             print("built channel4")
#             self.encoder14 = self._block(1, features, name="enc14")
#             self.pool14 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#             self.encoder24 = self._block(features, features * 2, name="enc24")
#             self.pool24 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#             self.encoder34 = self._block(features * 2, features * 4, name="enc34")
#             self.pool34 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#             self.encoder44 = self._block(features * 4, features * 8, name="enc44")
#             self.pool44 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#         if self.seperat_mod_channels > 4:
#             print("built channel5")
#             self.encoder15 = self._block(1, features, name="enc15")
#             self.pool15 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#             self.encoder25 = self._block(features, features * 2, name="enc25")
#             self.pool25 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#             self.encoder35 = self._block(features * 2, features * 4, name="enc35")
#             self.pool35 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#             self.encoder45 = self._block(features * 4, features * 8, name="enc45")
#             self.pool45 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#
#         print("features", features)
#         print("pathways for images", self.seperat_mod_channels," / totals pathways" , self.true_in_channels)
#         self.bottleneck = self._block(features * 8 * self.true_in_channels, features * 16, name="bottleneck")
#
#         self.upconv4 = nn.ConvTranspose3d(
#             features * 16, features * 8, kernel_size=2, stride=2
#         )
#         self.decoder4 = self._block((features * 8) * (self.true_in_channels+1), features * 8, name="dec4")
#         self.upconv3 = nn.ConvTranspose3d(
#             features * 8, features * 4, kernel_size=2, stride=2
#         )
#         self.decoder3 = self._block((features * 4) * (self.true_in_channels+1), features * 4, name="dec3")
#         self.upconv2 = nn.ConvTranspose3d(
#             features * 4, features * 2, kernel_size=2, stride=2
#         )
#         self.decoder2 = self._block((features * 2) * (self.true_in_channels+1), features * 2, name="dec2")
#         self.upconv1 = nn.ConvTranspose3d(
#             features * 2, features, kernel_size=2, stride=2
#         )
#         self.decoder1 = self._block(features * (self.true_in_channels+1), features, name="dec1")
#
#         self.conv = nn.Conv3d(
#             in_channels=features, out_channels=out_channels, kernel_size=1
#         )
#
#     def forward(self, x):
#
#         mod_list = []
#         med_data = None
#
#         for batch_item in x:# run thru all items in the batch
#             #[b1, b2, b3, b4, b5] = batch
#
#             if self.use_medical_data and not self.split_med_channels:
#                 if len(mod_list) == 0:  # mod list empty
#                     for j in range(self.true_in_channels):
#                         if j == 0: #get medical data
#                             xj = torch.unsqueeze(batch_item[0:5], 0)#get first item in batch and get med modalities
#                             med_data = xj
#                         else: #get other modalities
#                             xj = torch.unsqueeze(batch_item[j+4], 0)
#                             mod_list.append(xj)  # append split modalities to mod_list
#                 else:
#                     for j in range(self.true_in_channels):
#                         if j == 0: #get medical data
#                             xj = torch.unsqueeze(batch_item[0:5], 0)#get first item in batch and get med modalities
#                             med_data = torch.cat((med_data, xj), 0)
#                         else: #get other modalities
#                             xj = torch.unsqueeze(batch_item[j+4], 0)
#                             item = mod_list[j-1]  # get all other modalities in the batch of same type in mod_list
#                             mod_list[j-1] = torch.cat((item, xj), 0)  # append the new one
#
#
#             else: # no med data
#                 if len(mod_list) == 0: # mod list empty
#                     for j in range(self.true_in_channels):
#                         xj = torch.unsqueeze(batch_item[j], 0)#get first item in batch and split it by modality
#                         mod_list.append(xj)#append split modalities to mod_list
#                 else:
#                     for j in range(self.true_in_channels):# mod list not empty
#                         xj = torch.unsqueeze(batch_item[j], 0) #get item in batch and split it by modality
#                         item = mod_list[j] #get all other modalities in the batch of same type in mod_list
#                         mod_list[j] = torch.cat((item, xj), 0)# append the new one
#
#
#                 # x1 = torch.cat((x1, torch.unsqueeze(b1, 0)), 0)
#                 # x2 = torch.cat((x2, torch.unsqueeze(b2, 0)), 0)
#                 # x3 = torch.cat((x3, torch.unsqueeze(b3, 0)), 0)
#                 # x4 = torch.cat((x4, torch.unsqueeze(b4, 0)), 0)
#                 # x5 = torch.cat((x5, torch.unsqueeze(b5, 0)), 0)
#
#         if self.use_medical_data and not self.split_med_channels:
#             enc1_md = self.encoder1_md(med_data)
#             enc2_md = self.encoder2_md(self.pool12(enc1_md))
#             enc3_md = self.encoder3_md(self.pool22(enc2_md))
#             enc4_md = self.encoder4_md(self.pool32(enc3_md))
#             pool4_md = self.pool4_md(enc4_md)
#
#         if self.use_medical_data and self.split_med_channels:
#             enc11_md = self.encoder11_md(torch.unsqueeze(mod_list[0], 1))
#             enc21_md = self.encoder22_md(self.pool11_md(enc11_md))
#             enc31_md = self.encoder32_md(self.pool21_md(enc21_md))
#             enc41_md = self.encoder42_md(self.pool31_md(enc31_md))
#             pool41_md = self.pool41_md(enc41_md)
#
#             enc12_md = self.encoder12_md(torch.unsqueeze(mod_list[1], 1))
#             enc22_md = self.encoder22_md(self.pool12_md(enc12_md))
#             enc32_md = self.encoder32_md(self.pool22_md(enc22_md))
#             enc42_md = self.encoder42_md(self.pool32_md(enc32_md))
#             pool42_md = self.pool42_md(enc42_md)
#
#             enc13_md = self.encoder13_md(torch.unsqueeze(mod_list[2], 1))
#             enc23_md = self.encoder23_md(self.pool13_md(enc13_md))
#             enc33_md = self.encoder33_md(self.pool23_md(enc23_md))
#             enc43_md = self.encoder43_md(self.pool33_md(enc33_md))
#             pool43_md = self.pool43_md(enc43_md)
#
#             enc14_md = self.encoder14_md(torch.unsqueeze(mod_list[3], 1))
#             enc24_md = self.encoder24_md(self.pool14_md(enc14_md))
#             enc34_md = self.encoder34_md(self.pool24_md(enc24_md))
#             enc44_md = self.encoder44_md(self.pool34_md(enc34_md))
#             pool44_md = self.pool44_md(enc44_md)
#
#             enc15_md = self.encoder15_md(torch.unsqueeze(mod_list[4], 1))
#             enc25_md = self.encoder25_md(self.pool15_md(enc15_md))
#             enc35_md = self.encoder35_md(self.pool25_md(enc25_md))
#             enc45_md = self.encoder45_md(self.pool35_md(enc35_md))
#             pool45_md = self.pool45_md(enc45_md)
#
#             enc1_md = torch.cat((enc11_md, enc12_md, enc13_md, enc14_md, enc15_md), dim=1)
#             enc2_md = torch.cat((enc21_md, enc22_md, enc23_md, enc24_md, enc25_md), dim=1)
#             enc3_md = torch.cat((enc31_md, enc32_md, enc33_md, enc34_md, enc35_md), dim=1)
#             enc4_md = torch.cat((enc41_md, enc42_md, enc43_md, enc44_md, enc45_md), dim=1)
#             pool4_md = torch.cat((pool41_md, pool42_md, pool43_md, pool44_md, pool45_md), dim=1)
#
#
#
#         inp = torch.unsqueeze(mod_list[-5],1)
#         enc11 = self.encoder11(inp)
#         enc21 = self.encoder21(self.pool11(enc11))
#         enc31 = self.encoder31(self.pool21(enc21))
#         enc41 = self.encoder41(self.pool31(enc31))
#         pool41 = self.pool41(enc41)
#
#         if self.use_medical_data:
#             bottle_input = torch.cat((pool4_md, pool41), dim=1)
#             dec4_input = torch.cat((enc4_md, enc41), dim=1)
#             dec3_input = torch.cat((enc3_md, enc31), dim=1)
#             dec2_input = torch.cat((enc2_md, enc21), dim=1)
#             dec1_input = torch.cat((enc1_md, enc11), dim=1)
#         else:
#             bottle_input = pool41
#             dec4_input = enc41
#             dec3_input = enc31
#             dec2_input = enc21
#             dec1_input = enc11
#
#         if self.seperat_mod_channels > 1:
#             enc12 = self.encoder12(torch.unsqueeze(mod_list[-4],1))
#             dec1_input = torch.cat((dec1_input, enc12), dim=1)
#             enc22 = self.encoder22(self.pool12(enc12))
#             dec2_input = torch.cat((dec2_input, enc22), dim=1)
#             enc32 = self.encoder32(self.pool22(enc22))
#             dec3_input = torch.cat((dec3_input, enc32), dim=1)
#             enc42 = self.encoder42(self.pool32(enc32))
#             dec4_input = torch.cat((dec4_input, enc42), dim=1)
#             pool42 = self.pool42(enc42)
#             bottle_input = torch.cat((bottle_input, pool42), dim=1)
#         if self.seperat_mod_channels > 2:
#             enc13 = self.encoder13(torch.unsqueeze(mod_list[-3],1))
#             dec1_input = torch.cat((dec1_input, enc13), dim=1)
#             enc23 = self.encoder23(self.pool13(enc13))
#             dec2_input = torch.cat((dec2_input, enc23), dim=1)
#             enc33 = self.encoder33(self.pool23(enc23))
#             dec3_input = torch.cat((dec3_input, enc33), dim=1)
#             enc43 = self.encoder43(self.pool33(enc33))
#             dec4_input = torch.cat((dec4_input, enc43), dim=1)
#             pool43 = self.pool43(enc43)
#             bottle_input = torch.cat((bottle_input, pool43), dim=1)
#         if self.seperat_mod_channels > 3:
#             enc14 = self.encoder14(torch.unsqueeze(mod_list[-2],1))
#             dec1_input = torch.cat((dec1_input, enc14), dim=1)
#             enc24 = self.encoder24(self.pool14(enc14))
#             dec2_input = torch.cat((dec2_input, enc24), dim=1)
#             enc34 = self.encoder34(self.pool24(enc24))
#             dec3_input = torch.cat((dec3_input, enc34), dim=1)
#             enc44 = self.encoder44(self.pool34(enc34))
#             dec4_input = torch.cat((dec4_input, enc44), dim=1)
#             pool44 = self.pool44(enc44)
#             bottle_input = torch.cat((bottle_input, pool44), dim=1)
#         if self.seperat_mod_channels > 4:
#             enc15 = self.encoder15(torch.unsqueeze(mod_list[-1],1))
#             dec1_input = torch.cat((dec1_input, enc15), dim=1)
#             enc25 = self.encoder25(self.pool15(enc15))
#             dec2_input = torch.cat((dec2_input, enc25), dim=1)
#             enc35 = self.encoder35(self.pool25(enc25))
#             dec3_input = torch.cat((dec3_input, enc35), dim=1)
#             enc45 = self.encoder45(self.pool35(enc35))
#             dec4_input = torch.cat((dec4_input, enc45), dim=1)
#             pool45 = self.pool45(enc45)
#             bottle_input = torch.cat((bottle_input, pool45), dim=1)
#
#
#
#         bottleneck = self.bottleneck(bottle_input)
#
#         dec4 = self.upconv4(bottleneck)
#         dec4 = torch.cat((dec4, dec4_input), dim=1)
#         dec4 = self.decoder4(dec4)
#         dec3 = self.upconv3(dec4)
#         dec3 = torch.cat((dec3, dec3_input), dim=1)
#         dec3 = self.decoder3(dec3)
#         dec2 = self.upconv2(dec3)
#         dec2 = torch.cat((dec2, dec2_input), dim=1)
#         dec2 = self.decoder2(dec2)
#         dec1 = self.upconv1(dec2)
#         dec1 = torch.cat((dec1, dec1_input), dim=1)
#         dec1 = self.decoder1(dec1)
#         return torch.sigmoid(self.conv(dec1))
#
#     def _block(self, in_channels, features, name):
#         if self.batchnorm:
#             activation = nn.Sequential(
#                 nn.BatchNorm3d(num_features=features),
#                 nn.LeakyReLU(inplace=True) if self.leaky else nn.ReLU(inplace=True)
#             )
#         else:
#             activation = nn.Sequential(
#                 nn.LeakyReLU(inplace=True) if self.leaky else nn.ReLU(inplace=True)
#             )
#
#         return nn.Sequential(
#             OrderedDict(
#                 [
#                     (
#                         name + "conv1",
#                         nn.Conv3d(
#                             in_channels=in_channels,
#                             out_channels=features,
#                             kernel_size=3,
#                             padding=1,
#                             bias=False,
#                         ),
#                     ),
#                     (name + "activation1", activation),
#                     (
#                         name + "conv2",
#                         nn.Conv3d(
#                             in_channels=features,
#                             out_channels=features,
#                             kernel_size=3,
#                             padding=1,
#                             bias=False,
#                         ),
#                     ),
#                     (name + "activation2", activation),
#                 ]
#             )
#         )

# class UNetTest(nn.Module): # Max Pooling, BatchNorm & LeakyReLU on demand
#     """
#     The layout of the model has been adapted from https://github.com/mateuszbuda/brain-segmentation-pytorch,
#     using 3-dimensional kernels and other modifications.
#
#
#     MIT License
#
# Copyright (c) 2019 mateuszbuda
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE."""
#     def __init__(self, in_channels, out_channels, features, batchnorm=True, leaky=True, max_pool=True):
#         '''
#         :param in_channels: modalities
#         :param out_channels: output modalitites
#         :param features: size of input (each modality)
#         :param batchnorm: bool
#         :param leaky: bool
#         :param max_pool: bool
#         '''
#         super(UNetTest, self).__init__()
#         self.batchnorm = batchnorm
#         self.leaky = leaky
#         self.in_channels = in_channels
#         self.seperat_mod_channels = in_channels - 5# channels other than med data
#         self.true_in_channels = in_channels - 3
#         print("channels", self.seperat_mod_channels, self.true_in_channels)
#
#
#         self.encoder11_md = self._block(4, features, name="enc11_md")
#         self.pool11_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2,
#                                                                                                  stride=2)
#         self.encoder21_md = self._block(features, features * 2, name="enc21_md")
#         self.pool21_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2,
#                                                                                                  stride=2)
#         self.encoder31_md = self._block(features * 2, features * 4, name="enc31_md")
#         self.pool31_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2,
#                                                                                              stride=2)
#         self.encoder41_md = self._block(features * 4, features * 8, name="enc41_md")
#         self.pool41_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2,
#                                                                                                  stride=2)
#
#
#
#         self.encoder12_md = self._block(1, features, name="enc12_md")
#         self.pool12_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2,
#                                                                                                  stride=2)
#         self.encoder22_md = self._block(features, features * 2, name="enc22_md")
#         self.pool22_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2,
#                                                                                                  stride=2)
#         self.encoder32_md = self._block(features * 2, features * 4, name="enc32_md")
#         self.pool32_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2,
#                                                                                                  stride=2)
#         self.encoder42_md = self._block(features * 4, features * 8, name="enc42_md")
#         self.pool42_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2,
#                                                                                                  stride=2)
#
#         self.encoder11 = self._block(1, features, name="enc11")
#         self.pool11 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#         self.encoder21 = self._block(features, features * 2, name="enc21")
#         self.pool21 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#         self.encoder31 = self._block(features * 2, features * 4, name="enc31")
#         self.pool31 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#         self.encoder41 = self._block(features * 4, features * 8, name="enc41")
#         self.pool41 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#
#         if self.seperat_mod_channels > 1:
#             print("built channel2")
#             self.encoder12 = self._block(1, features, name="enc12")
#             self.pool12 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#             self.encoder22 = self._block(features, features * 2, name="enc22")
#             self.pool22 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#             self.encoder32 = self._block(features * 2, features * 4, name="enc32")
#             self.pool32 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#             self.encoder42 = self._block(features * 4, features * 8, name="enc42")
#             self.pool42 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#         if self.seperat_mod_channels > 2:
#             print("built channel3")
#             self.encoder13 = self._block(1, features, name="enc13")
#             self.pool13 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#             self.encoder23 = self._block(features, features * 2, name="enc23")
#             self.pool23 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#             self.encoder33 = self._block(features * 2, features * 4, name="enc33")
#             self.pool33 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#             self.encoder43 = self._block(features * 4, features * 8, name="enc43")
#             self.pool43 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#         if self.seperat_mod_channels > 3:
#             print("built channel4")
#             self.encoder14 = self._block(1, features, name="enc14")
#             self.pool14 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#             self.encoder24 = self._block(features, features * 2, name="enc24")
#             self.pool24 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#             self.encoder34 = self._block(features * 2, features * 4, name="enc34")
#             self.pool34 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#             self.encoder44 = self._block(features * 4, features * 8, name="enc44")
#             self.pool44 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#         if self.seperat_mod_channels > 4:
#             print("built channel5")
#             self.encoder15 = self._block(1, features, name="enc15")
#             self.pool15 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#             self.encoder25 = self._block(features, features * 2, name="enc25")
#             self.pool25 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#             self.encoder35 = self._block(features * 2, features * 4, name="enc35")
#             self.pool35 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#             self.encoder45 = self._block(features * 4, features * 8, name="enc45")
#             self.pool45 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
#
#         print("features", features)
#         print("pathways for images", self.seperat_mod_channels," / totals pathways" , self.true_in_channels)
#         self.bottleneck = self._block(features * 8 * self.true_in_channels, features * 16, name="bottleneck")
#
#         self.upconv4 = nn.ConvTranspose3d(
#             features * 16, features * 8, kernel_size=2, stride=2
#         )
#         self.decoder4 = self._block((features * 8) * (self.true_in_channels+1), features * 8, name="dec4")
#         self.upconv3 = nn.ConvTranspose3d(
#             features * 8, features * 4, kernel_size=2, stride=2
#         )
#         self.decoder3 = self._block((features * 4) * (self.true_in_channels+1), features * 4, name="dec3")
#         self.upconv2 = nn.ConvTranspose3d(
#             features * 4, features * 2, kernel_size=2, stride=2
#         )
#         self.decoder2 = self._block((features * 2) * (self.true_in_channels+1), features * 2, name="dec2")
#         self.upconv1 = nn.ConvTranspose3d(
#             features * 2, features, kernel_size=2, stride=2
#         )
#         self.decoder1 = self._block(features * (self.true_in_channels+1), features, name="dec1")
#
#         self.conv = nn.Conv3d(
#             in_channels=features, out_channels=out_channels, kernel_size=1
#         )
#
#     def forward(self, x):
#
#         mod_list = []
#         med_data = None
#         tici_data = None
#
#         for batch_item in x:# run thru all items in the batch
#
#             if len(mod_list) == 0:  # mod list empty
#                 for j in range(6):
#                     if j == 0: #get medical data
#                         xj = torch.unsqueeze(batch_item[1:5], 0)#get first item in batch and get med modalities
#                         med_data = xj
#                         tici_data_ap = torch.unsqueeze(batch_item[0], 0)
#                         tici_data = torch.unsqueeze(tici_data_ap, 0)
#                     else: #get other modalities
#                         xj = torch.unsqueeze(batch_item[j+4], 0)
#                         mod_list.append(xj)  # append split modalities to mod_list
#             else:
#                 for j in range(6):
#                     if j == 0: #get medical data
#                         xj = torch.unsqueeze(batch_item[1:5], 0)#get first item in batch and get med modalities
#                         med_data = torch.cat((med_data, xj), 0)
#                         tici_data_ap = torch.unsqueeze(batch_item[0], 0)
#                         tici_data_ap = torch.unsqueeze(tici_data_ap, 0)
#                         tici_data = torch.cat((tici_data, tici_data_ap), 0)
#                     else: #get other modalities
#                         xj = torch.unsqueeze(batch_item[j+4], 0)
#                         item = mod_list[j-1]  # get all other modalities in the batch of same type in mod_list
#                         mod_list[j-1] = torch.cat((item, xj), 0)  # append the new one
#
#
#         enc11_md = self.encoder11_md(med_data)
#         enc21_md = self.encoder22_md(self.pool11_md(enc11_md))
#         enc31_md = self.encoder32_md(self.pool21_md(enc21_md))
#         enc41_md = self.encoder42_md(self.pool31_md(enc31_md))
#         pool41_md = self.pool41_md(enc41_md)
#
#         enc12_md = self.encoder12_md(tici_data)
#         enc22_md = self.encoder22_md(self.pool12_md(enc12_md))
#         enc32_md = self.encoder32_md(self.pool22_md(enc22_md))
#         enc42_md = self.encoder42_md(self.pool32_md(enc32_md))
#         pool42_md = self.pool42_md(enc42_md)
#
#
#
#         inp = torch.unsqueeze(mod_list[0],1)
#         enc11 = self.encoder11(inp)
#         enc21 = self.encoder21(self.pool11(enc11))
#         enc31 = self.encoder31(self.pool21(enc21))
#         enc41 = self.encoder41(self.pool31(enc31))
#         pool41 = self.pool41(enc41)
#
#         bottle_input = torch.cat((pool41_md, pool42_md, pool41), dim = 1)
#         dec4_input = torch.cat((enc41_md, enc42_md, enc41), dim = 1)
#         dec3_input = torch.cat((enc31_md, enc32_md, enc31), dim = 1)
#         dec2_input = torch.cat((enc21_md, enc22_md, enc21), dim = 1)
#         dec1_input = torch.cat((enc11_md, enc12_md, enc11), dim = 1)
#
#         if self.seperat_mod_channels > 1:
#             enc12 = self.encoder12(torch.unsqueeze(mod_list[1],1))
#             dec1_input = torch.cat((dec1_input, enc12), dim=1)
#             enc22 = self.encoder22(self.pool12(enc12))
#             dec2_input = torch.cat((dec2_input, enc22), dim=1)
#             enc32 = self.encoder32(self.pool22(enc22))
#             dec3_input = torch.cat((dec3_input, enc32), dim=1)
#             enc42 = self.encoder42(self.pool32(enc32))
#             dec4_input = torch.cat((dec4_input, enc42), dim=1)
#             pool42 = self.pool42(enc42)
#             bottle_input = torch.cat((bottle_input, pool42), dim=1)
#         if self.seperat_mod_channels > 2:
#             enc13 = self.encoder13(torch.unsqueeze(mod_list[2],1))
#             dec1_input = torch.cat((dec1_input, enc13), dim=1)
#             enc23 = self.encoder23(self.pool13(enc13))
#             dec2_input = torch.cat((dec2_input, enc23), dim=1)
#             enc33 = self.encoder33(self.pool23(enc23))
#             dec3_input = torch.cat((dec3_input, enc33), dim=1)
#             enc43 = self.encoder43(self.pool33(enc33))
#             dec4_input = torch.cat((dec4_input, enc43), dim=1)
#             pool43 = self.pool43(enc43)
#             bottle_input = torch.cat((bottle_input, pool43), dim=1)
#         if self.seperat_mod_channels > 3:
#             enc14 = self.encoder14(torch.unsqueeze(mod_list[3],1))
#             dec1_input = torch.cat((dec1_input, enc14), dim=1)
#             enc24 = self.encoder24(self.pool14(enc14))
#             dec2_input = torch.cat((dec2_input, enc24), dim=1)
#             enc34 = self.encoder34(self.pool24(enc24))
#             dec3_input = torch.cat((dec3_input, enc34), dim=1)
#             enc44 = self.encoder44(self.pool34(enc34))
#             dec4_input = torch.cat((dec4_input, enc44), dim=1)
#             pool44 = self.pool44(enc44)
#             bottle_input = torch.cat((bottle_input, pool44), dim=1)
#         if self.seperat_mod_channels > 4:
#             enc15 = self.encoder15(torch.unsqueeze(mod_list[4],1))
#             dec1_input = torch.cat((dec1_input, enc15), dim=1)
#             enc25 = self.encoder25(self.pool15(enc15))
#             dec2_input = torch.cat((dec2_input, enc25), dim=1)
#             enc35 = self.encoder35(self.pool25(enc25))
#             dec3_input = torch.cat((dec3_input, enc35), dim=1)
#             enc45 = self.encoder45(self.pool35(enc35))
#             dec4_input = torch.cat((dec4_input, enc45), dim=1)
#             pool45 = self.pool45(enc45)
#             bottle_input = torch.cat((bottle_input, pool45), dim=1)
#
#
#
#         bottleneck = self.bottleneck(bottle_input)
#
#         dec4 = self.upconv4(bottleneck)
#         dec4 = torch.cat((dec4, dec4_input), dim=1)
#         dec4 = self.decoder4(dec4)
#         dec3 = self.upconv3(dec4)
#         dec3 = torch.cat((dec3, dec3_input), dim=1)
#         dec3 = self.decoder3(dec3)
#         dec2 = self.upconv2(dec3)
#         dec2 = torch.cat((dec2, dec2_input), dim=1)
#         dec2 = self.decoder2(dec2)
#         dec1 = self.upconv1(dec2)
#         dec1 = torch.cat((dec1, dec1_input), dim=1)
#         dec1 = self.decoder1(dec1)
#         return torch.sigmoid(self.conv(dec1))
#
#     def _block(self, in_channels, features, name):
#         if self.batchnorm:
#             activation = nn.Sequential(
#                 nn.BatchNorm3d(num_features=features),
#                 nn.LeakyReLU(inplace=True) if self.leaky else nn.ReLU(inplace=True)
#             )
#         else:
#             activation = nn.Sequential(
#                 nn.LeakyReLU(inplace=True) if self.leaky else nn.ReLU(inplace=True)
#             )
#
#         return nn.Sequential(
#             OrderedDict(
#                 [
#                     (
#                         name + "conv1",
#                         nn.Conv3d(
#                             in_channels=in_channels,
#                             out_channels=features,
#                             kernel_size=3,
#                             padding=1,
#                             bias=False,
#                         ),
#                     ),
#                     (name + "activation1", activation),
#                     (
#                         name + "conv2",
#                         nn.Conv3d(
#                             in_channels=features,
#                             out_channels=features,
#                             kernel_size=3,
#                             padding=1,
#                             bias=False,
#                         ),
#                     ),
#                     (name + "activation2", activation),
#                 ]
#             )
#         )

#----------------------------------------------------------------------------------
#
#                                GPU CHECKPOINT TEST
#
#----------------------------------------------------------------------------------


class UNetSepMedDataCheckpoint(nn.Module): # Max Pooling, BatchNorm & LeakyReLU on demand
    def __init__(self, in_channels, out_channels, features, batchnorm=True, leaky=True, max_pool=True):
        '''
        :param in_channels: modalities
        :param out_channels: output modalitites
        :param features: size of input (each modality)
        :param batchnorm: bool
        :param leaky: bool
        :param max_pool: bool
        '''
        super(UNetSepMedDataCheckpoint, self).__init__()
        self.batchnorm = batchnorm
        self.leaky = leaky
        self.in_channels = in_channels
        self.imaging_modalities = in_channels - 5

        self.encoder11_md = self._block(1, features, name="enc11_md")
        self.pool11_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder21_md = self._block(features, features * 2, name="enc21_md")
        self.pool21_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder31_md = self._block(features * 2, features * 4, name="enc31_md")
        self.pool31_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder41_md = self._block(features * 4, features * 8, name="enc41_md")
        self.pool41_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)

        self.encoder12_md = self._block(1, features, name="enc12_md")
        self.pool12_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder22_md = self._block(features, features * 2, name="enc22_md")
        self.pool22_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder32_md = self._block(features * 2, features * 4, name="enc32_md")
        self.pool32_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder42_md = self._block(features * 4, features * 8, name="enc42_md")
        self.pool42_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)

        self.encoder13_md = self._block(1, features, name="enc13_md")
        self.pool13_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder23_md = self._block(features, features * 2, name="enc23_md")
        self.pool23_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder33_md = self._block(features * 2, features * 4, name="enc33_md")
        self.pool33_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder43_md = self._block(features * 4, features * 8, name="enc43_md")
        self.pool43_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)

        self.encoder14_md = self._block(1, features, name="enc14_md")
        self.pool14_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder24_md = self._block(features, features * 2, name="enc24_md")
        self.pool24_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder34_md = self._block(features * 2, features * 4, name="enc34_md")
        self.pool34_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder44_md = self._block(features * 4, features * 8, name="enc44_md")
        self.pool44_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)

        self.encoder15_md = self._block(1, features, name="enc15_md")
        self.pool15_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder25_md = self._block(features, features * 2, name="enc25_md")
        self.pool25_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder35_md = self._block(features * 2, features * 4, name="enc35_md")
        self.pool35_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder45_md = self._block(features * 4, features * 8, name="enc45_md")
        self.pool45_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)

        self.encoder11 = self._block(1, features, name="enc11")
        self.pool11 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder21 = self._block(features, features * 2, name="enc21")
        self.pool21 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder31 = self._block(features * 2, features * 4, name="enc31")
        self.pool31 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder41 = self._block(features * 4, features * 8, name="enc41")
        self.pool41 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)

        if self.imaging_modalities > 1:
            print("built channel2")
            self.encoder12 = self._block(1, features, name="enc12")
            self.pool12 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder22 = self._block(features, features * 2, name="enc22")
            self.pool22 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder32 = self._block(features * 2, features * 4, name="enc32")
            self.pool32 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder42 = self._block(features * 4, features * 8, name="enc42")
            self.pool42 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        if self.imaging_modalities > 2:
            print("built channel3")
            self.encoder13 = self._block(1, features, name="enc13")
            self.pool13 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder23 = self._block(features, features * 2, name="enc23")
            self.pool23 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder33 = self._block(features * 2, features * 4, name="enc33")
            self.pool33 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder43 = self._block(features * 4, features * 8, name="enc43")
            self.pool43 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        if self.imaging_modalities > 3:
            print("built channel4")
            self.encoder14 = self._block(1, features, name="enc14")
            self.pool14 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder24 = self._block(features, features * 2, name="enc24")
            self.pool24 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder34 = self._block(features * 2, features * 4, name="enc34")
            self.pool34 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder44 = self._block(features * 4, features * 8, name="enc44")
            self.pool44 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        if self.imaging_modalities > 4:
            print("built channel5")
            self.encoder15 = self._block(1, features, name="enc15")
            self.pool15 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder25 = self._block(features, features * 2, name="enc25")
            self.pool25 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder35 = self._block(features * 2, features * 4, name="enc35")
            self.pool35 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder45 = self._block(features * 4, features * 8, name="enc45")
            self.pool45 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)

        print("features", features)
        print("pathways for images ", self.in_channels)
        self.bottleneck = self._block(features * 8 * self.in_channels, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self._block((features * 8) * (self.in_channels + 1), features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self._block((features * 4) * (self.in_channels + 1), features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self._block((features * 2) * (self.in_channels + 1), features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = self._block(features * (self.in_channels + 1), features, name="dec1")

        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):

        inp = torch.unsqueeze(x[:, -1, :, :, :], 1)
        enc11 = checkpoint(self.custom(self.encoder11), inp)
        pool11 = self.pool11(enc11)
        enc21 = checkpoint(self.custom(self.encoder21), pool11)
        pool21 = self.pool21(enc21)
        enc31 = checkpoint(self.custom(self.encoder31), pool21)
        pool31 = self.pool31(enc31)
        enc41 = checkpoint(self.custom(self.encoder41), pool31)
        pool41 = self.pool41(enc41)

        bottle_input = pool41
        dec4_input = enc41
        dec3_input = enc31
        dec2_input = enc21
        dec1_input = enc11

        if self.imaging_modalities > 1:
            inp = torch.unsqueeze(x[:, -2, :, :, :], 1)
            enc12 = checkpoint(self.custom(self.encoder12), inp)
            pool12 = self.pool12(enc12)
            enc22 = checkpoint(self.custom(self.encoder22), pool12)
            pool22 = self.pool22(enc22)
            enc32 = checkpoint(self.custom(self.encoder32), pool22)
            pool32 = self.pool32(enc32)
            enc42 = checkpoint(self.custom(self.encoder42), pool32)
            pool42 = self.pool42(enc42)

            bottle_input = torch.cat((bottle_input, pool42), dim=1)
            dec4_input = torch.cat((dec4_input, enc42), dim=1)
            dec3_input = torch.cat((dec3_input, enc32), dim=1)
            dec2_input = torch.cat((dec2_input, enc22), dim=1)
            dec1_input = torch.cat((dec1_input, enc12), dim=1)

        if self.imaging_modalities > 2:
            inp = torch.unsqueeze(x[:, -3, :, :, :], 1)
            enc13 = checkpoint(self.custom(self.encoder13), inp)
            pool13 = self.pool13(enc13)
            enc23 = checkpoint(self.custom(self.encoder23), pool13)
            pool23 = self.pool23(enc23)
            enc33 = checkpoint(self.custom(self.encoder33), pool23)
            pool33 = self.pool33(enc33)
            enc43 = checkpoint(self.custom(self.encoder43), pool33)
            pool43 = self.pool43(enc43)

            bottle_input = torch.cat((bottle_input, pool43), dim=1)
            dec4_input = torch.cat((dec4_input, enc43), dim=1)
            dec3_input = torch.cat((dec3_input, enc33), dim=1)
            dec2_input = torch.cat((dec2_input, enc23), dim=1)
            dec1_input = torch.cat((dec1_input, enc13), dim=1)

        if self.imaging_modalities > 3:
            inp = torch.unsqueeze(x[:, -4, :, :, :], 1)
            enc14 = checkpoint(self.custom(self.encoder14), inp)
            pool14 = self.pool14(enc14)
            enc24 = checkpoint(self.custom(self.encoder24), pool14)
            pool24 = self.pool24(enc24)
            enc34 = checkpoint(self.custom(self.encoder34), pool24)
            pool34 = self.pool34(enc34)
            enc44 = checkpoint(self.custom(self.encoder44), pool34)
            pool44 = self.pool44(enc44)

            bottle_input = torch.cat((bottle_input, pool44), dim=1)
            dec4_input = torch.cat((dec4_input, enc44), dim=1)
            dec3_input = torch.cat((dec3_input, enc34), dim=1)
            dec2_input = torch.cat((dec2_input, enc24), dim=1)
            dec1_input = torch.cat((dec1_input, enc14), dim=1)

        if self.imaging_modalities > 4:
            inp = torch.unsqueeze(x[:, -5, :, :, :], 1)
            enc15 = checkpoint(self.custom(self.encoder15), inp)
            pool15 = self.pool15(enc15)
            enc25 = checkpoint(self.custom(self.encoder25), pool15)
            pool25 = self.pool25(enc25)
            enc35 = checkpoint(self.custom(self.encoder35), pool25)
            pool35 = self.pool35(enc35)
            enc45 = checkpoint(self.custom(self.encoder45), pool35)
            pool45 = self.pool45(enc45)

            bottle_input = torch.cat((bottle_input, pool45), dim=1)
            dec4_input = torch.cat((dec4_input, enc45), dim=1)
            dec3_input = torch.cat((dec3_input, enc35), dim=1)
            dec2_input = torch.cat((dec2_input, enc25), dim=1)
            dec1_input = torch.cat((dec1_input, enc15), dim=1)

        inp = torch.unsqueeze(x[:, 0, :, :, :], dim=1)
        enc11_md = checkpoint(self.custom(self.encoder11_md), inp)
        pool11_md = self.pool11(enc11_md)
        enc21_md = checkpoint(self.custom(self.encoder21_md), pool11_md)
        pool21_md = self.pool21(enc21_md)
        enc31_md = checkpoint(self.custom(self.encoder31_md), pool21_md)
        pool31_md = self.pool31(enc31_md)
        enc41_md = checkpoint(self.custom(self.encoder41_md), pool31_md)
        pool41_md = self.pool41(enc41_md)

        bottle_input = torch.cat((bottle_input, pool41_md), dim=1)
        dec4_input = torch.cat((dec4_input, enc41_md), dim=1)
        dec3_input = torch.cat((dec3_input, enc31_md), dim=1)
        dec2_input = torch.cat((dec2_input, enc21_md), dim=1)
        dec1_input = torch.cat((dec1_input, enc11_md), dim=1)

        inp = torch.unsqueeze(x[:, 1, :, :, :], dim=1)
        enc12_md = checkpoint(self.custom(self.encoder12_md), inp)
        pool12_md = self.pool12(enc12_md)
        enc22_md = checkpoint(self.custom(self.encoder22_md), pool12_md)
        pool22_md = self.pool22(enc22_md)
        enc32_md = checkpoint(self.custom(self.encoder32_md), pool22_md)
        pool32_md = self.pool32(enc32_md)
        enc42_md = checkpoint(self.custom(self.encoder42_md), pool32_md)
        pool42_md = self.pool42(enc42_md)

        bottle_input = torch.cat((bottle_input, pool42_md), dim=1)
        dec4_input = torch.cat((dec4_input, enc42_md), dim=1)
        dec3_input = torch.cat((dec3_input, enc32_md), dim=1)
        dec2_input = torch.cat((dec2_input, enc22_md), dim=1)
        dec1_input = torch.cat((dec1_input, enc12_md), dim=1)

        inp = torch.unsqueeze(x[:, 2, :, :, :], dim=1)
        enc13_md = checkpoint(self.custom(self.encoder13_md), inp)
        pool13_md = self.pool13(enc13_md)
        enc23_md = checkpoint(self.custom(self.encoder23_md), pool13_md)
        pool23_md = self.pool23(enc23_md)
        enc33_md = checkpoint(self.custom(self.encoder33_md), pool23_md)
        pool33_md = self.pool33(enc33_md)
        enc43_md = checkpoint(self.custom(self.encoder43_md), pool33_md)
        pool43_md = self.pool43(enc43_md)

        bottle_input = torch.cat((bottle_input, pool43_md), dim=1)
        dec4_input = torch.cat((dec4_input, enc43_md), dim=1)
        dec3_input = torch.cat((dec3_input, enc33_md), dim=1)
        dec2_input = torch.cat((dec2_input, enc23_md), dim=1)
        dec1_input = torch.cat((dec1_input, enc13_md), dim=1)

        inp = torch.unsqueeze(x[:, 3, :, :, :], dim=1)
        enc14_md = checkpoint(self.custom(self.encoder14_md), inp)
        pool14_md = self.pool14(enc14_md)
        enc24_md = checkpoint(self.custom(self.encoder24_md), pool14_md)
        pool24_md = self.pool24(enc24_md)
        enc34_md = checkpoint(self.custom(self.encoder34_md), pool24_md)
        pool34_md = self.pool34(enc34_md)
        enc44_md = checkpoint(self.custom(self.encoder44_md), pool34_md)
        pool44_md = self.pool44(enc44_md)

        bottle_input = torch.cat((bottle_input, pool44_md), dim=1)
        dec4_input = torch.cat((dec4_input, enc44_md), dim=1)
        dec3_input = torch.cat((dec3_input, enc34_md), dim=1)
        dec2_input = torch.cat((dec2_input, enc24_md), dim=1)
        dec1_input = torch.cat((dec1_input, enc14_md), dim=1)

        inp = torch.unsqueeze(x[:, 4, :, :, :], dim=1)
        enc15_md = checkpoint(self.custom(self.encoder15_md), inp)
        pool15_md = self.pool15(enc15_md)
        enc25_md = checkpoint(self.custom(self.encoder25_md), pool15_md)
        pool25_md = self.pool25(enc25_md)
        enc35_md = checkpoint(self.custom(self.encoder35_md), pool25_md)
        pool35_md = self.pool35(enc35_md)
        enc45_md = checkpoint(self.custom(self.encoder45_md), pool35_md)
        pool45_md = self.pool45(enc45_md)

        bottle_input = torch.cat((bottle_input, pool45_md), dim=1)
        dec4_input = torch.cat((dec4_input, enc45_md), dim=1)
        dec3_input = torch.cat((dec3_input, enc35_md), dim=1)
        dec2_input = torch.cat((dec2_input, enc25_md), dim=1)
        dec1_input = torch.cat((dec1_input, enc15_md), dim=1)



        bottleneck = checkpoint(self.custom(self.bottleneck), bottle_input)

        upconv4 = self.upconv4(bottleneck)
        dec4_input = torch.cat((upconv4, dec4_input), dim=1)
        dec4 = checkpoint(self.custom(self.decoder4), dec4_input)

        upconv3 = self.upconv3(dec4)
        dec3_input = torch.cat((upconv3, dec3_input), dim=1)
        dec3 = checkpoint(self.custom(self.decoder3), dec3_input)

        upconv2 = self.upconv2(dec3)
        dec2_input = torch.cat((upconv2, dec2_input), dim=1)
        dec2 = checkpoint(self.custom(self.decoder2), dec2_input)

        upconv1 = self.upconv1(dec2)
        dec1_input = torch.cat((upconv1, dec1_input), dim=1)
        dec1 = checkpoint(self.custom(self.decoder1), dec1_input)
        return torch.sigmoid(self.conv(dec1))

        # define custom forward function, needed for checkpointing

    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs

        return custom_forward

    def _block(self, in_channels, features, name):
        if self.batchnorm:
            activation = nn.Sequential(
                nn.BatchNorm3d(num_features=features, momentum=(1 - math.sqrt(0.9))),
                nn.LeakyReLU(inplace=True) if self.leaky else nn.ReLU(inplace=True)
            )
        else:
            activation = nn.Sequential(
                nn.LeakyReLU(inplace=True) if self.leaky else nn.ReLU(inplace=True)
            )

        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "activation1", activation),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "activation2", activation),
                ]
            )
        )

class UNetSepMedDataCombinedCheckpoint(nn.Module):
    def __init__(self, in_channels, out_channels, features, batchnorm=True, leaky=True, max_pool=True):
        '''
        :param in_channels: modalities
        :param out_channels: output modalitites
        :param features: size of input (each modality)
        :param batchnorm: bool
        :param leaky: bool
        :param max_pool: bool
        '''
        super(UNetSepMedDataCombinedCheckpoint, self).__init__()
        self.batchnorm = batchnorm
        self.leaky = leaky
        self.in_channels = in_channels
        self.imaging_modalities = in_channels - 5

        self.encoder11_md = self._block(5, features, name="enc11_md")
        self.pool11_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder21_md = self._block(features, features * 2, name="enc21_md")
        self.pool21_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder31_md = self._block(features * 2, features * 4, name="enc31_md")
        self.pool31_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder41_md = self._block(features * 4, features * 8, name="enc41_md")
        self.pool41_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)



        self.encoder11 = self._block(1, features, name="enc11")
        self.pool11 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder21 = self._block(features, features * 2, name="enc21")
        self.pool21 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder31 = self._block(features * 2, features * 4, name="enc31")
        self.pool31 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder41 = self._block(features * 4, features * 8, name="enc41")
        self.pool41 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)

        if self.imaging_modalities > 1 :
            print("built channel2")
            self.encoder12 = self._block(1, features, name="enc12")
            self.pool12 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder22 = self._block(features, features * 2, name="enc22")
            self.pool22 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder32 = self._block(features * 2, features * 4, name="enc32")
            self.pool32 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder42 = self._block(features * 4, features * 8, name="enc42")
            self.pool42 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        if self.imaging_modalities > 2:
            print("built channel3")
            self.encoder13 = self._block(1, features, name="enc13")
            self.pool13 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder23 = self._block(features, features * 2, name="enc23")
            self.pool23 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder33 = self._block(features * 2, features * 4, name="enc33")
            self.pool33 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder43 = self._block(features * 4, features * 8, name="enc43")
            self.pool43 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        if self.imaging_modalities > 3:
            print("built channel4")
            self.encoder14 = self._block(1, features, name="enc14")
            self.pool14 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder24 = self._block(features, features * 2, name="enc24")
            self.pool24 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder34 = self._block(features * 2, features * 4, name="enc34")
            self.pool34 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder44 = self._block(features * 4, features * 8, name="enc44")
            self.pool44 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        if self.imaging_modalities > 4:
            print("built channel5")
            self.encoder15 = self._block(1, features, name="enc15")
            self.pool15 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder25 = self._block(features, features * 2, name="enc25")
            self.pool25 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder35 = self._block(features * 2, features * 4, name="enc35")
            self.pool35 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder45 = self._block(features * 4, features * 8, name="enc45")
            self.pool45 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)

        print("features", features)
        print("pathways for images ", self.in_channels)
        self.bottleneck = self._block(features * 8 * (self.imaging_modalities+1), features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self._block((features * 8) * (self.imaging_modalities+2), features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self._block((features * 4) * (self.imaging_modalities+2), features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self._block((features * 2) * (self.imaging_modalities+2), features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = self._block(features * (self.imaging_modalities+2), features, name="dec1")

        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):

        inp = torch.unsqueeze(x[:,-1,:,:,:],1)
        enc11 = checkpoint(self.custom(self.encoder11),inp)
        pool11 = self.pool11(enc11)
        enc21 = checkpoint(self.custom(self.encoder21), pool11)
        pool21 = self.pool21(enc21)
        enc31 = checkpoint(self.custom(self.encoder31),pool21)
        pool31 = self.pool31(enc31)
        enc41 = checkpoint(self.custom(self.encoder41),pool31)
        pool41 = self.pool41(enc41)


        bottle_input = pool41
        dec4_input = enc41
        dec3_input = enc31
        dec2_input = enc21
        dec1_input = enc11

        if self.imaging_modalities > 1:
            inp = torch.unsqueeze(x[:, -2, :, :, :], 1)
            enc12 = checkpoint(self.custom(self.encoder12), inp)
            pool12 = self.pool12(enc12)
            enc22 = checkpoint(self.custom(self.encoder22), pool12)
            pool22 = self.pool22(enc22)
            enc32 = checkpoint(self.custom(self.encoder32), pool22)
            pool32 = self.pool32(enc32)
            enc42 = checkpoint(self.custom(self.encoder42), pool32)
            pool42 = self.pool42(enc42)

            bottle_input = torch.cat((bottle_input, pool42), dim=1)
            dec4_input = torch.cat((dec4_input, enc42), dim=1)
            dec3_input = torch.cat((dec3_input, enc32), dim=1)
            dec2_input = torch.cat((dec2_input, enc22), dim=1)
            dec1_input = torch.cat((dec1_input, enc12), dim=1)

        if self.imaging_modalities > 2:
            inp = torch.unsqueeze(x[:, -3, :, :, :], 1)
            enc13 = checkpoint(self.custom(self.encoder13), inp)
            pool13 = self.pool13(enc13)
            enc23 = checkpoint(self.custom(self.encoder23), pool13)
            pool23 = self.pool23(enc23)
            enc33 = checkpoint(self.custom(self.encoder33), pool23)
            pool33 = self.pool33(enc33)
            enc43 = checkpoint(self.custom(self.encoder43), pool33)
            pool43 = self.pool43(enc43)

            bottle_input = torch.cat((bottle_input, pool43), dim=1)
            dec4_input = torch.cat((dec4_input, enc43), dim=1)
            dec3_input = torch.cat((dec3_input, enc33), dim=1)
            dec2_input = torch.cat((dec2_input, enc23), dim=1)
            dec1_input = torch.cat((dec1_input, enc13), dim=1)

        if self.imaging_modalities > 3:
            inp = torch.unsqueeze(x[:, -4, :, :, :], 1)
            enc14 = checkpoint(self.custom(self.encoder14), inp)
            pool14 = self.pool14(enc14)
            enc24 = checkpoint(self.custom(self.encoder24), pool14)
            pool24 = self.pool24(enc24)
            enc34 = checkpoint(self.custom(self.encoder34), pool24)
            pool34 = self.pool34(enc34)
            enc44 = checkpoint(self.custom(self.encoder44), pool34)
            pool44 = self.pool44(enc44)

            bottle_input = torch.cat((bottle_input, pool44), dim=1)
            dec4_input = torch.cat((dec4_input, enc44), dim=1)
            dec3_input = torch.cat((dec3_input, enc34), dim=1)
            dec2_input = torch.cat((dec2_input, enc24), dim=1)
            dec1_input = torch.cat((dec1_input, enc14), dim=1)

        if self.imaging_modalities > 4:
            inp = torch.unsqueeze(x[:, -5, :, :, :], 1)
            enc15 = checkpoint(self.custom(self.encoder15), inp)
            pool15 = self.pool15(enc15)
            enc25 = checkpoint(self.custom(self.encoder25), pool15)
            pool25 = self.pool25(enc25)
            enc35 = checkpoint(self.custom(self.encoder35), pool25)
            pool35 = self.pool35(enc35)
            enc45 = checkpoint(self.custom(self.encoder45), pool35)
            pool45 = self.pool45(enc45)

            bottle_input = torch.cat((bottle_input, pool45), dim=1)
            dec4_input = torch.cat((dec4_input, enc45), dim=1)
            dec3_input = torch.cat((dec3_input, enc35), dim=1)
            dec2_input = torch.cat((dec2_input, enc25), dim=1)
            dec1_input = torch.cat((dec1_input, enc15), dim=1)

        inp = x[:, :5, :, :, :]
        enc11_md = checkpoint(self.custom(self.encoder11_md), inp)
        pool11_md = self.pool11(enc11_md)
        enc21_md = checkpoint(self.custom(self.encoder21_md), pool11_md)
        pool21_md = self.pool21(enc21_md)
        enc31_md = checkpoint(self.custom(self.encoder31_md), pool21_md)
        pool31_md = self.pool31(enc31_md)
        enc41_md = checkpoint(self.custom(self.encoder41_md), pool31_md)
        pool41_md = self.pool41(enc41_md)


        bottle_input = torch.cat((bottle_input, pool41_md), dim=1)
        dec4_input = torch.cat((dec4_input, enc41_md), dim=1)
        dec3_input = torch.cat((dec3_input, enc31_md), dim=1)
        dec2_input = torch.cat((dec2_input, enc21_md), dim=1)
        dec1_input = torch.cat((dec1_input, enc11_md), dim=1)




        bottleneck = checkpoint(self.custom(self.bottleneck),bottle_input)

        upconv4 = self.upconv4(bottleneck)
        dec4_input = torch.cat((upconv4, dec4_input), dim=1)
        dec4 = checkpoint(self.custom(self.decoder4),dec4_input)

        upconv3 = self.upconv3(dec4)
        dec3_input = torch.cat((upconv3, dec3_input), dim=1)
        dec3 = checkpoint(self.custom(self.decoder3),dec3_input)

        upconv2 = self.upconv2(dec3)
        dec2_input = torch.cat((upconv2, dec2_input), dim=1)
        dec2 = checkpoint(self.custom(self.decoder2),dec2_input)

        upconv1 = self.upconv1(dec2)
        dec1_input = torch.cat((upconv1, dec1_input), dim=1)
        dec1 = checkpoint(self.custom(self.decoder1),dec1_input)
        return torch.sigmoid(self.conv(dec1))

    #define custom forward function, needed for checkpointing
    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward

    def _block(self, in_channels, features, name):
        if self.batchnorm:
            activation = nn.Sequential(
                nn.BatchNorm3d(num_features=features, momentum=(1-math.sqrt(0.9))),
                nn.LeakyReLU(inplace=True) if self.leaky else nn.ReLU(inplace=True)
            )
        else:
            activation = nn.Sequential(
                nn.LeakyReLU(inplace=True) if self.leaky else nn.ReLU(inplace=True)
            )

        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "activation1", activation),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "activation2", activation),
                ]
            )
        )

class UNetSepMedDataCombinedRecaSepCheckpoint(nn.Module):
    def __init__(self, in_channels, out_channels, features, batchnorm=True, leaky=True, max_pool=True, dropout = False):
        '''
        :param in_channels: modalities
        :param out_channels: output modalitites
        :param features: size of input (each modality)
        :param batchnorm: bool
        :param leaky: bool
        :param max_pool: bool
        '''
        super(UNetSepMedDataCombinedRecaSepCheckpoint, self).__init__()
        self.batchnorm = batchnorm
        self.leaky = leaky
        self.in_channels = in_channels
        self.imaging_modalities = in_channels - 5
        self.dropout = dropout

        self.encoder11_md = self._block(1, features, name="enc11_md")
        self.pool11_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder21_md = self._block(features, features * 2, name="enc21_md")
        self.pool21_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder31_md = self._block(features * 2, features * 4, name="enc31_md")
        self.pool31_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder41_md = self._block(features * 4, features * 8, name="enc41_md")
        self.pool41_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)

        self.encoder12_md = self._block(4, features, name="enc12_md")
        self.pool12_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder22_md = self._block(features, features * 2, name="enc22_md")
        self.pool22_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder32_md = self._block(features * 2, features * 4, name="enc32_md")
        self.pool32_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder42_md = self._block(features * 4, features * 8, name="enc42_md")
        self.pool42_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)



        self.encoder11 = self._block(1, features, name="enc11")
        self.pool11 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder21 = self._block(features, features * 2, name="enc21")
        self.pool21 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder31 = self._block(features * 2, features * 4, name="enc31")
        self.pool31 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder41 = self._block(features * 4, features * 8, name="enc41")
        self.pool41 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)

        if self.imaging_modalities > 1 :
            print("built channel2")
            self.encoder12 = self._block(1, features, name="enc12")
            self.pool12 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder22 = self._block(features, features * 2, name="enc22")
            self.pool22 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder32 = self._block(features * 2, features * 4, name="enc32")
            self.pool32 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder42 = self._block(features * 4, features * 8, name="enc42")
            self.pool42 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        if self.imaging_modalities > 2:
            print("built channel3")
            self.encoder13 = self._block(1, features, name="enc13")
            self.pool13 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder23 = self._block(features, features * 2, name="enc23")
            self.pool23 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder33 = self._block(features * 2, features * 4, name="enc33")
            self.pool33 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder43 = self._block(features * 4, features * 8, name="enc43")
            self.pool43 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        if self.imaging_modalities > 3:
            print("built channel4")
            self.encoder14 = self._block(1, features, name="enc14")
            self.pool14 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder24 = self._block(features, features * 2, name="enc24")
            self.pool24 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder34 = self._block(features * 2, features * 4, name="enc34")
            self.pool34 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder44 = self._block(features * 4, features * 8, name="enc44")
            self.pool44 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        if self.imaging_modalities > 4:
            print("built channel5")
            self.encoder15 = self._block(1, features, name="enc15")
            self.pool15 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder25 = self._block(features, features * 2, name="enc25")
            self.pool25 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder35 = self._block(features * 2, features * 4, name="enc35")
            self.pool35 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder45 = self._block(features * 4, features * 8, name="enc45")
            self.pool45 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)

        print("features", features)
        print("pathways for images ", self.in_channels)
        self.bottleneck = self._block(features * 8 * (self.imaging_modalities+2), features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self._block((features * 8) * (self.imaging_modalities+3), features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self._block((features * 4) * (self.imaging_modalities+3), features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self._block((features * 2) * (self.imaging_modalities+3), features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = self._block(features * (self.imaging_modalities+3), features, name="dec1")

        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):

        inp = torch.unsqueeze(x[:,-1,:,:,:],1)
        enc11 = checkpoint(self.custom(self.encoder11),inp)
        if self.dropout:
            enc11 = nn.Dropout3d(p=0.1)(enc11)
        pool11 = self.pool11(enc11)
        enc21 = checkpoint(self.custom(self.encoder21), pool11)
        if self.dropout:
            enc21 = nn.Dropout3d(p=0.1)(enc21)
        pool21 = self.pool21(enc21)
        enc31 = checkpoint(self.custom(self.encoder31),pool21)
        if self.dropout:
            enc31 = nn.Dropout3d(p=0.1)(enc31)
        pool31 = self.pool31(enc31)
        enc41 = checkpoint(self.custom(self.encoder41),pool31)
        if self.dropout:
            enc41 = nn.Dropout3d(p=0.1)(enc41)
        pool41 = self.pool41(enc41)


        bottle_input = pool41
        dec4_input = enc41
        dec3_input = enc31
        dec2_input = enc21
        dec1_input = enc11

        if self.imaging_modalities > 1:
            inp = torch.unsqueeze(x[:, -2, :, :, :], 1)
            enc12 = checkpoint(self.custom(self.encoder12), inp)
            if self.dropout:
                enc12 = nn.Dropout3d(p=0.1)(enc12)
            pool12 = self.pool12(enc12)
            enc22 = checkpoint(self.custom(self.encoder22), pool12)
            if self.dropout:
                enc22 = nn.Dropout3d(p=0.1)(enc22)
            pool22 = self.pool22(enc22)
            enc32 = checkpoint(self.custom(self.encoder32), pool22)
            if self.dropout:
                enc32 = nn.Dropout3d(p=0.1)(enc32)
            pool32 = self.pool32(enc32)
            enc42 = checkpoint(self.custom(self.encoder42), pool32)
            if self.dropout:
                enc42 = nn.Dropout3d(p=0.1)(enc42)
            pool42 = self.pool42(enc42)

            bottle_input = torch.cat((bottle_input, pool42), dim=1)
            dec4_input = torch.cat((dec4_input, enc42), dim=1)
            dec3_input = torch.cat((dec3_input, enc32), dim=1)
            dec2_input = torch.cat((dec2_input, enc22), dim=1)
            dec1_input = torch.cat((dec1_input, enc12), dim=1)

        if self.imaging_modalities > 2:
            inp = torch.unsqueeze(x[:, -3, :, :, :], 1)
            enc13 = checkpoint(self.custom(self.encoder13), inp)
            if self.dropout:
                enc13 = nn.Dropout3d(p=0.1)(enc13)
            pool13 = self.pool13(enc13)
            enc23 = checkpoint(self.custom(self.encoder23), pool13)
            if self.dropout:
                enc23 = nn.Dropout3d(p=0.1)(enc23)
            pool23 = self.pool23(enc23)
            enc33 = checkpoint(self.custom(self.encoder33), pool23)
            if self.dropout:
                enc33 = nn.Dropout3d(p=0.1)(enc33)
            pool33 = self.pool33(enc33)
            enc43 = checkpoint(self.custom(self.encoder43), pool33)
            if self.dropout:
                enc43 = nn.Dropout3d(p=0.1)(enc43)
            pool43 = self.pool43(enc43)

            bottle_input = torch.cat((bottle_input, pool43), dim=1)
            dec4_input = torch.cat((dec4_input, enc43), dim=1)
            dec3_input = torch.cat((dec3_input, enc33), dim=1)
            dec2_input = torch.cat((dec2_input, enc23), dim=1)
            dec1_input = torch.cat((dec1_input, enc13), dim=1)

        if self.imaging_modalities > 3:
            inp = torch.unsqueeze(x[:, -4, :, :, :], 1)
            enc14 = checkpoint(self.custom(self.encoder14), inp)
            if self.dropout:
                enc14 = nn.Dropout3d(p=0.1)(enc14)
            pool14 = self.pool14(enc14)
            enc24 = checkpoint(self.custom(self.encoder24), pool14)
            if self.dropout:
                enc24 = nn.Dropout3d(p=0.1)(enc24)
            pool24 = self.pool24(enc24)
            enc34 = checkpoint(self.custom(self.encoder34), pool24)
            if self.dropout:
                enc34 = nn.Dropout3d(p=0.1)(enc34)
            pool34 = self.pool34(enc34)
            enc44 = checkpoint(self.custom(self.encoder44), pool34)
            if self.dropout:
                enc44 = nn.Dropout3d(p=0.1)(enc44)
            pool44 = self.pool44(enc44)

            bottle_input = torch.cat((bottle_input, pool44), dim=1)
            dec4_input = torch.cat((dec4_input, enc44), dim=1)
            dec3_input = torch.cat((dec3_input, enc34), dim=1)
            dec2_input = torch.cat((dec2_input, enc24), dim=1)
            dec1_input = torch.cat((dec1_input, enc14), dim=1)

        if self.imaging_modalities > 4:
            inp = torch.unsqueeze(x[:, -5, :, :, :], 1)
            enc15 = checkpoint(self.custom(self.encoder15), inp)
            if self.dropout:
                enc15 = nn.Dropout3d(p=0.1)(enc15)
            pool15 = self.pool15(enc15)
            enc25 = checkpoint(self.custom(self.encoder25), pool15)
            if self.dropout:
                enc25 = nn.Dropout3d(p=0.1)(enc25)
            pool25 = self.pool25(enc25)
            enc35 = checkpoint(self.custom(self.encoder35), pool25)
            if self.dropout:
                enc35 = nn.Dropout3d(p=0.1)(enc35)
            pool35 = self.pool35(enc35)
            enc45 = checkpoint(self.custom(self.encoder45), pool35)
            if self.dropout:
                enc45 = nn.Dropout3d(p=0.1)(enc45)
            pool45 = self.pool45(enc45)

            bottle_input = torch.cat((bottle_input, pool45), dim=1)
            dec4_input = torch.cat((dec4_input, enc45), dim=1)
            dec3_input = torch.cat((dec3_input, enc35), dim=1)
            dec2_input = torch.cat((dec2_input, enc25), dim=1)
            dec1_input = torch.cat((dec1_input, enc15), dim=1)

        inp = torch.unsqueeze(x[:, 0, :, :, :], dim = 1)
        enc11_md = checkpoint(self.custom(self.encoder11_md), inp)
        if self.dropout:
            enc11_md = nn.Dropout3d(p=0.1)(enc11_md)
        pool11_md = self.pool11(enc11_md)
        enc21_md = checkpoint(self.custom(self.encoder21_md), pool11_md)
        if self.dropout:
            enc21_md = nn.Dropout3d(p=0.1)(enc21_md)
        pool21_md = self.pool21(enc21_md)
        enc31_md = checkpoint(self.custom(self.encoder31_md), pool21_md)
        if self.dropout:
            enc31_md = nn.Dropout3d(p=0.1)(enc31_md)
        pool31_md = self.pool31(enc31_md)
        enc41_md = checkpoint(self.custom(self.encoder41_md), pool31_md)
        if self.dropout:
            enc41_md = nn.Dropout3d(p=0.1)(enc41_md)
        pool41_md = self.pool41(enc41_md)


        bottle_input = torch.cat((bottle_input, pool41_md), dim=1)
        dec4_input = torch.cat((dec4_input, enc41_md), dim=1)
        dec3_input = torch.cat((dec3_input, enc31_md), dim=1)
        dec2_input = torch.cat((dec2_input, enc21_md), dim=1)
        dec1_input = torch.cat((dec1_input, enc11_md), dim=1)


        inp = x[:,1:5,:,:,:]
        enc12_md = checkpoint(self.custom(self.encoder12_md), inp)
        if self.dropout:
            enc12_md = nn.Dropout3d(p=0.1)(enc12_md)
        pool12_md = self.pool12(enc12_md)
        enc22_md = checkpoint(self.custom(self.encoder22_md), pool12_md)
        if self.dropout:
            enc22_md = nn.Dropout3d(p=0.1)(enc22_md)
        pool22_md = self.pool22(enc22_md)
        enc32_md = checkpoint(self.custom(self.encoder32_md), pool22_md)
        if self.dropout:
            enc32_md = nn.Dropout3d(p=0.1)(enc32_md)
        pool32_md = self.pool32(enc32_md)
        enc42_md = checkpoint(self.custom(self.encoder42_md), pool32_md)
        if self.dropout:
            enc42_md = nn.Dropout3d(p=0.1)(enc42_md)
        pool42_md = self.pool42(enc42_md)

        bottle_input = torch.cat((bottle_input, pool42_md), dim=1)
        dec4_input = torch.cat((dec4_input, enc42_md), dim=1)
        dec3_input = torch.cat((dec3_input, enc32_md), dim=1)
        dec2_input = torch.cat((dec2_input, enc22_md), dim=1)
        dec1_input = torch.cat((dec1_input, enc12_md), dim=1)




        bottleneck = checkpoint(self.custom(self.bottleneck),bottle_input)

        upconv4 = self.upconv4(bottleneck)
        dec4_input = torch.cat((upconv4, dec4_input), dim=1)
        dec4 = checkpoint(self.custom(self.decoder4),dec4_input)
        if self.dropout:
            dec4 = nn.Dropout3d(p=0.1)(dec4)

        upconv3 = self.upconv3(dec4)
        dec3_input = torch.cat((upconv3, dec3_input), dim=1)
        dec3 = checkpoint(self.custom(self.decoder3),dec3_input)
        if self.dropout:
            dec3 = nn.Dropout3d(p=0.1)(dec3)

        upconv2 = self.upconv2(dec3)
        dec2_input = torch.cat((upconv2, dec2_input), dim=1)
        dec2 = checkpoint(self.custom(self.decoder2),dec2_input)
        if self.dropout:
            dec2 = nn.Dropout3d(p=0.1)(dec2)

        upconv1 = self.upconv1(dec2)
        dec1_input = torch.cat((upconv1, dec1_input), dim=1)
        dec1 = checkpoint(self.custom(self.decoder1),dec1_input)
        return torch.sigmoid(self.conv(dec1))

    #define custom forward function, needed for checkpointing
    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward

    def _block(self, in_channels, features, name):
        if self.batchnorm:
            activation = nn.Sequential(
                nn.BatchNorm3d(num_features=features, momentum=(1-math.sqrt(0.9))),
                nn.LeakyReLU(inplace=True) if self.leaky else nn.ReLU(inplace=True)
            )
        else:
            activation = nn.Sequential(
                nn.LeakyReLU(inplace=True) if self.leaky else nn.ReLU(inplace=True)
            )

        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "activation1", activation),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "activation2", activation),
                ]
            )
        )


class UNetSepCheckpoint(nn.Module):
    def __init__(self, in_channels, out_channels, features, batchnorm=True, leaky=True, max_pool=True):
        '''
        :param in_channels: modalities
        :param out_channels: output modalitites
        :param features: size of input (each modality)
        :param batchnorm: bool
        :param leaky: bool
        :param max_pool: bool
        '''
        super(UNetSepCheckpoint, self).__init__()
        self.batchnorm = batchnorm
        self.leaky = leaky
        self.in_channels = in_channels



        self.encoder11 = self._block(1, features, name="enc11")
        self.pool11 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder21 = self._block(features, features * 2, name="enc21")
        self.pool21 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder31 = self._block(features * 2, features * 4, name="enc31")
        self.pool31 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder41 = self._block(features * 4, features * 8, name="enc41")
        self.pool41 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)

        if self.in_channels > 1 :
            print("built channel2")
            self.encoder12 = self._block(1, features, name="enc12")
            self.pool12 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder22 = self._block(features, features * 2, name="enc22")
            self.pool22 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder32 = self._block(features * 2, features * 4, name="enc32")
            self.pool32 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder42 = self._block(features * 4, features * 8, name="enc42")
            self.pool42 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        if self.in_channels > 2:
            print("built channel3")
            self.encoder13 = self._block(1, features, name="enc13")
            self.pool13 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder23 = self._block(features, features * 2, name="enc23")
            self.pool23 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder33 = self._block(features * 2, features * 4, name="enc33")
            self.pool33 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder43 = self._block(features * 4, features * 8, name="enc43")
            self.pool43 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        if self.in_channels > 3:
            print("built channel4")
            self.encoder14 = self._block(1, features, name="enc14")
            self.pool14 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder24 = self._block(features, features * 2, name="enc24")
            self.pool24 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder34 = self._block(features * 2, features * 4, name="enc34")
            self.pool34 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder44 = self._block(features * 4, features * 8, name="enc44")
            self.pool44 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        if self.in_channels > 4:
            print("built channel5")
            self.encoder15 = self._block(1, features, name="enc15")
            self.pool15 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder25 = self._block(features, features * 2, name="enc25")
            self.pool25 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder35 = self._block(features * 2, features * 4, name="enc35")
            self.pool35 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder45 = self._block(features * 4, features * 8, name="enc45")
            self.pool45 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)

        print("features", features)
        print("pathways for images ", self.in_channels)
        self.bottleneck = self._block(features * 8 * self.in_channels, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self._block((features * 8) * (self.in_channels+1), features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self._block((features * 4) * (self.in_channels+1), features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self._block((features * 2) * (self.in_channels+1), features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = self._block(features * (self.in_channels+1), features, name="dec1")

        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):

        inp = torch.unsqueeze(x[:,-1,:,:,:],1)
        enc11 = checkpoint(self.custom(self.encoder11),inp)
        pool11 = self.pool11(enc11)
        enc21 = checkpoint(self.custom(self.encoder21), pool11)
        pool21 = self.pool21(enc21)
        enc31 = checkpoint(self.custom(self.encoder31),pool21)
        pool31 = self.pool31(enc31)
        enc41 = checkpoint(self.custom(self.encoder41),pool31)
        pool41 = self.pool41(enc41)


        bottle_input = pool41
        dec4_input = enc41
        dec3_input = enc31
        dec2_input = enc21
        dec1_input = enc11

        if self.in_channels > 1:
            inp = torch.unsqueeze(x[:, -2, :, :, :], 1)
            enc12 = checkpoint(self.custom(self.encoder12), inp)
            pool12 = self.pool12(enc12)
            enc22 = checkpoint(self.custom(self.encoder22), pool12)
            pool22 = self.pool22(enc22)
            enc32 = checkpoint(self.custom(self.encoder32), pool22)
            pool32 = self.pool32(enc32)
            enc42 = checkpoint(self.custom(self.encoder42), pool32)
            pool42 = self.pool42(enc42)

            bottle_input = torch.cat((bottle_input, pool42), dim=1)
            dec4_input = torch.cat((dec4_input, enc42), dim=1)
            dec3_input = torch.cat((dec3_input, enc32), dim=1)
            dec2_input = torch.cat((dec2_input, enc22), dim=1)
            dec1_input = torch.cat((dec1_input, enc12), dim=1)

        if self.in_channels > 2:
            inp = torch.unsqueeze(x[:, -3, :, :, :], 1)
            enc13 = checkpoint(self.custom(self.encoder13), inp)
            pool13 = self.pool13(enc13)
            enc23 = checkpoint(self.custom(self.encoder23), pool13)
            pool23 = self.pool23(enc23)
            enc33 = checkpoint(self.custom(self.encoder33), pool23)
            pool33 = self.pool33(enc33)
            enc43 = checkpoint(self.custom(self.encoder43), pool33)
            pool43 = self.pool43(enc43)

            bottle_input = torch.cat((bottle_input, pool43), dim=1)
            dec4_input = torch.cat((dec4_input, enc43), dim=1)
            dec3_input = torch.cat((dec3_input, enc33), dim=1)
            dec2_input = torch.cat((dec2_input, enc23), dim=1)
            dec1_input = torch.cat((dec1_input, enc13), dim=1)

        if self.in_channels > 3:
            inp = torch.unsqueeze(x[:, -4, :, :, :], 1)
            enc14 = checkpoint(self.custom(self.encoder14), inp)
            pool14 = self.pool14(enc14)
            enc24 = checkpoint(self.custom(self.encoder24), pool14)
            pool24 = self.pool24(enc24)
            enc34 = checkpoint(self.custom(self.encoder34), pool24)
            pool34 = self.pool34(enc34)
            enc44 = checkpoint(self.custom(self.encoder44), pool34)
            pool44 = self.pool44(enc44)

            bottle_input = torch.cat((bottle_input, pool44), dim=1)
            dec4_input = torch.cat((dec4_input, enc44), dim=1)
            dec3_input = torch.cat((dec3_input, enc34), dim=1)
            dec2_input = torch.cat((dec2_input, enc24), dim=1)
            dec1_input = torch.cat((dec1_input, enc14), dim=1)

        if self.in_channels > 4:
            inp = torch.unsqueeze(x[:, -5, :, :, :], 1)
            enc15 = checkpoint(self.custom(self.encoder15), inp)
            pool15 = self.pool15(enc15)
            enc25 = checkpoint(self.custom(self.encoder25), pool15)
            pool25 = self.pool25(enc25)
            enc35 = checkpoint(self.custom(self.encoder35), pool25)
            pool35 = self.pool35(enc35)
            enc45 = checkpoint(self.custom(self.encoder45), pool35)
            pool45 = self.pool45(enc45)

            bottle_input = torch.cat((bottle_input, pool45), dim=1)
            dec4_input = torch.cat((dec4_input, enc45), dim=1)
            dec3_input = torch.cat((dec3_input, enc35), dim=1)
            dec2_input = torch.cat((dec2_input, enc25), dim=1)
            dec1_input = torch.cat((dec1_input, enc15), dim=1)







        bottleneck = checkpoint(self.custom(self.bottleneck),bottle_input)

        upconv4 = self.upconv4(bottleneck)
        dec4_input = torch.cat((upconv4, dec4_input), dim=1)
        dec4 = checkpoint(self.custom(self.decoder4),dec4_input)

        upconv3 = self.upconv3(dec4)
        dec3_input = torch.cat((upconv3, dec3_input), dim=1)
        dec3 = checkpoint(self.custom(self.decoder3),dec3_input)

        upconv2 = self.upconv2(dec3)
        dec2_input = torch.cat((upconv2, dec2_input), dim=1)
        dec2 = checkpoint(self.custom(self.decoder2),dec2_input)

        upconv1 = self.upconv1(dec2)
        dec1_input = torch.cat((upconv1, dec1_input), dim=1)
        dec1 = checkpoint(self.custom(self.decoder1),dec1_input)
        return torch.sigmoid(self.conv(dec1))

    #define custom forward function, needed for checkpointing
    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward

    def _block(self, in_channels, features, name):
        if self.batchnorm:
            activation = nn.Sequential(
                nn.BatchNorm3d(num_features=features, momentum=(1-math.sqrt(0.9))),
                nn.LeakyReLU(inplace=True) if self.leaky else nn.ReLU(inplace=True)
            )
        else:
            activation = nn.Sequential(
                nn.LeakyReLU(inplace=True) if self.leaky else nn.ReLU(inplace=True)
            )

        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "activation1", activation),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "activation2", activation),
                ]
            )
        )



class UNetSmallCheckpoint(nn.Module): # Max Pooling, BatchNorm & LeakyReLU on demand
    def __init__(self, in_channels, out_channels, features, batchnorm=True, leaky=True, max_pool=True):
        super(UNetSmallCheckpoint, self).__init__()
        self.batchnorm = batchnorm
        self.leaky = leaky

        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)

        self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = self._block(features * 2, features, name="dec1")

        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        pool1 = self.pool1(enc1)
        enc2 = checkpoint(self.custom(self.encoder2),pool1)
        pool2 = self.pool2(enc2)
        enc3 = checkpoint(self.custom(self.encoder3),pool2)
        pool3 = self.pool3(enc3)
        enc4 = checkpoint(self.custom(self.encoder4),pool3)
        pool4 = self.pool4(enc4)

        bottleneck = checkpoint(self.custom(self.bottleneck),pool4)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = checkpoint(self.custom(self.decoder4),dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = checkpoint(self.custom(self.decoder3),dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = checkpoint(self.custom(self.decoder2),dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = checkpoint(self.custom(self.decoder1),dec1)
        return torch.sigmoid(self.conv(dec1))

    # define custom forward function, needed for checkpointing
    def custom(self, module):
        def custom_forward(*inputs):
            outputs = module(inputs[0])
            return outputs

        return custom_forward

    def _block(self, in_channels, features, name):
        if self.batchnorm:
            activation = nn.Sequential(
                nn.BatchNorm3d(num_features=features, momentum=(1-math.sqrt(0.9))),
                nn.LeakyReLU(inplace=True) if self.leaky else nn.ReLU(inplace=True)
            )
        else:
            activation = nn.Sequential(
                nn.LeakyReLU(inplace=True) if self.leaky else nn.ReLU(inplace=True)
            )

        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "activation1", activation),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "activation2", activation),
                ]
            )
        )


class UNetSmallSepMedDataCheckpoint(nn.Module):

    def __init__(self, in_channels, out_channels, features, batchnorm=True, leaky=True, max_pool=True):
        super(UNetSmallSepMedDataCheckpoint, self).__init__()
        self.batchnorm = batchnorm
        self.leaky = leaky
        self.imaging_mods = in_channels - 5
        self.pathways = 2

        self.encoder1_md = self._block(5, features, name="enc1_md")
        self.pool1_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder2_md = self._block(features, features * 2, name="enc2_md")
        self.pool2_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder3_md = self._block(features * 2, features * 4, name="enc3_md")
        self.pool3_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder4_md = self._block(features * 4, features * 8, name="enc4_md")
        self.pool4_md = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)


        self.encoder1 = self._block(self.imaging_mods, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)

        self.bottleneck = self._block(features * 8 * self.pathways, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self._block((features * 8) * (self.pathways+1), features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self._block((features * 4) * (self.pathways+1), features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self._block((features * 2) * (self.pathways+1), features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = self._block(features * (self.pathways+1), features, name="dec1")

        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        x_md = x[:,0:5,:,:,:]
        x_images = x[:,5:,:,:,:]

        enc1 = checkpoint(self.custom(self.encoder1),x_images)
        pool1 = self.pool1(enc1)
        enc2 = checkpoint(self.custom(self.encoder2),pool1)
        pool2 = self.pool2( enc2)
        enc3 = checkpoint(self.custom(self.encoder3),pool2)
        pool3 = self.pool3( enc3)
        enc4 = checkpoint(self.custom(self.encoder4),pool3)
        pool4 = self.pool4( enc4)

        enc1_md = checkpoint(self.custom(self.encoder1_md), x_md)
        pool1_md = self.pool1_md( enc1_md)
        enc2_md = checkpoint(self.custom(self.encoder2_md), pool1_md)
        pool2_md = self.pool2_md( enc2_md)
        enc3_md = checkpoint(self.custom(self.encoder3_md), pool2_md)
        pool3_md = self.pool3_md( enc3_md)
        enc4_md = checkpoint(self.custom(self.encoder4_md), pool3_md)
        pool4_md = self.pool4_md( enc4_md)


        dec4_inp = torch.cat([enc4, enc4_md], dim= 1)
        dec3_inp = torch.cat([enc3, enc3_md], dim=1)
        dec2_inp = torch.cat([enc2, enc2_md], dim=1)
        dec1_inp = torch.cat([enc1, enc1_md], dim=1)

        bottle_input = torch.cat([pool4, pool4_md], dim=1)




        bottleneck = checkpoint(self.custom(self.bottleneck),bottle_input)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, dec4_inp), dim=1)
        dec4 = checkpoint(self.custom(self.decoder4),dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, dec3_inp), dim=1)
        dec3 = checkpoint(self.custom(self.decoder3),dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, dec2_inp), dim=1)
        dec2 = checkpoint(self.custom(self.decoder2),dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, dec1_inp), dim=1)
        dec1 = checkpoint(self.custom(self.decoder1),dec1)
        return torch.sigmoid(self.conv(dec1))


    #define custom forward function, needed for checkpointing
    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward

    def _block(self, in_channels, features, name):
        if self.batchnorm:
            activation = nn.Sequential(
                nn.BatchNorm3d(num_features=features, momentum=(1-math.sqrt(0.9))),
                nn.LeakyReLU(inplace=True) if self.leaky else nn.ReLU(inplace=True)
            )
        else:
            activation = nn.Sequential(
                nn.LeakyReLU(inplace=True) if self.leaky else nn.ReLU(inplace=True)
            )

        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "activation1", activation),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "activation2", activation),
                ]
            )
        )





